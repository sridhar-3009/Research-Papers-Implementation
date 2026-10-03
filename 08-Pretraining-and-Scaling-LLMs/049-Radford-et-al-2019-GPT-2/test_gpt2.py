import math

import torch

from gpt2 import (GPT2, SIZES, BloomFilter, ByteBPE, cloze_choice, lambada_predict, model_size, ngrams, normalize,
                  overlap_fraction, per_unit, pretokenize, qa_prompt, tldr_prompt, translation_prompt)

TEXT = "the dog ran. the dog barked! a dog? dogs and cats are great. " * 30


def test_pretokenizer_keeps_categories_apart():
    assert pretokenize("I'm 25 dogs. dog?") == ["I", "'m", " 25", " dogs", ".", " dog", "?"]


def test_byte_bpe_is_invertible_on_any_unicode_and_never_merges_across_categories():
    bpe = ByteBPE(TEXT, 60)
    for s in ["the dog barked", "naïve café 🐶 — 漢字", "\t\n  spaces  ", ""]:
        assert bpe.decode(bpe.encode(s)) == s                                      # any string, exactly
    for tok in bpe.vocab.values():
        if len(tok) > 1:
            s = tok.decode("utf-8", errors="ignore").strip()
            assert not (any(c.isalpha() for c in s) and any(c in ".!?" for c in s))   # no 'dog.' tokens
    assert len(bpe.encode("the dog")) < len("the dog".encode())                    # merges compress


def test_model_sizes_table_2():
    counts = {k: model_size(*v) for k, v in SIZES.items()}
    with torch.device("meta"):
        m = GPT2(d=768, layers=12, heads=12)
    assert sum(p.numel() for p in m.parameters()) == counts["117M"]
    assert [round(c / 1e6) for c in counts.values()] == [124, 355, 774, 1558]      # the true counts of the 4 sizes


def test_pre_ln_final_ln_and_scaled_residual_init():
    torch.manual_seed(0)
    m = GPT2(vocab=100, n_ctx=16, d=64, layers=8, heads=4)
    assert abs(m.blocks[0].proj.weight.std().item() - 0.02 / math.sqrt(16)) < 0.001  # 0.02 / sqrt(2 * layers)
    assert abs(m.blocks[0].fc.weight.std().item() - 0.02) < 0.002                    # non-residual layers unscaled
    ids = torch.randint(0, 100, (1, 10))
    ids2 = ids.clone(); ids2[0, 7] = (ids[0, 7] + 1) % 100
    assert torch.allclose(m(ids)[0, :7], m(ids2)[0, :7], atol=1e-5)                  # causal
    assert isinstance(m.ln_f, torch.nn.LayerNorm)


def test_generation_top_k_and_sequence_logprob():
    torch.manual_seed(0)
    m = GPT2(vocab=20, n_ctx=16, d=32, layers=2, heads=4).eval()
    lg = m(torch.tensor([[1, 2, 3]]))[0, -1]
    top2 = set(lg.topk(2).indices.tolist())
    for _ in range(20):
        assert m.generate([1, 2, 3], 1, top_k=2)[0] in top2
    assert m.generate([1, 2, 3], 1, greedy=True)[0] == int(lg.argmax())
    lp = torch.log_softmax(m(torch.tensor([[1, 2, 3, 4]]))[0], -1)
    assert abs(m.sequence_logprob([1, 2, 3, 4]) - (lp[0, 2] + lp[1, 3] + lp[2, 4]).item()) < 1e-4


def test_per_unit_conversions():
    r = per_unit(total_nll_nats=100 * math.log(8), n_units=100)
    assert abs(r["perplexity"] - 8) < 1e-9 and abs(r["bits per unit"] - 3) < 1e-9


def test_lambada_stop_word_filter_and_cloze():
    cands = [("the", -0.5), ("mountain", -1.0), ("it", -1.2)]
    assert lambada_predict(cands, stop_filter=False) == "the"
    assert lambada_predict(cands, stop_filter=True) == "mountain"
    score = lambda s: len(s)                                                      # a toy score: longer is better
    assert cloze_choice(score, "I saw a ", ["cat", "elephant"], " today.") == "elephant"


def test_prompts():
    assert translation_prompt([("hello", "bonjour")], "cat") == "hello = bonjour\ncat ="
    assert tldr_prompt("Long article.").endswith("TL;DR:")
    assert qa_prompt([("2+2?", "4")], "3+3?") == "Q: 2+2?\nA: 4\nQ: 3+3?\nA:"


def test_bloom_filter_overlap():
    bf = BloomFilter(m=2 ** 16, k=4)
    train = "The quick brown fox jumps over the lazy dog near the old river bank today"
    for g in ngrams(normalize(train).split()):
        bf.add(g)
    assert normalize("The QUICK, brown fox!") == "the quick brown fox"
    assert overlap_fraction(bf, train) == 1.0                                       # no false negatives
    assert overlap_fraction(bf, "completely different words appear in this sentence about other things") == 0.0
    assert bf.false_positive_rate() < 1e-10
