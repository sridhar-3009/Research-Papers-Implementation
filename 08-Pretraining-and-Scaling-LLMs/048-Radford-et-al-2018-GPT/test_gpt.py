import math

import torch
import torch.nn.functional as F

from gpt import (GPT, FineTuner, Specials, apply_bpe, avg_logprob, classification_input, count_params,
                 entailment_input, learn_bpe, multiple_choice_inputs, next_token_logprobs, optimizer, pad_batch,
                 similarity_inputs, transfer_layers, warmup_cosine, warmup_linear)


def small(vocab=30):
    return GPT(vocab, n_ctx=16, d=32, layers=2, heads=4, ff=64, dropout=0.0)


def test_size_matches_117M():
    with torch.device("meta"):
        m = GPT(40478)                                                             # 40,000 merges + base symbols
    assert abs(count_params(m) / 1e6 - 116.5) < 0.5


def test_output_layer_is_tied_and_causal():
    torch.manual_seed(0)
    g = small()
    ids = torch.randint(0, 30, (1, 8))
    h = g.hidden(ids)
    assert torch.allclose(g.logits(h), h @ g.tok.weight.T)                         # softmax(h W_e^T)
    ids2 = ids.clone(); ids2[0, 5] = (ids[0, 5] + 1) % 30
    assert torch.allclose(g.hidden(ids)[0, :5], g.hidden(ids2)[0, :5], atol=1e-6)   # no peeking ahead


def test_lm_loss_is_next_token_cross_entropy_and_masks_padding():
    torch.manual_seed(0)
    g = small()
    ids = torch.randint(1, 30, (2, 6))
    lg = g.logits(g.hidden(ids))
    manual = F.cross_entropy(lg[:, :-1].reshape(-1, 30), ids[:, 1:].reshape(-1))
    assert torch.allclose(g.lm_loss(ids), manual, atol=1e-6)
    mask = torch.ones(2, 6, dtype=torch.bool); mask[1, 3:] = False
    ce = F.cross_entropy(lg[:, :-1].reshape(-1, 30), ids[:, 1:].reshape(-1), reduction="none").view(2, 5)
    m = mask[:, 1:].float()
    assert torch.allclose(g.lm_loss(ids, mask), (ce * m).sum() / m.sum(), atol=1e-6)


def test_input_transformations_figure_1():
    sp = Specials(100, 101, 102)
    assert classification_input([1, 2], sp) == [100, 1, 2, 102]
    assert entailment_input([1], [2, 3], sp) == [100, 1, 101, 2, 3, 102]
    assert similarity_inputs([1], [2], sp) == [[100, 1, 101, 2, 102], [100, 2, 101, 1, 102]]
    assert multiple_choice_inputs([1], [[7], [8, 9]], sp) == [[100, 1, 101, 7, 102], [100, 1, 101, 8, 9, 102]]


def test_finetuner_reads_the_extract_token_and_combines_losses():
    torch.manual_seed(0)
    g = small()
    new = g.extend_vocab(3)
    assert new == [30, 31, 32] and g.tok.weight.shape[0] == 33
    sp = Specials(*new)
    f = FineTuner(g, 3).eval()
    ids, mask = pad_batch([classification_input([1, 2, 3], sp), classification_input([4], sp)])
    feats, h = f.features(ids, mask)
    assert torch.allclose(feats[1], h[1, 2]) and torch.allclose(feats[0], h[0, 4])  # the <e> positions
    y = torch.tensor([0, 2])
    l2 = F.cross_entropy(f(ids, mask), y)
    assert torch.allclose(f.loss(ids, mask, y, lam=0), l2, atol=1e-6)
    assert f.loss(ids, mask, y, lam=0.5) > l2                                        # + lambda * LM loss


def test_layer_transfer_and_zero_shot_scoring():
    torch.manual_seed(0)
    a, b = small(), small()
    transfer_layers(a, b, 1)
    assert torch.equal(a.blocks[0].ln1.weight, b.blocks[0].ln1.weight)
    assert torch.equal(a.tok.weight, b.tok.weight)
    assert not torch.equal(a.blocks[1].attn.in_proj_weight, b.blocks[1].attn.in_proj_weight)
    lp = next_token_logprobs(a, [1, 2, 3])
    assert abs(lp.exp().sum().item() - 1) < 1e-5
    seq = [1, 2, 3, 4]
    lg = F.log_softmax(a.logits(a.hidden(torch.tensor([seq])))[0], -1)
    assert abs(avg_logprob(a, [1, 2], [3, 4]) - (lg[1, 3] + lg[2, 4]).item() / 2) < 1e-5


def test_schedules_and_weight_decay_groups():
    assert warmup_cosine(0, 2000, 10000) == 0 and warmup_cosine(2000, 2000, 10000) == 1
    assert abs(warmup_cosine(10000, 2000, 10000)) < 1e-12
    assert warmup_linear(5, 10, 110) == 0.5 and warmup_linear(110, 10, 110) == 0
    opt = optimizer(small(), 1e-3)
    assert opt.param_groups[0]["weight_decay"] == 0.01 and opt.param_groups[1]["weight_decay"] == 0.0
    assert all(p.dim() < 2 for p in opt.param_groups[1]["params"])


def test_bpe_reused_from_034():
    merges = learn_bpe({"lower": 5, "lowest": 3, "low": 4}, 3)
    assert len(merges) == 3 and len(apply_bpe("lower", merges)) < len("lower")
