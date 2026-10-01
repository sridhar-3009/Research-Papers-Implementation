"""Tests for Grammar as a Foreign Language (Vinyals et al. 2015). A couple of seconds.

Run with:  python3 -m pytest -q
"""

import random

import torch

import grammar
from parser import (EOS, GO, PAD, Cell, Deep, LSTMA, Vocab, balance, batch_pairs, beam_search, clean_ptb, delinearize,
                    evalb, is_well_formed, linearize, parse_tree, to_string, words)

torch.manual_seed(0)
JOHN = "( (S (NP-SBJ (NNP John)) (VP (VBZ has) (NP (DT a) (NN dog))) (. .)) )"


def test_linearization_is_figure_2():
    t = clean_ptb(parse_tree(JOHN))
    assert " ".join(linearize(t, normalize_pos=False)) == "(S (NP NNP )NP (VP VBZ (NP DT NN )NP )VP . )S"
    assert " ".join(linearize(t)) == "(S (NP XX )NP (VP XX (NP XX XX )NP )VP XX )S"


def test_linearization_is_invertible_given_the_words():
    for t in grammar.corpus(50, seed=3):
        back = delinearize(linearize(t), words(t))
        assert words(back) == words(t)
        assert linearize(back) == linearize(t)
        assert evalb([t], [back])["F1"] == 100


def test_treebank_cleaning():
    t = clean_ptb(parse_tree("( (S (NP-SBJ-1 (-NONE- *T*)) (NP-SBJ=2 (PRP He)) (VP (VBD ran))) )"))
    assert to_string(t) == "(S (NP (PRP He)) (VP (VBD ran)))"


def test_malformed_outputs_are_balanced_at_the_ends():
    assert balance(["(S", "XX", "(VP", "XX"]) == ["(S", "XX", "(VP", "XX", ")VP", ")S"]
    assert balance([")NP", "XX", ")S"])[:2] == ["(S", "(NP"]
    assert is_well_formed(balance(["(S", ")NP", ")VP", "XX"]))
    assert not is_well_formed(["(S", "XX"]) and not is_well_formed([")S", "(S"])


def test_delinearize_keeps_every_word_even_with_the_wrong_number_of_terminals():
    sent = ["a", "b", "c"]
    for seq in (["(S", "XX", ")S"], ["(S", "XX", "XX", "XX", "XX", "XX", ")S"], ["(NP", "XX", "(VP", "XX"]):
        assert words(delinearize(seq, sent)) == sent


def test_evalb_ignores_punctuation_and_counts_labeled_spans():
    gold = clean_ptb(parse_tree(JOHN))
    # attach the object NP wrongly and mislabel: (S (NP John) (VP has) (NP a dog) .)
    pred = delinearize("(S (NP XX )NP (VP XX )VP (NP XX XX )NP XX )S".split(), words(gold))
    r = evalb([gold], [pred])
    # gold spans (no punctuation): S[0,4] NP[0,1] VP[1,4] NP[2,4]; pred: S[0,4] NP[0,1] VP[1,2] NP[2,4]
    assert abs(r["P"] - 75) < 1e-9 and abs(r["R"] - 75) < 1e-9
    prt = parse_tree("(S (VP (VB give) (PRT (RP up))))")
    advp = parse_tree("(S (VP (VB give) (ADVP (RP up))))")
    assert evalb([prt], [advp])["F1"] == 100                                       # EVALB: ADVP = PRT


def test_lstm_cell_is_section_2():
    c = Cell(3, 4)
    x, h, m = torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 4)
    i, i2, f, o = (c.Wx(x) + c.Wh(h)).chunk(4, -1)
    m2 = m * torch.sigmoid(f) + torch.sigmoid(i) * torch.tanh(i2)
    h3, m3 = c(x, (h, m))
    assert torch.allclose(m3, m2) and torch.allclose(h3, m2 * torch.sigmoid(o))   # h = m o, no tanh


def test_dropout_only_between_layers():
    d = Deep(3, 4, layers=1, dropout=0.9, cell_tanh=False).train()
    z = torch.zeros(2, 4)
    x = torch.randn(2, 3)
    assert torch.equal(d(x, [(z, z)])[0], d(x, [(z, z)])[0])                       # one layer: nothing to drop
    d3 = Deep(3, 4, layers=3, dropout=0.9, cell_tanh=False).train()
    assert not torch.equal(d3(x, [(z, z)] * 3)[0], d3(x, [(z, z)] * 3)[0])


def test_attention_uses_the_current_decoder_state_and_ignores_padding():
    m = LSTMA(20, 15, d=8, emb=6, layers=2)
    src = torch.tensor([[4, 5], [6, 7], [8, PAD]])
    S = m.start(src)
    c1, a1 = m.attend(torch.randn(2, 8), S["H"], S["W1H"], S["mask"])
    c2, a2 = m.attend(torch.randn(2, 8), S["H"], S["W1H"], S["mask"])
    assert torch.allclose(a1.sum(0), torch.ones(2)) and a1[2, 1] == 0
    assert not torch.allclose(a1, a2)                                               # u_ti depends on d_t
    assert m.dec.cells[0].Wx.in_features == 6 + 16                                  # [d ; d'] fed to the next step


def test_padding_does_not_change_log_prob():
    m = LSTMA(20, 15, d=8, emb=6, layers=2)
    a = m.log_prob(*batch_pairs([([4, 5, 6], [7, 8])]))
    src, tgt = batch_pairs([([4, 5, 6], [7, 8]), ([4, 5, 6, 9, 10], [7, 8, 9, 10])])
    assert torch.allclose(a, m.log_prob(src, tgt)[:1], atol=1e-5)


def test_input_is_reversed_but_not_the_tree():
    src, tgt = batch_pairs([([4, 5, 6], [7, 8])])
    assert src[:, 0].tolist() == [6, 5, 4] and tgt[:, 0].tolist() == [7, 8, EOS]


def test_greedy_and_ensemble_of_copies():
    m = LSTMA(20, 15, d=8, emb=6, layers=2).eval()
    src = torch.tensor([[5], [6], [7]])
    _, best, _ = beam_search(m, src, beam=1, max_len=6)
    S, y, greedy = m.start(src), torch.tensor([GO]), []
    for _ in range(6):
        lp, _, S = m.step(y, S)
        y = lp.argmax(-1)
        if y.item() == EOS:
            break
        greedy.append(y.item())
    assert best == greedy
    assert beam_search([m, m], src, beam=3, max_len=6)[1] == beam_search(m, src, beam=3, max_len=6)[1]


def test_lstm_with_attention_learns_to_parse_the_toy_grammar():
    torch.manual_seed(0)
    train, test = grammar.corpus(1500, 0, max_words=7), grammar.corpus(30, 99, max_words=7)
    wv = Vocab([w for t in train for w in words(t)])
    lv = Vocab([s for t in train for s in linearize(t)])
    enc = lambda t: (wv.encode(words(t)), lv.encode(linearize(t)))
    pairs, rng = [enc(t) for t in train], random.Random(0)
    m = LSTMA(len(wv), len(lv), d=32, emb=16, layers=1)
    opt = torch.optim.Adam(m.parameters(), 5e-3)
    for _ in range(200):
        src, tgt = batch_pairs(rng.sample(pairs, 32))
        loss = -m.log_prob(src, tgt).sum() / (tgt != PAD).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    preds = [delinearize([lv.itos[s] for s in beam_search(m, batch_pairs([enc(t)])[0], beam=1)[1]], words(t))
             for t in test]
    assert evalb(test, preds)["F1"] > 90
