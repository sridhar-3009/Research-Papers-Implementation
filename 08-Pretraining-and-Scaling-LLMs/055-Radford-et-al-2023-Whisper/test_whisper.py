import math

import numpy as np
import torch

from whisper import (FIGURE_2, PAPER_PARAMS, SIZES, SpecialTokens, Whisper, add_noise, beam_search, build_sequence,
                     compression_ratio, decode_with_fallback, edit_distance, greedy_decode, hz_to_mel, is_silence,
                     log_mel, mel_filters, mel_to_hz, normalize, param_count, relative_error_reduction, wer)


def test_log_mel_shape_and_range():
    torch.manual_seed(0)
    mel = log_mel(torch.randn(16000 * 30) * 0.1)
    assert mel.shape == (80, 3000)                                                # 30 s at a 10 ms stride
    assert mel.max() <= 1.0 + 1e-6 and mel.max() - mel.min() <= 2.0 + 1e-6        # clamp to max - 8 in log10, / 4


def test_mel_filters_cover_the_spectrum_and_tone_lands_in_right_bin():
    W = mel_filters()
    assert W.shape == (80, 201) and (W >= 0).all()
    assert np.allclose(mel_to_hz(hz_to_mel([0, 500, 1000, 4000])), [0, 500, 1000, 4000])
    t = torch.arange(16000) / 16000
    lo = log_mel(torch.sin(2 * math.pi * 300 * t)).mean(1).argmax()
    hi = log_mel(torch.sin(2 * math.pi * 3000 * t)).mean(1).argmax()
    assert lo < hi                                                                 # higher pitch -> higher mel bin


def test_parameter_counts_match_table_1_and_the_built_model():
    m = Whisper()                                                                 # tiny
    assert sum(p.numel() for p in m.parameters()) == param_count(4, 384)
    for name, (L, d, _) in SIZES.items():
        assert abs(param_count(L, d) / PAPER_PARAMS[name] - 1) < 0.05             # paper's rounded labels


def test_encoder_halves_time_and_decoder_is_causal():
    torch.manual_seed(0)
    m = Whisper(vocab=50, n_mels=8, n_audio_ctx=20, n_text_ctx=10, d=16, heads=2, layers=1).eval()
    mel = torch.randn(1, 8, 40)
    assert m.encoder(mel).shape == (1, 20, 16)
    ids = torch.randint(0, 50, (1, 6))
    ids2 = ids.clone(); ids2[0, 5] = (ids[0, 5] + 1) % 50
    assert torch.allclose(m(mel, ids)[0, :5], m(mel, ids2)[0, :5], atol=1e-5)


def test_multitask_format_and_timestamps():
    sp = SpecialTokens(100, ["en", "fr"])
    assert sp.timestamp(1.234) == sp.ts0 + 62 and abs(sp.time_of(sp.timestamp(1.234)) - 1.24) < 1e-9
    assert sp.n_ts == 1501
    toks, mask = build_sequence(sp, "fr", "transcribe", [(0.0, 1.0, [5, 6]), (1.2, 2.0, [7])], prev=[1, 2])
    assert toks[:4] == [sp.id["prev"], 1, 2, sp.id["startoftranscript"]] and mask[:4] == [0, 0, 0, 0]
    assert toks[4:6] == [sp.id["lang:fr"], sp.id["transcribe"]]
    assert toks[6:10] == [sp.timestamp(0.0), 5, 6, sp.timestamp(1.0)] and toks[-1] == sp.id["endoftext"]
    assert all(mask[4:])
    t2, _ = build_sequence(sp, "en", "translate", [(0, 1, [9])], timestamps=False)
    assert t2 == [sp.id["startoftranscript"], sp.id["lang:en"], sp.id["translate"], sp.id["notimestamps"], 9, sp.id["endoftext"]]
    assert build_sequence(sp, "en", "transcribe", [])[0] == [sp.id["startoftranscript"], sp.id["nospeech"], sp.id["endoftext"]]


def test_wer_and_normaliser():
    assert edit_distance("a b c".split(), "a x c d".split()) == 2
    assert wer("the cat sat", "the cat sat") == 0 and abs(wer("the cat sat", "the hat") - 2 / 3) < 1e-12
    ref, hyp = "You're my favourite colour, Mr. Smith!", "you are my favorite color mister smith"
    assert wer(ref, hyp) > 0.5 and wer(normalize(ref), normalize(hyp)) == 0
    assert normalize("Um, I have [laughs] two cats.") == "i have 2 cats"


def test_decoding_heuristics():
    loop = "the the the the the the the the the the the the the the the the the the the the"
    assert compression_ratio(loop) > 2.4 and compression_ratio("a quick brown fox jumps over the lazy dog") < 2.4
    calls = []

    def dec(T):
        calls.append(T)
        return (loop, -0.2) if T < 0.4 else ("a normal sentence here", -0.5)
    text, lp, T = decode_with_fallback(dec)
    assert calls == [0.0, 0.2, 0.4] and T == 0.4 and text == "a normal sentence here"
    assert is_silence(0.7, -1.5) and not is_silence(0.7, -0.3) and not is_silence(0.3, -1.5)


def test_snr_and_rer():
    torch.manual_seed(0)
    s, n = torch.randn(16000), torch.randn(16000) * 3
    mixed = add_noise(s, n, 10)
    assert abs(10 * math.log10(s.pow(2).mean() / (mixed - s).pow(2).mean()) - 10) < 1e-3
    assert abs(relative_error_reduction(*FIGURE_2["Artie"]) - 0.747) < 0.001
    assert abs(relative_error_reduction(*FIGURE_2["LibriSpeech Other"]) - 0.161) < 0.001


def test_greedy_and_beam_run():
    torch.manual_seed(0)
    m = Whisper(vocab=12, n_mels=8, n_audio_ctx=10, n_text_ctx=12, d=16, heads=2, layers=1).eval()
    mel, prefix = torch.randn(2, 8, 20), torch.tensor([[10], [10]])
    out, lp = greedy_decode(m, mel, prefix, eot=11, max_len=5)
    assert out.shape[0] == 2 and out.shape[1] <= 5 and (lp <= 0).all()
    seq, score = beam_search(m, mel, prefix, eot=11, beams=3, max_len=5)
    assert len(seq) <= 5 and score <= 0
