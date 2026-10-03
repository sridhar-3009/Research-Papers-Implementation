# The code, explained simply

How the code in this folder implements Whisper (Radford et al. 2022).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `whisper.py` | log-Mel front end (Slaney Mel filters, STFT, clamp and scale), noise at a chosen SNR, the encoder–decoder (conv stem, sinusoidal/learned positions, pre-LN blocks with cross-attention, tied embeddings), Table 1 parameter counts, the multitask special tokens and sequence builder, WER, the text normaliser, the decoding heuristics (compression ratio, fallback, silence rule), greedy and beam decoding, the paper's numbers |
| `experiments.py` | LibriSpeech: clean vs diverse training under noise, data scaling, multitask ablation, long-form heuristics, normaliser, released models (heavy, not run here) |
| `demo.py` | front end and sizes, the token format, a toy keypad-tone "speech" model (clean vs diverse training under noise), WER normalisation, decoding heuristics, the paper's numbers (~24 seconds) |
| `test_whisper.py` | 9 quick tests (~0.6 seconds) |

**Run it** (from `08-Pretraining-and-Scaling-LLMs/055-Radford-et-al-2023-Whisper`):
```
python3 -m pytest -q             # ~0.6 seconds
python3 demo.py                  # ~24 seconds
python3 experiments.py --quick
```

**Note:** our file is called `whisper.py`, which shadows the `openai-whisper` pip package inside this folder. `experiments.py` E6 removes this folder from `sys.path` before importing the real package.

---

## 2. `whisper.py`

### Front end
| Function | What it does |
|---|---|
| `hz_to_mel`, `mel_to_hz` | Slaney scale: linear below 1 kHz (f / 66.67), logarithmic above |
| `mel_filters()` | 80 triangular filters over 201 FFT bins, area-normalised |
| `log_mel(audio, n_mels)` | Hann-window STFT (400 / 160), power, Mel, log10, clamp to max − 8, (x+4)/4; drops the last frame, as Whisper does |
| `pad_or_trim` | pads or cuts audio to 30 s |
| `add_noise(audio, noise, snr_db)` | scales the noise so that 10·log10(P_signal / P_noise) equals snr_db |

### Model
- **`AudioEncoder`:**
  1. Conv1d(n_mels→d, 3) + GELU;
  2. Conv1d(d→d, 3, stride 2) + GELU;
  3. add sinusoids (a buffer, not trained);
  4. pre-LN `Block`s;
  5. a final LN.
- **`TextDecoder`:**
  - token embedding + learned positions;
  - `Block`s with a causal mask and cross-attention to the encoder output;
  - final LN;
  - logits = h · Eᵀ (tied).
- **`Whisper(vocab, n_mels, n_audio_ctx, n_text_ctx, d, heads, layers)`:** `forward(mel, ids)` returns logits.
- **`param_count(L, d)`:** the exact count for this architecture. `SIZES` and `PAPER_PARAMS` hold Table 1.

### Tokens
- **`SpecialTokens(n_text, languages)`:**
  - ids for endoftext, startoftranscript, one per language, translate, transcribe, nospeech, notimestamps and prev;
  - then 1501 timestamp tokens;
  - `timestamp(seconds)` rounds to 20 ms.
- **`build_sequence(sp, language, task, segments, timestamps, prev)`:**
  - returns (tokens, loss mask);
  - the previous text and <|startoftranscript|> are context (mask 0), everything else is predicted;
  - with no segments it returns <|nospeech|>.

### Metrics and decoding rules
| Function | What it does |
|---|---|
| `edit_distance`, `wer` | word-level Levenshtein distance / reference length |
| `normalize` | lowercase; drop [..] (..) <..> asides and fillers; expand contractions; standardise spellings and small numbers; strip punctuation |
| `relative_error_reduction(a, b)` | (a − b)/a |
| `compression_ratio(text)` | bytes / gzip bytes |
| `needs_fallback(lp, text)` | lp < −1 or ratio > 2.4 |
| `is_silence(p_ns, lp)` | p_ns > 0.6 and lp < −1 |
| `decode_with_fallback(fn)` | tries T = 0, 0.2, … 1.0 until `needs_fallback` is false |
| `greedy_decode` | batched; greedy or temperature sampling; returns ids and the mean log-prob |
| `beam_search` | one example; sum-log-prob beams; returns the best finished hypothesis by mean log-prob |

### The paper's numbers
`TABLE_6` (data scaling), `FIGURE_2` (wav2vec 2.0 vs Whisper), `TABLE_7` (long-form heuristics).

---

## 3. `experiments.py`

**Data and text:**
- LibriSpeech via `torchaudio.datasets.LIBRISPEECH` (downloads into `data/`);
- character tokens (`CHARS`) plus the special tokens;
- **babble** noise = the sum of 5 other utterances.

**Pieces:**
- **`make_examples`:** optional noise at random SNR (−5 … 40 dB, white or babble), optional timestamps, optional replacement by silence (no-speech examples).
- **`train`:** AdamW, gradient clipping, linear warm-up then linear decay to 0.
- **`transcribe`:** batched greedy decoding with the forced prefix <|sot|><|en|><|transcribe|><|notimestamps|>.
- **`corpus_wer`:** pooled WER, normalised or raw.

| Function | Reproduces |
|---|---|
| `e1` | clean-only vs diverse training; WER on test-clean, test-other, white/babble noise at 7 SNRs (Figure 5's idea) |
| `e2` | 10 / 25 / 50 / 100% of the hours; log-log slope (Table 6's idea) |
| `e3` | transcription only, + timestamps, + no-speech segments |
| `e4` | whole chapters, fixed 30 s windows: greedy, beam 5, + temperature fallback, + previous-text conditioning (Table 7) |
| `e5` | raw vs normalised WER for the E1 models and a released model |
| `e6` | released `tiny.en` / `base.en` / `small.en`: test-clean, test-other, babble at 10 and 0 dB |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_log_mel_shape_and_range` | 80 × 3000 for 30 s; range ≤ 2 after the clamp |
| `test_mel_filters_cover_the_spectrum_and_tone_lands_in_right_bin` | Mel ↔ Hz round trip; 300 Hz lands below 3000 Hz |
| `test_parameter_counts_match_table_1_and_the_built_model` | the formula equals the built tiny model; within 5% of every Table 1 label |
| `test_encoder_halves_time_and_decoder_is_causal` | 40 frames become 20 positions; future tokens don't change past logits |
| `test_multitask_format_and_timestamps` | 20 ms rounding, 1501 timestamps, prev masked, translate/notimestamps/nospeech sequences |
| `test_wer_and_normaliser` | edit distance; WER > 0.5 raw, 0 after normalising |
| `test_decoding_heuristics` | a looping text triggers fallback; T steps 0 → 0.2 → 0.4; the silence rule needs both conditions |
| `test_snr_and_rer` | the mixed signal has exactly 10 dB SNR; Figure 2's RERs |
| `test_greedy_and_beam_run` | shapes; log-probs ≤ 0 |

---

## 5. Try it yourself

1. In `demo.py`, train the diverse model on {clean, 20 dB} only. How does it do at 0 and −5 dB?
2. Add `<|translate|>` examples to the toy: for language B, make the target the digits *in reverse order* (a made-up "translation"). Does one model learn both tasks?
3. Turn on timestamps in the toy (`timestamps=True`) and check how close the predicted start times are to the truth (they are rounded to 20 ms).
4. Feed `greedy_decode` a silent clip at increasing noise levels and record P(<|nospeech|>). Where does `is_silence` start to fire?
5. Extend `normalize` with written-out numbers above twelve ("twenty one" → "21") and check its effect on `wer`.
