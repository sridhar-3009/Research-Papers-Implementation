# The code, explained simply

How the code in this folder implements the evaluation methods of the GPT-4o System Card (OpenAI 2024).
Read [EXPLAINED.md](EXPLAINED.md) first.

GPT-4o itself is not available, so the code implements:
- the **decision rules**;
- the **metrics**;
- a toy **voice-output classifier**;
- a simulated **TTS evaluation pipeline**.

---

## 1. The files

| File | What it is |
|---|---|
| `gpt4o.py` | Preparedness rules (overall = max, deploy/continue thresholds), the Apollo rating rule, safe-behaviour / not-unsafe / not-over-refuse metrics, cons@k, per-attempt rate from pass@k, the two-proportion z-test, synthetic voices, a spectral speaker embedding, the streaming voice check, precision/recall, the card's numbers |
| `experiments.py` | LibriSpeech speaker verification, a real TTS → ASR pipeline, disparate WER by speaker sex, cons@k vs pass@k on a small model (heavy, not run here) |
| `demo.py` | the scorecard, the voice classifier trade-off and streaming block, the TTS pipeline, safe-behaviour tables, pass@k/cons@k, a significance test, reported numbers (~0.4 seconds) |
| `test_gpt4o.py` | 5 quick tests (~0.1 seconds) |

**Run it** (from `08-Pretraining-and-Scaling-LLMs/058-OpenAI-2024-GPT-4o-System-Card`):
```
python3 -m pytest -q             # ~0.1 seconds
python3 demo.py                  # ~0.4 seconds
python3 experiments.py --quick
```

---

## 2. `gpt4o.py`

### Decision rules
| Function | Rule |
|---|---|
| `overall_risk(scorecard)` | the highest of low < medium < high < critical |
| `can_deploy` | overall ≤ medium |
| `can_continue_development` | overall ≤ high |
| `apollo_rating(pass_rates)` | ●●● if ≥ 50% on hard, ●●○ on medium, ●○○ on easy, else ○○○ |

### Metrics
| Function | Meaning |
|---|---|
| `safe_behaviour_accuracy(records)` | (accuracy on should-refuse items, accuracy on should-comply items) |
| `not_unsafe_and_not_over_refuse(records)` | the fraction of outputs not unsafe; on benign prompts, the fraction not refused |
| `consensus_at_k(answers, truth)` | the majority answer is right (ties go to the first seen) |
| `per_attempt_rate(pass_at_k, k)` | 1 − (1 − pass@k)^(1/k) |
| `two_proportion_z(a, n_a, b, n_b)` | z and two-sided p-value with the pooled proportion |
| `effect_size_ratio` | AI effect / human effect |

### Toy voice-output classifier
- **`synth_voice(pitch, formants, ...)`:**
  - harmonics of a pitch, shaped by formant resonances;
  - vibrato, amplitude changes and a little noise;
  - a per-clip **drift** of ~3% in pitch and formants, so clips of the same voice vary.
- **`voice_embedding(audio)`:** the time-averaged log energy in 40 log-spaced frequency bands, mean-removed and unit-normalised. It captures timbre, ignores timing, and is a stand-in for a trained speaker encoder.
- **`streaming_voice_check(audio, approved, threshold, chunk)`:** scores each 0.5 s chunk as it arrives and returns the first chunk below the threshold. That is where the deployed system would block the output.
- **`precision_recall(flags, truth)`:** precision and recall of the alarms.

### The card's numbers
`TABLE_2_VOICE_CLASSIFIER`, `TABLE_3_SPEAKER_ID`, `TABLE_4_UGI_STA`, `TABLE_5_TEXT_VS_AUDIO`, `CTF_PASS_AT_10`, `PERSUASION`, `AUTONOMY`, `MEDQA_0SHOT`.

---

## 3. `experiments.py`

| Function | What it does |
|---|---|
| `e1` | trains `SpeakerEncoder` (log-Mel from paper 055 → convs → mean+std pooling → 128-d, softmax over speakers) on train-clean-100; enrols held-out speakers from 3 clips; compares EER, the recall-1.0 threshold, the per-chunk false-alarm rate and conversation cut-off probabilities with the band embedding |
| `e2` | speaks prompts with macOS `say` (or pyttsx3); adds noise or echo; transcribes with released Whisper; checks whether a keyword filter makes the same decision on the transcript as on the text, plus WER |
| `e3` | released Whisper on test-clean; WER by speaker sex from SPEAKERS.TXT; a speaker-level bootstrap CI for the gap and a word-level z-test |
| `e4` | a small LLaMA-style model trained on 3-digit addition; single sample vs cons@k vs pass@k for k up to 32 |

`released_whisper` removes this folder and paper 055's folder from `sys.path`, so `import whisper` finds the pip package rather than our `whisper.py`.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_preparedness_rules` | the GPT-4o scorecard is medium and deployable; high blocks deployment; critical blocks development |
| `test_apollo_rating_rule` | all four ratings |
| `test_eval_metrics` | the safe-behaviour pair, not-unsafe / not-over-refuse, cons@k, the pass@k ↔ per-attempt round trip |
| `test_two_proportion_z` | z = 0 for equal groups; tiny p for 90% vs 60% |
| `test_voice_embedding_separates_voices_and_streaming_blocks_switch` | same-voice clips score above other-voice clips; unit norm; the block happens exactly at the switch; precision/recall |

---

## 5. Try it yourself

1. In `demo.py`, move the "similar-sounding voice" formants closer to (or farther from) the approved voice. How does the threshold needed for recall 1.0 change, and what does it cost in false alarms?
2. Make the chunks 1 second instead of 0.5. Are the embeddings more stable? What happens to blocking latency?
3. In the TTS simulation, let garbled prompts make the model refuse *more* (0.4 instead of 0.25). Which of the two metrics moves?
4. Use `two_proportion_z` to find the smallest test-set size at which a 1-point accuracy gap (80% vs 79%) becomes significant at p < 0.05.
5. Simulate a question where the right answer is chosen 30% of the time but one wrong answer 40%. Compare single sample, cons@10 and pass@10.
