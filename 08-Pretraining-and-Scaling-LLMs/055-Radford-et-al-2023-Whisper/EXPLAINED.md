# Whisper: Robust Speech Recognition via Large-Scale Weak Supervision, explained simply

**Paper:** Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey & Ilya Sutskever (OpenAI), *Robust Speech Recognition via Large-Scale Weak Supervision*, arXiv 2022 / ICML 2023.

**In one sentence:** take a plain encoder–decoder Transformer and train it to predict the transcripts of **680,000 hours** of audio from the internet, in many languages and for several tasks at once. The result is a speech recogniser that works **out of the box** (zero-shot) on almost any recording, and is far more robust than models trained on one clean dataset.

---

## 1. The big idea

### 1.1 The problem: superhuman on the benchmark, subhuman in the wild
- **On the benchmark:** by 2021, speech recognisers trained on **LibriSpeech** (audiobooks read aloud) reached **1.4% word error rate** on its test set, better than a human (5.8%).
- **In the wild:** the same models fail badly on phone calls, accents, meetings and noisy rooms.
- **The explanation:** a model trained on one dataset learns that dataset's quirks (microphone, reading style, vocabulary). Its test score measures **in-distribution** skill, while a human's score measures **out-of-distribution** skill, since humans never trained on LibriSpeech specifically.
- This is the same story as CLIP (paper 053) and ImageNet.

### 1.2 Two ways to scale speech data
1. **Self-supervised pre-training** (wav2vec 2.0) on huge amounts of **unlabelled** audio (up to 1,000,000 hours).
   - It learns a great audio **encoder** but no **decoder**: you still have to fine-tune on labelled data for every new setting.
   - That fine-tuning is exactly where dataset quirks sneak back in.
2. **Weakly supervised** training on audio with **imperfect transcripts found on the internet** (Whisper).
   - **Weak** means the transcripts are noisy and not checked by experts.
   - **The trade:** quality for quantity and diversity. 680k hours is about **680×** a typical academic dataset.

**Whisper's bet is option 2,** with a decoder trained end to end, so no fine-tuning is needed.

---

## 2. The data (Section 2.1)
**680,000 hours** of (audio, transcript) pairs:
- **438,000 h** of English speech with English transcripts;
- **117,000 h** of 96 other languages;
- **125,000 h** of **X → English translation** (audio in another language with an English transcript).

**Minimal processing:**
- The text is the **raw transcript**: punctuation, capitals and all, with no normalisation. The model must learn to output natural text.
- **Machine-generated transcripts are removed.** Many internet "transcripts" are the output of other speech recognisers. Training on those would teach Whisper their mistakes ("transcript-ese"). Detection heuristics:
  - text that is ALL CAPS or all lowercase;
  - text that never contains a comma.
- **Language check:** a language detector checks that the spoken language matches the transcript's language.
  - On a mismatch, the pair is dropped.
  - **Exception:** if the transcript is English, the pair becomes a *translation* example.
- **Duplicates:** fuzzy de-duplication of transcripts.
- **Segmentation:** audio is cut into **30-second segments** with the matching piece of transcript. Segments with **no speech** are kept (sub-sampled) to teach voice-activity detection.
- **A second filtering pass:** after training a first model, sources with high error rates were inspected by hand. Many turned out to be partial, misaligned or machine-made transcripts, and were removed.

---

## 3. From sound to numbers: the log-Mel spectrogram (Section 2.2)
**Raw audio** is a list of numbers: the air pressure measured 16,000 times per second (16 kHz).
- 30 seconds = 480,000 numbers.

**A spectrogram** asks, in each short slice of time, how much energy there is at each frequency:
- **Window:** 25 ms = **400 samples**.
- **Stride:** 10 ms = **160 samples**. Windows overlap, and there are **100 frames per second**, so 30 s gives **3000 frames**.
- **Spectrum:** for each window, the Fourier transform (STFT, with a Hann window) gives the energy at 201 frequencies between 0 and 8000 Hz. (8000 Hz is half the sampling rate, the highest frequency 16 kHz audio can represent.)

**The Mel scale** matches human hearing:
- We easily tell 200 Hz from 300 Hz, but not 7200 Hz from 7300 Hz.
- The Mel scale is roughly **linear below 1000 Hz and logarithmic above**. In the Slaney version Whisper uses:
  - mel = f / (200/3) below 1 kHz, so 1000 Hz = 15 mel;
  - above that, mel = 15 + 27 · ln(f/1000) / ln 6.4.
- **The filters:** 80 triangular filters, equally spaced on this scale, merge the 201 frequencies into **80 Mel channels**. There are fine channels for low frequencies and coarse ones for high frequencies.
- **The log:** energy spans a huge range, and loudness is perceived logarithmically, so Whisper takes **log₁₀**.

**Normalising:**
- Clip everything more than 8 below the maximum (8 log₁₀ units = **80 dB** of dynamic range).
- Then compute (x + 4)/4, so values lie roughly in [−1, 1].

**Result:** an 80 × 3000 "image" of the sound.

---

## 4. The model (Section 2.2, Table 1)
An **off-the-shelf encoder–decoder Transformer** (paper 018's architecture). They deliberately did not invent a new model, so the gains can be attributed to the data.

**Encoder:**
1. **Two 1-D convolutions**, kernel width 3, with GELU. The second has **stride 2**, halving time: 3000 frames become **1500 positions**, i.e. 20 ms each.
2. **Sinusoidal position encodings** (fixed sin/cos waves, as in the original Transformer).
3. **Pre-LayerNorm Transformer blocks** (LayerNorm before attention and before the MLP, as in GPT-2, paper 049).
4. A final LayerNorm.

**Decoder:**
- learned position embeddings, up to **448 tokens**;
- self-attention (causal) plus **cross-attention** to the encoder output, so every text token can look at all 1500 audio positions;
- **tied** input/output token embeddings, i.e. the same matrix (paper 049);
- GPT-2's **byte-level BPE** tokenizer. For multilingual models it is refit on multilingual text with the same vocabulary size, because GPT-2's BPE splits non-English words into too many pieces.

| Model | Layers | Width | Heads | Parameters |
|---|---|---|---|---|
| Tiny | 4 | 384 | 6 | 39M |
| Base | 6 | 512 | 8 | 74M |
| Small | 12 | 768 | 12 | 244M |
| Medium | 24 | 1024 | 16 | 769M |
| Large | 32 | 1280 | 20 | 1550M |

(Encoder and decoder have the same width and the same number of blocks.)

---

## 5. One model, many tasks: the token format (Section 2.3, Figure 1)
A full speech system needs more than recognising words:
- detecting whether anyone is speaking (**voice activity detection**);
- recognising **which language** is spoken;
- **translating**;
- giving **timestamps**.

**Whisper does all of these in one decoder by writing the task into the token sequence:**
```
[<|prev|> previous text]  <|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|> The quick brown fox <|1.52|> ...  <|endoftext|>
```
1. **Previous text** (optional): the transcript of the preceding audio, as context. It helps resolve ambiguous words. The loss is **not** applied to it.
2. **<|startoftranscript|>**.
3. **A language token**, one of 99 (e.g. <|en|>), predicted by the model. That is language identification. If there is no speech, the model predicts **<|nospeech|>** instead (voice activity detection).
4. **The task:** <|transcribe|> (same-language text) or <|translate|> (English text).
5. **<|notimestamps|>**, or else interleaved **timestamp tokens**.
   - Timestamps are times within the 30-second window, rounded to **20 ms**, giving 30/0.02 + 1 = **1501** time tokens.
   - **Example:** 1.234 s rounds to 1.24 s, which is time token number 62.
6. **The text**, then **<|endoftext|>**.

**Training:** the model is trained to predict every token after the context, including the language and the timestamps.
**At test time** you **force** the task tokens you want. For example, give <|startoftranscript|> <|fr|> <|translate|> and the model translates French speech into English.

---

## 6. Training (Section 2.4)
- **Optimiser:**
  - AdamW with gradient-norm clipping;
  - linear warm-up for 2048 updates, then linear decay to 0;
  - **batch 256** segments, **2²⁰ ≈ 1M updates** = only 2–3 passes over the data.
- **No data augmentation and no regularisation:** the diversity of the data does that job. (A later **Large-v2** trained 2.5× longer, adding SpecAugment, stochastic depth and BPE dropout.)
- **Speaker names:** many transcripts contain speakers' names, so the model learned to **guess names** it couldn't possibly know. A short fine-tune on transcripts without speaker labels removed this.

---

## 7. Measuring: WER and the normaliser (Section 3.2)
**Word error rate** = (substitutions + deletions + insertions) / words in the reference. It is computed with edit distance (paper 054 and our `edit_distance`).

**Example:** reference "the cat sat on the mat", hypothesis "the cat sat on mat".
- One deletion out of 6 words gives WER = **16.7%**.
- Because insertions count, WER can exceed 100%: a model that invents many extra words can make more errors than there are reference words.

**The problem for a zero-shot model:** every dataset has its own transcription style ("you're" vs "you are", "Mr." vs "mister", "10" vs "ten", [laughs]).
- A model that never saw the dataset's style is penalised for innocuous differences.
- **The fix:** Whisper's **text normaliser** (Appendix C) puts both texts into a standard form before computing WER: lowercase, remove punctuation and bracketed asides, standardise spellings, contractions and numbers.
- **The effect:** on some datasets this cut WER by up to **50%**.
- **The risk the authors flag:** the normaliser was developed while looking at Whisper's outputs. They checked it against an independent normaliser (FairSpeech); results were similar except on WSJ, CallHome and Switchboard.

---

## 8. Results

### 8.1 Robustness: the main finding (Figure 2)
**The comparison:** wav2vec 2.0 Large (trained on LibriSpeech) and zero-shot **Whisper Large V2** both score **2.7%** on LibriSpeech test-clean. Elsewhere:

| Dataset | wav2vec 2.0 | Whisper | Relative error reduction |
|---|---|---|---|
| LibriSpeech Clean | 2.7 | 2.7 | 0% |
| Artie | 24.5 | 6.2 | 74.7% |
| Common Voice | 29.9 | 9.0 | 69.9% |
| CHiME6 | 65.8 | 25.5 | 61.2% |
| Switchboard | 28.3 | 13.8 | 51.2% |
| AMI SDM1 | 67.6 | 36.4 | 46.2% |
| LibriSpeech Other | 6.2 | 5.2 | 16.1% |
| **Average (13 datasets, excluding LibriSpeech Clean)** | **29.3** | **12.8** | **55.2%** |

- **Relative error reduction** = (old − new)/old. For Artie: (24.5 − 6.2)/24.5 = 74.7%.
- **The headline:** the same benchmark score, but **55% fewer errors** on average away from LibriSpeech.
- **Compared with a human transcriber,** Whisper's robustness is close to human-like.

### 8.2 Noise (Figure 5)
- **The test:** white noise or pub noise (ambient noise plus chatter) added to LibriSpeech at a chosen **signal-to-noise ratio**:
  - SNR (dB) = 10·log₁₀(signal power / noise power);
  - 10 dB: the speech has 10× the noise's power; 0 dB: equal; −5 dB: the noise is 3.2× stronger.
- **Low noise:** LibriSpeech-trained models win, since they are in-distribution specialists.
- **As the noise grows:** they degrade faster than Whisper. Under pub noise below 10 dB SNR, Whisper is better.

### 8.3 More data helps, with diminishing returns (Table 6)
| Hours | English WER | Multilingual WER | X→En BLEU |
|---|---|---|---|
| 3,405 | 30.5 | 92.4 | 0.2 |
| 6,811 | 19.6 | 72.7 | 1.7 |
| 13,621 | 14.4 | 56.6 | 7.9 |
| 27,243 | 12.3 | 45.0 | 13.9 |
| 54,486 | 10.9 | 36.4 | 19.2 |
| 681,070 | 9.9 | 29.2 | 24.8 |

- **English:** WER improves fast up to ~13k hours, then slows. The last **12.5×** more data buys only **1 point**.
- **Per language** (Figure 3): log WER vs log hours of that language has r² = **0.83**, and **WER halves for every 16× more training data**.

### 8.4 Multitask and multilingual training don't hurt (if the model is big enough)
- **Small models:** joint training on all languages and tasks is slightly worse than English-only training at equal compute (negative transfer).
- **Large models:** it is **better**: they have room to share knowledge across languages and tasks.

### 8.5 Long audio needs decoding tricks (Section 4.5, Table 7)
Whisper sees 30 s at a time. For long recordings it transcribes a window, then **shifts the window** to the last predicted timestamp. Errors can snowball, so they added these heuristics:
- **Beam search** with 5 beams instead of greedy decoding. Greedy decoding more often gets stuck repeating itself.
- **Temperature fallback:**
  - start at T = 0 (greedy-like);
  - if the output looks bad, retry with T = 0.2, 0.4, … 1.0;
  - "bad" means the **average log-probability < −1** (the model is unsure) or the **gzip compression ratio > 2.4**.
  - Repetitive text ("thank you thank you thank you …") compresses extremely well, so a high ratio signals a repetition loop.
- **Voice activity detection:** a window is treated as silence only if **P(<|nospeech|>) > 0.6 and** the average log-prob < −1. The no-speech probability alone wasn't reliable.
- **Previous-text conditioning:** pass the last window's text as context (only when T < 0.5).
- **Initial timestamp constraint:** the first timestamp must lie in [0, 1] s. This stops the model from skipping the first words.

| Decoding | Average long-form WER |
|---|---|
| Greedy only | 11.0 |
| + Beam search | 10.6 |
| + Temperature fallback | 10.6 |
| + Voice activity detection | 10.2 |
| + Previous text | 10.0 |
| + Initial timestamp constraint | 10.0 |

### 8.6 Against professional transcribers (Section 3.9)
- **The test:** 25 recordings (Kincaid46) transcribed by 4 professional human services and 1 computer-assisted service.
- **The result:** Whisper's WER was close to theirs. The best (computer-assisted) service was only **1.15 points** better.

---

## 9. Limitations (Section 6)
- **Long-form errors:**
  - repeated loops;
  - skipped first or last words;
  - **hallucinated** text unrelated to the audio, a failure that a classic recogniser can't make in the same way.
- **Low-resource languages are weak.** Performance follows the training hours per language, and many languages have very little data.
- **The decoder is a language model.** It can produce fluent but **wrong** text, and its guesses look confident.
- **No fine-tuning studied:** they evaluated only zero-shot. Fine-tuning would likely help on specific domains.

---

## 10. Why it works (the intuition)
- **Diversity is the regulariser.** 680k hours from countless sources cover many microphones, rooms, accents, speaking styles and noise. No single dataset's quirks dominate, so the model can't rely on them.
- **Weak labels are fine at scale.** Individual transcripts are imperfect, but systematic errors (machine-generated transcripts) were filtered, and random errors average out.
- **One decoder for everything.** Writing the task as tokens lets one network share what it learns about speech across languages and tasks. It also removes the need for a fine-tuned decoder per dataset, which was where dataset quirks re-entered in the wav2vec approach.

---

## 11. What our code found
Everything below runs on a laptop CPU; the demo takes ~24 s. No real speech is used in the demo (see `experiments.py`).

**Exact checks (tests):**
- **The front end:** 30 s of audio gives an **80 × 3000** log-Mel; the values span at most 2 after clamping and scaling; a higher pitch lands in a higher Mel bin.
- **Parameter counts:** our implementation gives **37.2M / 71.8M / 240.6M / 762.4M / 1541.5M** for tiny … large. That is 1–5% below the paper's labels (39M / 74M / 244M / 769M / 1550M), which are rounded and may count things slightly differently (e.g. bias terms). The built tiny model matches our formula exactly.
- **The encoder halves time;** the decoder is causal.
- **The token format** (previous text masked, language/task/timestamp tokens, 20 ms rounding, 1501 time tokens, the no-speech sequence) is checked exactly.
- **The normaliser** turns "You're my favourite colour, Mr. Smith!" vs "you are my favorite color mister smith" from **WER > 50%** into **0%**.
- **The fallback rule** retries until the output is neither repetitive (gzip ratio > 2.4) nor unsure (log-prob < −1).
- **The silence rule** requires both conditions.
- **SNR mixing** is exact, and the RER values match Figure 2.

**The demo's toy "speech":**
- **Words:** each digit is a pair of tones, like a telephone keypad.
- **Two "languages":** they use different tone sets.
- **Clips:** 10% are silent.
- **The model:** a tiny Whisper (d = 64, 2+2 layers) trained in the exact multitask token format, predicting language, transcription, or <|nospeech|>.

| SNR | Clean-only model WER | Diverse-noise model WER |
|---|---|---|
| clean | **0.4%** | 12.4% |
| 20 dB | 257.6% | **5.1%** |
| 10 dB | 262.1% | **9.9%** |
| 0 dB | 274.4% | **20.7%** |
| −5 dB (never seen) | 277.7% | **36.3%** |

- **The clean-only specialist wins at home but collapses with any noise.** Its WER above 100% means it invents extra digits; the hallucination failure mode in miniature.
- **The diversely trained model degrades gracefully,** even at a noise level it never saw. That is the Figure 2 / Figure 5 story in miniature.
- **Honest oddities:**
  - The diverse model is *worse* on perfectly clean audio (12.4%) than at 20 dB (5.1%). Only 1/5 of its training data was clean, and in clean audio the silent gaps hit the log-Mel floor, which looks unusual to it.
  - The clean model got 500 training steps vs 900 to keep the demo short. In a dev run with 900 steps, it was just as fragile under noise (134% WER at 20 dB).
- **The diverse model's language/silence predictions** are 100% correct down to 0 dB.

**`experiments.py` (written, not run on this laptop)** trains on real LibriSpeech:
- **E1:** a clean-only vs a diverse (clean + other + noise) model, tested under white and babble noise from 40 to −10 dB;
- **E2:** WER vs hours;
- **E3:** transcription only vs + timestamps vs + no-speech;
- **E4:** long-form decoding: greedy, beam, fallback, previous text;
- **E5:** the effect of the normaliser;
- **E6:** the released OpenAI models (needs `pip install openai-whisper`), for comparison with the paper.

---

## 12. Check yourself

1. How many spectrogram frames are in 30 s of audio, and how many encoder positions?
<details><summary>Answer</summary>The stride is 10 ms, so there are 100 frames per second and 3000 frames for 30 s. The second convolution has stride 2, so the Transformer sees 1500 positions, each covering 20 ms. That 20 ms is also the timestamp resolution.</details>

2. Why use the Mel scale and a logarithm?
<details><summary>Answer</summary>Human pitch perception is roughly linear at low frequencies and logarithmic at high ones, so Mel channels spend resolution where it matters for speech. Loudness is perceived logarithmically, and energy spans many orders of magnitude, so log10 compresses it into a usable range (clamped to 80 dB).</details>

3. How does Whisper know whether to transcribe or translate?
<details><summary>Answer</summary>The task is a token in the decoder's prefix: <|transcribe|> or <|translate|>. At inference you force the prefix you want (e.g. <|startoftranscript|><|fr|><|translate|>), and the model continues accordingly.</details>

4. Why does Whisper remove machine-generated transcripts, but keep noisy and diverse audio?
<details><summary>Answer</summary>Diverse audio teaches robustness. Machine-generated transcripts carry systematic errors and a narrow style (no punctuation, all lowercase), so training on them would teach Whisper to imitate another recogniser's mistakes. Audio diversity helps; transcript errors that are systematic hurt.</details>

5. A model's WER is 2.7% on LibriSpeech and 24.5% on Artie; Whisper's are 2.7% and 6.2%. What is the relative error reduction on Artie, and what does it tell you?
<details><summary>Answer</summary>(24.5 − 6.2)/24.5 = 74.7%. The same in-distribution score can hide very different robustness: the LibriSpeech-trained model's 2.7% reflects dataset-specific skill that doesn't transfer.</details>

6. Why can WER exceed 100%?
<details><summary>Answer</summary>Insertions count as errors. A model that outputs many words not in the reference (hallucination) can make more errors than there are reference words. Our clean-only toy model reaches 258% this way under noise.</details>

7. What does a gzip compression ratio above 2.4 indicate, and what does Whisper do about it?
<details><summary>Answer</summary>Highly repetitive output (a decoding loop) compresses very well, so the ratio is high. Whisper then re-decodes that window at a higher temperature (0.2, 0.4, … 1.0) to escape the loop.</details>

8. Why is a text normaliser needed for fair zero-shot evaluation?
<details><summary>Answer</summary>Datasets transcribe in different styles ("you're" vs "you are", "10" vs "ten", punctuation, [laughs]). A zero-shot model hasn't seen a dataset's style, so it gets penalised for differences that don't change the meaning. Normalising both texts removes those differences (up to 50% WER reduction on some datasets).</details>

9. Going from 54k to 681k hours (12.5×) improved English WER by only 1 point. Give two possible explanations.
<details><summary>Answer</summary>(1) Saturation: performance is approaching the human level and label noise, so there is little left to gain. (2) The models are under-trained or too small to use that much data, so longer training or bigger models could unlock more. The paper says it can't yet tell which.</details>

10. In our toy, why might the diverse model do worse on perfectly clean audio than at 20 dB?
<details><summary>Answer</summary>It's a distribution effect: only 1/5 of its training clips were clean, and in clean audio the silent parts hit the log-Mel floor (max − 8), producing flat regions that never occur in noisy clips. Mildly noisy audio is closer to most of what it saw.</details>
