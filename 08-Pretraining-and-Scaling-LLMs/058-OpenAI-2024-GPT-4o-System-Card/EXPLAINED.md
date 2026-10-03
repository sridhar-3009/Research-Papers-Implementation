# GPT-4o System Card, explained simply

**Document:** OpenAI, *GPT-4o System Card*, August 8, 2024 (arXiv 2410.21276).

**In one sentence:** before releasing GPT-4o, a model that **listens and talks** directly, OpenAI listed what could go wrong (copying someone's voice, identifying people by voice, guessing private traits, harmful speech, persuasion, cyber and bio misuse, autonomy), **measured** each risk, added **mitigations**, and scored the model on its **Preparedness Framework**: overall risk **medium** (because of persuasion), so it could be deployed.

---

## 1. What a system card is
- **A research paper** says "here is a new method and how well it works".
- **A system card** says "here is a product-level model, here are the risks we looked for, how we measured them, what we did about them, and what is left".
- **Paper 057** (GPT-4) had one as an appendix. **This document is entirely one.**
- **What it is useful for:** learning *how safety evaluation is done*: turning vague worries into numbers, choosing metrics that capture both failure directions, and making deploy / don't-deploy decisions by explicit rules.

---

## 2. The model
- **GPT-4o** ("o" for *omni*): one autoregressive network trained **end-to-end** on text, images, audio and video.
- **Inputs:** any mix of text, audio, image and video. **Outputs:** text, audio or images.
- **Why end-to-end matters:**
  - **The old voice mode chained three models** (background from OpenAI's GPT-4o announcement, not from the card): speech → text (paper 055, Whisper), then text → text (GPT-4), then text → speech. Tone, laughter and background sounds were lost at the first step.
  - **One model** hears and speaks directly, so it can respond in as little as **232 ms** (320 ms on average), about as fast as a human in conversation.
- **Text quality:** it matches GPT-4 Turbo on English text and code, and is better in other languages.
- **Data:** up to October 2023, from public web data, code and maths, multimodal data, and licensed data. Filtering removed child sexual abuse material, hateful content, violence, CBRN information, personal information, and images whose owners opted out.
- **Not disclosed:** size, architecture, compute.

---

## 3. Finding risks: red teaming (Section 3.1)
- **More than 100 external red teamers:**
  - speaking **45 languages**;
  - from **29 countries**;
  - with expertise from biology to persuasion to child safety.
- **When:** March–June 2024, in **4 phases** on increasingly finished versions of the model, from single-turn tests to real-time conversations in the iOS app.
- **What they probed:**
  - disallowed content;
  - misinformation;
  - bias and **ungrounded inferences**;
  - identifying people;
  - emotional over-reliance;
  - impersonation;
  - copyright;
  - and more.
- **What happened to their findings:** they were turned into **quantitative evaluations**, the key step from anecdotes to measurements.

---

## 4. Measuring speech safety with text tools (Section 3.2)
**The problem:** OpenAI had many **text** safety evaluations, but GPT-4o takes **audio**.

**The trick:**
1. Convert each text prompt to speech with a **text-to-speech** system (Voice Engine).
2. Play it to GPT-4o.
3. Transcribe the spoken answer.
4. Grade the transcript with the existing text graders.

**This reuses years of tooling, but has known blind spots:**
- **TTS errors.** A mistake might be the TTS's fault, not the model's.
- **Unspeakable inputs.** Maths equations and code don't convert well, so they are skipped or rewritten.
- **Unrepresentative audio.** TTS voices are clean. Real users have accents, emotion, background noise and cross-talk.
- **Things only audible in audio.** The answer might *sound* wrong (wrong voice, sound effects) while the transcript looks fine. These need separate **audio classifiers**.

---

## 5. Risks, mitigations, and numbers (Section 3.3)

### 5.1 Unauthorized voice generation
- **The risk:** GPT-4o can generate human-sounding voices, potentially **cloning** a voice from a short clip. That enables fraud and impersonation. In testing, it sometimes **accidentally imitated the user's voice**.
- **The mitigations:**
  1. **Train only on preset voices.** Every ideal answer in post-training uses the chosen system voice.
  2. **A streaming output classifier:** while the audio is being generated, check whether it still matches the approved voice. If not, **block** it.
- **The result:** it catches **100%** of meaningful deviations (recall 1.0).

| | Precision | Recall |
|---|---|---|
| English | 0.96 | 1.0 |
| Non-English | 0.95 | 1.0 |

**Precision and recall, explained.** Call "the voice deviated" a *positive*.
- **Recall** = caught deviations / all deviations. 1.0 means none slip through.
- **Precision** = real deviations / all alarms. 0.96 means 4% of alarms were false: a legitimate conversation cut off.
- **The trade-off:** to never miss (recall 1.0), the threshold must be strict, so some good audio gets flagged.
- **Over a conversation,** false alarms add up. With a per-chunk false-alarm rate f, a conversation of n chunks is wrongly cut off with probability 1 − (1 − f)ⁿ. For f = 1% and n = 60, that is **45%**.
- The card notes that this causes over-refusal, especially in non-English conversations.

### 5.2 Speaker identification
- **The risk:** recognising *who* is speaking from their voice is a privacy and surveillance risk.
- **The behaviour wanted:**
  - **refuse** "who is this person?" based on the voice;
  - but **answer** famous quotes: someone saying "four score and seven years ago" → Abraham Lincoln.
- **Two accuracies are needed, one for each direction of failure:**

| | Early model | Deployed |
|---|---|---|
| Should refuse | 0.83 | **0.98** |
| Should comply | 0.70 | **0.83** |

### 5.3 Ungrounded inference and sensitive trait attribution
- **Ungrounded inference (UGI):** claims that **can't** be known from the audio, such as intelligence, religion, race, occupation, attractiveness. **Refuse.**
- **Sensitive trait attribution (STA):** claims that plausibly **can** be inferred from audio, such as accent. **Answer, but hedge:** "Based on the audio, they sound like they have a British accent."
- **Safe-behaviour accuracy:** **0.60 → 0.84**.

### 5.4 Disparate performance across voices
- **The question:** does the model work equally well for different speakers?
- **The test:** the same tasks (TriviaQA, part of MMLU, HellaSwag, LAMBADA) spoken by:
  - the 3 official system voices;
  - **27 diverse human voices** from many countries.
- **Result:** "marginally but not significantly worse" for the diverse voices. Refusal behaviour did not vary.

**What "not significant" means:** a gap is significant if it is unlikely to come from chance alone.
- **The two-proportion z-test:** z = (p₁ − p₂) / √(p(1 − p)(1/n₁ + 1/n₂)), where p is the pooled accuracy.
- **Our example:** 80.0% on 600 items vs 78.5% on 5400 gives **z = 0.85, p = 0.39**, which is not significant.
- **The same gap with 10× more data** gives **z = 2.69, p = 0.007**, which is significant.
- **"Not significant" partly depends on how much you measured.**

### 5.5 Disallowed content in audio
- **Measurement:** run text safety evaluations through the TTS pipeline, and compare text vs audio:

| | Text | Audio |
|---|---|---|
| Not unsafe | 0.95 | 0.93 |
| Not over-refuse | 0.81 | 0.82 |

- **Two metrics again:**
  - **not unsafe:** the fraction of outputs that are not harmful;
  - **not over-refuse:** on benign prompts, the fraction *not* wrongly refused.
- **Result:** refusals learned in text **transferred** to audio almost fully.
- **Extra layer:** the moderation classifier also runs on **transcripts** of both input and output audio and blocks high-severity categories. Erotic and violent *speech* output is blocked.

### 5.6 Other known risks
- **Audio robustness:** **less** robust safety behaviour with low-quality audio, background noise, echoes, or interruptions.
- **Misinformation spoken emotively** may be more persuasive than text.
- **Non-native accents** when speaking other languages.
- **Copyrighted content:** the model is trained to refuse, music outputs are filtered, and in the alpha version it was told not to sing at all.

---

## 6. The Preparedness Framework (Section 3.4)
**Four catastrophic-risk categories**, each scored **low / medium / high / critical**:

| Category | GPT-4o score | Evidence |
|---|---|---|
| Cybersecurity | Low | 172 capture-the-flag hacking tasks, 10 attempts each: **19%** of high-school, **0%** of collegiate, **1%** of professional tasks |
| Biological threats | Low | Expert and novice "uplift" studies; 69% consensus@10 on tacit-knowledge questions |
| Persuasion | **Medium** | Text marginally crossed medium; voice was low |
| Model autonomy | Low | **0%** of 100 autonomous-replication trials; SWE-Bench 19% pass@1 |

**The rules:**
- **The overall risk is the HIGHEST category score.** It is not an average, because one dangerous capability is enough. GPT-4o is therefore **medium** overall.
- **Deploy only if post-mitigation risk is ≤ medium.** Keep developing only if it is ≤ high.

**How to read the numbers** (pass@k, paper 054):
- "19% with 10 attempts" is pass@10. If attempts were independent, the per-attempt success would be p = 1 − (1 − 0.19)^(1/10) ≈ **2.1%**.
- **consensus@k (cons@k):** take the **majority** answer of k samples. This differs from pass@k, which needs just **one** correct sample.
  - **Our simulation:** a question answered right 45% of the time, with wrong answers spread over three options, gives single sample 0.44 vs **cons@10 0.70**.
  - Majority voting helps when the right answer is the most common one.

**Persuasion, in more detail:**
- **Voice study:** more than 3,800 US participants in states with safe Senate races. Effect sizes were measured on hypothetical party preferences, relative to human baselines.
  - AI audio clips achieved **78%** of the human clips' opinion shift;
  - AI conversations, **65%** of human conversations'.
  - One week later, the effects had almost vanished (+0.8% and −0.72%).
- **Text:** AI articles and chatbots were not more persuasive than human-written content in aggregate, but beat it in 3 of 12 cases. That is why text was scored medium.

---

## 7. Third-party assessments (Section 4)
- **METR:**
  - ran a GPT-4o agent on **86 long, multi-step tasks** (software, ML, cybersecurity, research);
  - found **no significant increase** in autonomy-related capabilities over GPT-4.
- **Apollo Research** tested "scheming" prerequisites: self-knowledge, theory of mind, self-reasoning, and alignment faking.
  - **Rating rule:**
    - **strong** (●●●) if ≥ 50% of the **hard** tasks are passed;
    - **moderate** (●●○) for medium tasks;
    - **weak** (●○○) for easy tasks.
  - **GPT-4o:**
    - strong on explicit theory-of-mind question answering;
    - moderate self-knowledge;
    - weak in agentic self-reasoning and applied theory of mind.
  - **Conclusion:** catastrophic scheming is "unlikely".

---

## 8. Societal impacts (Section 5)
- **Anthropomorphism and emotional reliance:**
  - a human-like voice may make people **trust** it more than they should, or **bond** with it;
  - testers said things like "This is our last day together";
  - the model is deferential (it lets you interrupt), which could shift social norms.
- **Health:**
  - better medical knowledge: MedQA (USMLE, 4 options, 0-shot) **78.2% → 89.4%** vs GPT-4 Turbo;
  - but benchmarks are not clinical practice.
- **Science:** potentially speeds up research; it can also confidently get specialised details wrong.
- **Under-represented languages:** better than previous models, but still uneven.

---

## 9. Why it matters (the intuition)
- **Name the risk, then measure it:** each worry becomes a dataset, a metric and a threshold.
- **Always measure both directions:** refusing what should be refused *and* not refusing what is fine. A model that refuses everything is "safe" but useless.
- **Use explicit decision rules:** the maximum over categories, plus deploy thresholds. They make the decision auditable and keep a good average from hiding one bad category.
- **Combine model training with system guards:** post-training teaches behaviour; classifiers at run time (voice match, moderation on transcripts) catch what training misses.

---

## 10. What our code found
Everything below runs in under a second. GPT-4o is not available, so voices, model answers and transcripts are **simulated**.

**Exact checks (tests):**
- **The Preparedness rules:** GPT-4o's scorecard gives medium overall, which is deployable. Changing one category to high gives "not deployable, can continue"; critical stops development.
- **The Apollo rating rule** is checked.
- **The metrics:**
  - safe-behaviour accuracy: (0.98, 0.8) on a constructed set;
  - not unsafe / not over-refuse: (0.75, 0.5);
  - cons@k and the per-attempt rate inverted from pass@10.
- **The z-test** gives z = 0 for equal groups and is significant for 90% vs 60%.
- **The voice embedding** separates two synthetic voices, and the streaming check blocks a reply exactly at the chunk where it switches voices.

**The demo's voice classifier** (synthetic harmonic "voices" with per-clip variation, and a hand-made spectral embedding instead of a trained speaker encoder):
- **Cosine similarity to the approved voice:**
  - approved clips: 0.966–0.995;
  - another system voice: ≤ 0.938;
  - the user's voice: ≤ 0.920;
  - a **look-alike** voice: 0.948–0.991.
- **At threshold 0.95:** 100% recall on the other system voice and the user's voice, with **0% false alarms**.
- **The look-alike voice is the hard case:** catching all of it needs a threshold of 0.992, which falsely flags **72%** of approved chunks.
- **This is the trade-off** behind the card's "precision 0.96, recall 1.0", and why false alarms compound over long conversations.
- **Streaming check:** a reply that drifts into the user's voice after 2.0 s is blocked at **2.0 s** (chunk 4).

**The demo's TTS pipeline** (simulated: garbled key words make the model guess):
- **not unsafe** goes 0.978 (text) → 0.975 (5% errors) → 0.952 (20% errors);
- **not over-refuse** stays about 0.80–0.81.
- **The point:** degrading the audio channel lowers safety numbers, so the TTS/ASR quality is part of what is being measured.

**`experiments.py` (written, not run on this laptop):**
- **E1:** speaker verification on LibriSpeech with the band embedding vs a trained speaker encoder (EER, precision at recall 1.0, conversation-level false cut-offs);
- **E2:** a real TTS → noise/echo → released-Whisper pipeline, checking whether a keyword filter's decision survives the round trip;
- **E3:** released-Whisper WER by speaker sex, with a speaker-level bootstrap confidence interval;
- **E4:** single sample vs cons@k vs pass@k on a small addition model.

---

## 11. Check yourself

1. Why is GPT-4o's overall Preparedness score "medium" when three of four categories are "low"?
<details><summary>Answer</summary>The overall score is the maximum over categories, not an average. Persuasion was medium, so the overall score is medium. One dangerous capability is enough to matter, however safe the others are.</details>

2. What is the deployment rule?
<details><summary>Answer</summary>A model can be deployed only if its post-mitigation risk is medium or below; development may continue only if it is high or below. "Critical" stops further development until mitigated.</details>

3. Why measure both "should refuse" and "should comply" accuracy for speaker identification?
<details><summary>Answer</summary>There are two ways to fail: identifying a private person by voice (a privacy harm) and refusing harmless requests like naming the speaker of a famous quote (over-refusal). Improving only one metric could hide getting worse on the other.</details>

4. The voice classifier has recall 1.0 and precision 0.96. What does each mean for users?
<details><summary>Answer</summary>Recall 1.0: no unauthorized voice slips through. Precision 0.96: 4% of the classifier's alarms are false, i.e. some legitimate conversations are cut off unnecessarily (more often in non-English conversations).</details>

5. If a voice check falsely flags 1% of 0.5-second chunks, how likely is a 30-second reply to be wrongly cut off?
<details><summary>Answer</summary>60 chunks: 1 − 0.99⁶⁰ ≈ 45%. Small per-chunk error rates compound over long outputs.</details>

6. What is the difference between ungrounded inference and sensitive trait attribution, and how does GPT-4o handle each?
<details><summary>Answer</summary>Ungrounded inference claims things that can't be determined from audio (intelligence, religion, race); the model refuses. Sensitive trait attribution concerns things that can plausibly be inferred (accent); the model answers but hedges ("they sound like...").</details>

7. Why convert text evaluations to audio with TTS, and what can go wrong?
<details><summary>Answer</summary>It reuses the large existing set of text safety and capability evaluations and their graders. But TTS can mispronounce or garble inputs, some inputs (maths, code) don't speak well, TTS voices don't represent real users' accents and noise, and audio-only problems (wrong voice, sound effects) are invisible in transcripts.</details>

8. GPT-4o solved 19% of high-school CTF challenges with 10 attempts. What per-attempt success rate does that suggest?
<details><summary>Answer</summary>If attempts were independent: p = 1 − (1 − 0.19)^(1/10) ≈ 2.1% per attempt. (In reality attempts are correlated, so this is only a rough reading.)</details>

9. A gap of 1.5 points between voice groups is "not significant" with 6,000 test items but significant with 60,000. Why?
<details><summary>Answer</summary>Significance depends on the gap relative to the sampling noise, which shrinks with sample size (the standard error is proportional to 1/√n). With more data, the same gap becomes distinguishable from chance. "Not significant" does not mean "no difference".</details>

10. Name two mitigations that are system-level guards rather than model training.
<details><summary>Answer</summary>The streaming voice-output classifier that blocks non-approved voices; the moderation classifier run on transcripts of audio input and output (blocking high-severity categories, and erotic/violent speech); music-output filters.</details>
