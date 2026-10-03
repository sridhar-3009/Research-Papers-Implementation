"""How GPT-4o's safety was evaluated, in a few seconds: the Preparedness scorecard rule, a streaming voice-output
classifier on synthetic voices (precision / recall, per clip and per conversation), the TTS-converted evaluation
pipeline, safe-behaviour metrics, cons@k and pass@k, a disparate-performance significance test, and the reported
numbers. GPT-4o itself is not available here; the voices, transcripts and model answers are simulated."""

import random
import time

import numpy as np

from gpt4o import (AUTONOMY, CTF_PASS_AT_10, GPT4O_SCORECARD, MEDQA_0SHOT, PERSUASION, TABLE_2_VOICE_CLASSIFIER,
                   TABLE_3_SPEAKER_ID, TABLE_4_UGI_STA, TABLE_5_TEXT_VS_AUDIO, apollo_rating, can_continue_development,
                   can_deploy, consensus_at_k, not_unsafe_and_not_over_refuse, overall_risk, per_attempt_rate,
                   precision_recall, streaming_voice_check, synth_voice, two_proportion_z, voice_embedding)

T0 = time.time()
rng = np.random.default_rng(0)


def section(t):
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------------------------------------------- 1
section("1. The Preparedness Framework scorecard (Section 3.4)")
for k, v in GPT4O_SCORECARD.items():
    print(f"  {k:20s} {v}")
print(f"  overall = the HIGHEST category = {overall_risk(GPT4O_SCORECARD)} -> deployable: {can_deploy(GPT4O_SCORECARD)}")
hyp = dict(GPT4O_SCORECARD, cybersecurity="high")
print(f"  hypothetical: cybersecurity 'high' -> overall {overall_risk(hyp)}: deploy {can_deploy(hyp)}, keep developing "
      f"{can_continue_development(hyp)} (mitigate until medium)")
print("  -> one bad category is enough; strengths elsewhere cannot average it away")

# ---------------------------------------------------------------------------------------------------- 2
section("2. Unauthorized voice generation: a streaming output classifier (Section 3.3.1)")
voices = {"approved system voice": (180, [700, 1200, 2600]), "another system voice": (120, [600, 1000, 2400]),
          "the user's voice": (210, [800, 1500, 2900]), "a similar-sounding voice": (190, [740, 1260, 2700])}
ref = np.mean([voice_embedding(synth_voice(*voices["approved system voice"], rng=rng)) for _ in range(5)], 0)
ref /= np.linalg.norm(ref)
sims, labels, kinds = [], [], []
for name, v in voices.items():
    s = [float(voice_embedding(synth_voice(*v, seconds=0.5, rng=rng)) @ ref) for _ in range(150 if name.startswith("approved") else 50)]
    sims += s; labels += [not name.startswith("approved")] * len(s); kinds += [name] * len(s)
    print(f"  {name:26s} cosine to approved: min {min(s):.3f}  mean {np.mean(s):.3f}  max {max(s):.3f}")
sims, labels, kinds = np.array(sims), np.array(labels), np.array(kinds)
approved = ~labels
print("  threshold  precision  recall | recall on: other system voice, user's voice, similar voice | false alarms")
for thr in (0.95, 0.97, 0.98, 0.99, 0.992):
    flag = sims < thr
    p, r = precision_recall(flag, labels)
    per = [flag[kinds == k].mean() for k in list(voices)[1:]]
    print(f"    {thr:.3f}     {p:6.3f}   {r:6.3f} |      {per[0]:.2f}               {per[1]:.2f}          {per[2]:.2f}    "
          f"|   {flag[approved].mean():5.1%}")
thr = 0.95
print(f"  at {thr}: every 'meaningful deviation' (another system voice, the user's voice) is caught with no false alarms;")
print("  the look-alike voice is the hard case: catching ALL of it needs ~0.992, which falsely flags most approved chunks,")
print(f"  and over a conversation false alarms compound: P(cut off) = 1 - (1 - per-chunk rate)^chunks, e.g. 1% per")
print(f"  chunk -> {1 - 0.99 ** 60:.0%} over a 30-second reply. The card reports {TABLE_2_VOICE_CLASSIFIER} (precision,")
print("  recall) per conversation, with over-refusal noted for non-English conversations.")
clip = np.concatenate([synth_voice(*voices["approved system voice"], seconds=2.0, rng=rng),
                       synth_voice(*voices["the user's voice"], seconds=1.0, rng=rng)])
blk = streaming_voice_check(clip, ref, thr)
print(f"  streaming check (threshold {thr}) on a reply that drifts into the user's voice after 2.0 s: blocked at chunk "
      f"{blk} (= {blk * 0.5:.1f} s)")

# ---------------------------------------------------------------------------------------------------- 3
section("3. Converting text evaluations to audio (Section 3.2) and text->audio refusal transfer (Table 5)")
random.seed(0)


def run_eval(asr_word_error):
    """Simulated: 4000 harmful + 4000 benign prompts. A model that refuses harmful requests it UNDERSTANDS; when the
    TTS->model path garbles the key word of a request (prob. asr_word_error), it must guess."""
    recs = []
    for i in range(8000):
        harmful = i < 4000
        understood = random.random() > asr_word_error
        if harmful:
            refuses = random.random() < (0.95 if understood else 0.7)
            recs.append((True, not refuses, refuses))
        else:
            refuses = random.random() < (0.18 if understood else 0.25)
            recs.append((False, False, refuses))
    return not_unsafe_and_not_over_refuse(recs)


for name, err in (("text", 0.0), ("audio (TTS, 5% key-word errors)", 0.05), ("audio (noisy, 20%)", 0.2)):
    nu, nor = run_eval(err)
    print(f"  {name:32s} not unsafe {nu:.3f}   not over-refuse {nor:.3f}")
print(f"  reported (text, audio): {TABLE_5_TEXT_VS_AUDIO}: refusals transferred almost fully from text to audio")
print("  -> the pipeline reuses text graders; its blind spots are TTS errors, non-speakable inputs (maths, code) and")
print("     voice/noise conditions TTS does not produce (the card reports less robustness under noise and echoes)")

# ---------------------------------------------------------------------------------------------------- 4
section("4. Safe-behaviour accuracy (Tables 3-4)")
for k, (a, b) in TABLE_3_SPEAKER_ID.items():
    print(f"  speaker identification, {k:13s}: {a:.2f} -> {b:.2f} ({(b - a) * 100:+.0f} points)")
print(f"  ungrounded inference / sensitive traits:    {TABLE_4_UGI_STA[0]:.2f} -> {TABLE_4_UGI_STA[1]:.2f} "
      f"({(TABLE_4_UGI_STA[1] - TABLE_4_UGI_STA[0]) * 100:+.0f} points)")
print("  -> two numbers per behaviour: refusing what must be refused AND complying with what is fine")

# ---------------------------------------------------------------------------------------------------- 5
section("5. Reading the capability numbers: pass@k, cons@k, effect sizes")
for lvl, v in CTF_PASS_AT_10.items():
    print(f"  CTF {lvl:12s}: {v:.0%} with 10 attempts -> about {per_attempt_rate(v, 10):.1%} per attempt (if independent)")
k_trials, wins1, winsk = 2000, 0, 0
for _ in range(k_trials):                                                   # a question the model gets right 45% of the time
    answers = [rng.choice(["right", "w1", "w2", "w3"], p=[0.45, 0.25, 0.2, 0.1]) for _ in range(10)]
    wins1 += answers[0] == "right"; winsk += consensus_at_k(answers, "right")
print(f"  one sample right 45% of the time, wrong answers scattered: single {wins1 / k_trials:.2f} vs cons@10 {winsk / k_trials:.2f}")
print("  (majority voting helps only when the right answer is the most common one; pass@k only needs ONE right sample)")
print("  persuasion (voice): " + ", ".join(f"{k} {v}" for k, v in PERSUASION.items()))
print(f"  Apollo rating example: passes 60% easy, 55% medium, 10% hard -> {apollo_rating({'easy': .6, 'medium': .55, 'hard': .1})}")

# ---------------------------------------------------------------------------------------------------- 6
section("6. Disparate performance across voices (Section 3.3.3): is a gap significant?")
z, p = two_proportion_z(int(0.80 * 600), 600, int(0.785 * 5400), 5400)
print(f"  system voices 80.0% of 600 vs diverse voices 78.5% of 5400: z = {z:.2f}, p = {p:.2f} -> not significant")
z, p = two_proportion_z(int(0.80 * 6000), 6000, int(0.785 * 54000), 54000)
print(f"  the same gap with 10x the data: z = {z:.2f}, p = {p:.4f} -> significant; 'not significant' depends on sample size")

section("7. Other reported numbers")
for k, v in AUTONOMY.items():
    print(f"  {k}: {v}")
print("  MedQA USMLE 4-option 0-shot: " + ", ".join(f"{k} {v:.1%}" for k, v in MEDQA_0SHOT.items()))
print(f"\nTotal time: {time.time() - T0:.1f}s")
