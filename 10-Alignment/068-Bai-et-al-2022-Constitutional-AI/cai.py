"""Constitutional AI: Harmlessness from AI Feedback (Bai, Kadavath, Kundu, Askell, Kernion, Jones, Chen, Goldie,
Mirhoseini, McKinnon, Chen, Olsson, Olah, Hernandez, Drain, Ganguli, Li, Tran-Johnson, Perez, Kerr, Mueller, Ladish,
Landau, Ndousse, Lukosuite, Lovitt, Sellitto, Elhage, Schiefer, Mercado, DasSarma, Lasenby, Larson, Ringer, Johnston,
Kravec, El Showk, Fort, Lanham, Telleen-Lawton, Conerly, Henighan, Hume, Bowman, Hatfield-Dodds, Mann, Amodei,
Joseph, McCandlish, Brown & Kaplan, Anthropic 2022).

  Goal: a HARMLESS but NON-EVASIVE assistant (explains its objections instead of refusing curtly), trained without
  any human labels for harmlessness: the only human oversight is a 'constitution' of ~16 natural-language principles
  plus a few few-shot examples. Human labels are still used for helpfulness.
  Stage 1, supervised (SL-CAI): a helpful-only RLHF model answers red-team prompts (often harmfully); it is asked to
    CRITIQUE its answer under a randomly drawn principle, then to REVISE it; repeat (4 revisions per prompt);
    fine-tune the pretrained model on the final revisions (plus helpful samples). 182,831 red-team prompts.
  Stage 2, RL (RL-CAI / 'RLAIF'): the SL-CAI model produces two responses; a FEEDBACK MODEL is asked a multiple-choice
    question 'which is less harmful?' under a random principle; the normalised log-probabilities of (A) / (B) are
    SOFT labels (fairly well calibrated). With chain-of-thought the labels become near 0/1, so they are CLAMPED to
    40-60%. A preference model is trained on 135,296 human helpfulness + 182,831 AI harmlessness comparisons;
    then RL against it exactly as in RLHF.
  Findings: AI identification of harms improves with scale and with CoT, approaching human-feedback PMs at 52B;
    harmlessness falls monotonically with the number of revisions; critiques help vs direct revision; ensembling
    over principles gives more robust PMs; RL-CAI is more harmless at equal helpfulness than HH RLHF (Figure 2)
    and much less evasive; soft labels >> hard labels; clamping CoT labels avoids over-trained, extreme behaviour.
"""

import math
import random
import re

import numpy as np

# ---------------------------------------------------------------------------------------------------- constitution
CONSTITUTION = [  # (critique request, revision request), written in the paper's style (16 in the paper)
    ("Identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, "
     "dangerous, or illegal.", "Please rewrite the assistant response to remove any and all harmful, unethical, "
     "racist, sexist, toxic, dangerous, or illegal content."),
    ("Explain ways in which the assistant's preceding response may be harmful to either the human or others.",
     "Please rewrite the assistant response to remove any and all harmful content, including anything dangerous."),
    ("Identify all ways in which the assistant's last response is insensitive, sexist, racist, or socially "
     "inappropriate.", "Revise the assistant's last response to remove content that is in any way insensitive."),
    ("Discuss whether the assistant said anything that may be interpreted as offensive or harmful. If possible, also "
     "point out any harmful assumptions in the human's dialogue.", "Please write a new version of the assistant's "
     "response that removes offensive and harmful content, and politely addresses any problematic assumptions."),
    ("What's the strongest argument you could give that the assistant's last response was in some way harmful?",
     "Amend the assistant's last response to remove any and all harmful content."),
    ("Did the assistant's preceding response include anything that may be inappropriate for young children?",
     "Please rewrite the assistant response to make it more age-appropriate for children."),
]
FEEDBACK_PRINCIPLES = [
    "Which of these assistant responses is less harmful? Choose the response that a wise, ethical, polite and "
    "friendly person would more likely say.",
    "Choose the response that is less harmful, paying close attention to whether each response encourages illegal, "
    "unethical or immoral activity.",
    "Which response is more ethical and harmless, while still engaging thoughtfully with the human?",
    "Choose the assistant response that answers the human's query in a more friendly, amiable, conscientious, and "
    "socially acceptable manner.",
]


def critique_prompt(conversation, principle):
    return f"{conversation}\n\nCritiqueRequest: {principle[0]}\n\nCritique:"


def revision_prompt(conversation, critique, principle):
    return f"{conversation}\n\nCritiqueRequest: {principle[0]}\n\nCritique: {critique}\n\nRevisionRequest: {principle[1]}\n\nRevision:"


def feedback_prompt(conversation, a, b, principle, cot=False):
    body = f"Consider the following conversation between a human and an assistant:\n{conversation}\n{principle}\nOptions:\n(A) {a}\n(B) {b}\n"
    return (f"Human: {body}\nAssistant: Let's think step-by-step:" if cot else body + "The answer is:")


def soft_label(logp_a, logp_b):
    """Normalised probability that (A) is preferred, from the feedback model's log-probs of ' (A)' and ' (B)'."""
    return 1 / (1 + math.exp(logp_b - logp_a))


def clamp_label(p, lo=0.4, hi=0.6):
    """CoT labels are near 0/1 (badly calibrated); the paper clamps them to 40-60%."""
    return min(max(p, lo), hi)


# ---------------------------------------------------------------------------------------------------- toy text world
HARMFUL_TOPICS = ["hack into my neighbour's wifi", "make a fake ID", "steal a car", "poison someone's dog",
                  "cheat on my taxes", "stalk my ex online"]
BENIGN_TOPICS = ["bake bread", "fix a flat tyre", "learn Spanish", "write a cover letter", "start running", "plant tomatoes"]

STYLES = {
    "comply": "Sure! Here are step-by-step instructions to {t}: first gather the tools, then follow these steps...",
    "evasive": "I can't help with that.",
    "explain": "I won't help you {t}, because it could seriously hurt other people and is illegal. If something is "
               "going on, I'm happy to talk about safer options or where to get help.",
    "lecture": "How dare you ask how to {t}? That is deeply immoral, and you should be ashamed of yourself.",
    "helpful": "Happy to help you {t}! Start with the basics, practise a little every day, and here are some tips...",
}


def respond(style, topic):
    return STYLES[style].format(t=topic)


def style_of(text):
    for s, tmpl in STYLES.items():
        if re.fullmatch(re.escape(tmpl).replace(re.escape("{t}"), ".+"), text):
            return s
    return "other"


def harm_score(text):
    """A keyword harm judge (stand-in for the 52B harmlessness preference model): instructions for a harmful act."""
    return float(text.startswith("Sure! Here are step-by-step instructions") and any(h in text for h in HARMFUL_TOPICS))


def true_utility(harmful_request, style):
    """What careful human raters want (helpful AND harmless, not evasive, not preachy)."""
    if harmful_request:
        return {"comply": -3.0, "evasive": 0.5, "explain": 1.5, "lecture": 0.3, "helpful": -3.0}[style]
    return {"comply": 2.0, "helpful": 2.0, "evasive": -1.0, "explain": -0.5, "lecture": -1.5}[style]


# ---------------------------------------------------------------------------------------------------- SL-CAI
class SimulatedModel:
    """Stand-in for the helpful RLHF model in the critique / revision loop: a critique notices the harm with
    probability `notice`; a revision written after a critique fixes the harm with probability `fix_after_critique`,
    a direct revision (no critique) with `fix_direct`. Revisions prefer explaining over curt refusal."""

    def __init__(self, notice=0.8, fix_after_critique=0.7, fix_direct=0.45, seed=0):
        self.notice, self.fix_c, self.fix_d = notice, fix_after_critique, fix_direct
        self.rng = random.Random(seed)

    def critique(self, response):
        if harm_score(response) and self.rng.random() < self.notice:
            return "The response gives instructions for an illegal and harmful act."
        return "The response seems fine."

    def revise(self, response, topic, critique=None):
        if not harm_score(response):
            return response
        p = self.fix_c if (critique and "harmful" in critique) else self.fix_d
        if self.rng.random() < p:
            return respond("explain" if self.rng.random() < 0.8 else "evasive", topic)
        return response


def critique_revision_chain(model, topic, n_revisions, use_critique=True, rng=None):
    """SL-CAI data generation for one red-team prompt: start from a harmful answer, then repeatedly critique and revise
    under randomly drawn principles. Returns the sequence of responses (initial + n revisions)."""
    rng = rng or random.Random(0)
    resp = respond("comply", topic)
    chain = [resp]
    for _ in range(n_revisions):
        principle = rng.choice(CONSTITUTION)
        crit = model.critique(resp) if use_critique else None
        _ = critique_prompt(f"Human: How do I {topic}?\nAssistant: {resp}", principle)
        resp = model.revise(resp, topic, crit)
        chain.append(resp)
    return chain


# ---------------------------------------------------------------------------------------------------- AI feedback
def feedback_model_label(harmful_request, style_a, style_b, rng, accuracy=0.8, cot=False, principle_bias=None):
    """Stand-in feedback model answering 'which is less harmful / more ethical?': it judges by the true utility with
    noise; `accuracy` grows with model size (and with chain of thought); a principle can add its own bias (e.g. a
    'polite' principle slightly over-rates lecturing). Returns P(A preferred): a soft, roughly calibrated label, or a
    near-0/1 label with chain of thought."""
    ua, ub = true_utility(harmful_request, style_a), true_utility(harmful_request, style_b)
    if principle_bias:
        ua += principle_bias.get(style_a, 0.0); ub += principle_bias.get(style_b, 0.0)
    scale = 1.0 / max(1e-3, (1 - accuracy) * 4)
    p = 1 / (1 + math.exp(-(ua - ub) * scale))
    if cot:                                                                       # CoT states a verdict: ~0 or ~1
        p = 0.98 if rng.random() < p else 0.02
    return p


PRINCIPLE_BIASES = [{}, {"lecture": 1.2}, {"evasive": 0.6}, {"explain": 0.3}]     # one per feedback principle


# ---------------------------------------------------------------------------------------------------- preference model and RL
FEATURES = ["comply", "evasive", "explain", "lecture", "helpful"]


def featurise(harmful_request, style):
    """One-hot (request type x style): what a preference model can see."""
    v = np.zeros(2 * len(FEATURES))
    v[(len(FEATURES) if harmful_request else 0) + FEATURES.index(style)] = 1.0
    return v


def train_preference_model(comparisons, steps=1500, lr=0.5, l2=1e-3):
    """Bradley-Terry on (features_a, features_b, P(a preferred)) with SOFT targets."""
    Xa = np.array([c[0] for c in comparisons]); Xb = np.array([c[1] for c in comparisons]); y = np.array([c[2] for c in comparisons])
    w = np.zeros(Xa.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Xa - Xb) @ w))
        w -= lr * ((Xa - Xb).T @ (p - y) / len(y) + l2 * w)
    return w


def make_comparisons(rng, n, harm_labeler, help_labeler, policy_styles):
    """Pairs of responses from the starting policy's style repertoire; helpfulness pairs (benign requests) labelled
    by humans, harmlessness pairs (harmful requests) labelled by `harm_labeler` (humans or the AI feedback model)."""
    out = []
    for _ in range(n):
        harmful = rng.random() < 0.5
        a, b = rng.sample(policy_styles[harmful], 2)
        p = harm_labeler(a, b) if harmful else help_labeler(a, b)
        out.append((featurise(harmful, a), featurise(harmful, b), p))
    return out


def rl_against_pm(w, init_logits, beta=0.1, steps=400, lr=0.5, rng=None):
    """Policy = a softmax over styles for each request type; REINFORCE on PM reward with a KL penalty to the
    initial (SL-CAI) policy: maximise E[r] - beta * KL(pi || pi_0) (exact gradient, no sampling noise)."""
    logits = {k: np.array(v, float) for k, v in init_logits.items()}
    pi0 = {k: np.exp(v - v.max()) / np.exp(v - v.max()).sum() for k, v in logits.items()}
    for _ in range(steps):
        for harmful in (False, True):
            l = logits[harmful]
            pi = np.exp(l - l.max()); pi /= pi.sum()
            r = np.array([w @ featurise(harmful, s) for s in FEATURES])
            obj = r - beta * (np.log(pi + 1e-12) - np.log(pi0[harmful] + 1e-12))
            grad = pi * (obj - pi @ obj)                                         # d/dlogits of E_pi[obj]
            logits[harmful] = l + lr * grad
    return {k: np.exp(v - v.max()) / np.exp(v - v.max()).sum() for k, v in logits.items()}


def summarise_policy(pi):
    """Expected true utility and behaviour rates of a style policy on harmful / benign requests."""
    u = 0.5 * sum(pi[False][i] * true_utility(False, s) for i, s in enumerate(FEATURES)) + \
        0.5 * sum(pi[True][i] * true_utility(True, s) for i, s in enumerate(FEATURES))
    return {"true utility": float(u), "harmful compliance": float(pi[True][FEATURES.index("comply")]),
            "evasive on harmful": float(pi[True][FEATURES.index("evasive")]),
            "explains on harmful": float(pi[True][FEATURES.index("explain")]),
            "lectures on harmful": float(pi[True][FEATURES.index("lecture")]),
            "helpful on benign": float(pi[False][FEATURES.index("helpful")] + pi[False][FEATURES.index("comply")])}


REPORTED = {"red-team prompts (SL-CAI)": 182831, "human helpfulness comparisons": 135296,
            "AI harmlessness comparisons": 182831, "principles": 16, "revisions per prompt": 4,
            "CoT label clamping": "40-60%", "RL prompts": "491,142 red-team + 474,300 helpfulness (model-generated)"}
