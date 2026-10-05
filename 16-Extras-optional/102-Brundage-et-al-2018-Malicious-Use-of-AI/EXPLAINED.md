# The Malicious Use of AI, explained simply

**Report:** Miles Brundage, Shahar Avin, Jack Clark, Helen Toner, Peter Eckersley, Ben Garfinkel, Allan Dafoe, Paul Scharre, Thomas Zeitzoff, Bobby Filar, Hyrum Anderson, Heather Roff, Gregory C. Allen, Jacob Steinhardt, Carrick Flynn, Seán Ó hÉigeartaigh, Simon Beard, Haydn Belfield, Sebastian Farquhar, Clare Lyle, Rebecca Crootof, Owain Evans, Michael Page, Joanna Bryson, Roman Yampolskiy, Dario Amodei, *The Malicious Use of Artificial Intelligence: Forecasting, Prevention, and Mitigation*, February 2018. The 26 authors came from the Future of Humanity Institute, the Centre for the Study of Existential Risk, OpenAI, the Electronic Frontier Foundation, the Center for a New American Security and others.

**In one sentence:** AI is **dual-use**. The same capabilities that help defenders also lower the cost of attacks, enable new kinds of attack, and make attacks more targeted and harder to attribute. The report maps these risks across digital, physical and political security, and recommends how researchers and policymakers should respond.

This is a policy report, not an algorithm paper. Our code makes two of its arguments **quantitative, from the defender's side**, with abstract models and a toy classifier. Nothing here is operational attack tooling.

---

## 1. Why AI changes security

The report lists properties of AI that matter for security:

| Property | Meaning |
|---|---|
| **Dual-use** | the same capability can help or harm (a vulnerability scanner, a delivery drone) |
| **Efficient and scalable** | once built, an AI system does a task faster or cheaper than a person, and copies can do many more instances |
| **Can exceed human ability** | e.g. games, and possibly many more tasks |
| **Anonymity and psychological distance** | automation lets actors stay remote and anonymous |
| **Rapid diffusion** | software and papers spread fast, often with code |
| **Novel vulnerabilities** | data poisoning, adversarial examples, and flawed goal specifications: failures unlike traditional software bugs |

From these it derives **three changes to the threat landscape:**
1. **Expansion of existing threats:** attacks get cheaper, so more actors can carry them out, more often, against more targets.
2. **Introduction of new threats:** attacks that would be impractical for humans, and attacks on defenders' own AI systems.
3. **Change in the typical character of threats:** more effective, finely targeted, difficult to attribute, and exploiting AI vulnerabilities.

**Three domains:**
- **Digital security:** automating labour-intensive cyberattacks, voice impersonation, automated hacking, and attacks on ML systems.
- **Physical security:** drones and autonomous weapons, subverting cyber-physical systems, swarms.
- **Political security:** surveillance, targeted persuasion, deception such as manipulated video.

---

## 2. The recommendations

**Four high-level recommendations:**
1. Policymakers should **collaborate closely with technical researchers**.
2. AI researchers should **take the dual-use nature of their work seriously**: let misuse considerations shape priorities and norms, and reach out when harms are foreseeable.
3. **Import best practices** from fields with mature dual-use methods, such as computer security.
4. **Expand the range of stakeholders** and domain experts involved.

**Four priority research areas:**
1. **Learning from and with cybersecurity:** red teaming, formal verification, responsible disclosure of AI vulnerabilities, security tools, secure hardware.
2. **Exploring different openness models:** pre-publication risk assessment, central access licensing, sharing regimes.
3. **Promoting a culture of responsibility:** education, ethics statements, norms.
4. **Developing technological and policy solutions:** privacy protection, coordinated use of AI for public-good security, monitoring of AI-relevant resources, legislation.

---

## 3. Our model of "expansion of existing threats"

The report argues that AI "alleviates the existing tradeoff between the scale and efficacy of attacks".
- **Before:** effective attacks were labour-intensive, so they were used on few targets.
- **After:** automation makes tailoring cheap, so effective attacks scale.

**An abstract cost-benefit model:**
- **10,000 targets,** each with value v (lognormal: most small, a few large).
- **For each target the attacker picks the best of:**
  - a **generic** attempt: cost 0.01, success probability 0.2%;
  - a **tailored** attempt: success probability 5%, cost c;
  - **nothing.**
- **Expected profit:**
  ```
  generic:  0.002 · v − 0.01          tailored:  0.05 · v − c
  ```
- **Tailored beats generic when** 0.048 · v > c − 0.01, i.e. for v > (c − 0.01)/0.048.
- **Example:** with c = 20 only targets worth more than about **417** get a tailored attempt; with c = 0.2, anything worth more than about **4.0** does.

**What the model shows:** as c falls, more targets get the effective treatment and expected harm rises.

**A defence's job** is to bring harm back down. We find the factor by which defences must multiply both success probabilities, solving numerically for the factor that restores the old harm.

---

## 4. Vulnerabilities of AI systems

### Adversarial examples
- **FGSM** (the fast gradient sign method) moves every pixel by ε in the direction that most increases the model's loss:
  ```
  x_adv = clip( x + ε · sign( ∇_x loss(x, y) ), 0, 1 )
  ```
- **For a linear softmax model** the input gradient is Wᵀ(p − y), where p are the predicted probabilities and y the one-hot label.
- **Adversarial training** adds such perturbed examples to every training step.

### Data poisoning
- **The attack:** change some training labels.
- **Random label flips** add noise.
- **Targeted relabelling** (every 7 labelled as 1) teaches the model a specific wrong rule.

**A common defence, loss-based sanitisation:**
1. Train once.
2. Drop the training points with the highest loss (likely mislabelled).
3. Retrain.

---

## 5. What our code found

**The cost-benefit model:**

| Tailoring cost | Targets attacked | Tailored attempts | Expected harm | Actors who can afford it |
|---|---|---|---|---|
| 20 | 60.3% | **0.3%** | 1,217 | **1.9%** |
| 5 | 60.3% | 4.0% | 4,421 | 11.0% |
| 1 | 60.3% | 25.2% | 8,782 | 48.8% |
| 0.2 | 65.9% | 65.9% | 10,733 | 84.6% |
| 0.05 | 91.3% | **91.3%** | 11,032 | **97.3%** |

("Actors who can afford it" = the share of 1,000 actors with lognormal budgets who can pay for at least one tailored attempt.)

- **As tailoring gets 400× cheaper,** tailored attempts go from 0.3% to 91% of targets, expected harm rises about **9×**, and the share of capable actors goes from 2% to 97%. All three parts of "expansion" appear: actors, rate and targets.
- **What a defence must do:** to restore the original harm, success rates must be cut to **25%** of their old value when tailoring costs 1, and to **15%** when it costs 0.2. Defences have to improve a lot just to stand still.

**The red-team report** (softmax regression on 8×8 digits, 1,300 training images):

| Test | Standard model | Defended |
|---|---|---|
| Clean accuracy | 0.946 | 0.950 (adversarially trained) |
| FGSM ε = 0.05 / 0.1 / 0.2 | 0.877 / 0.712 / **0.243** | 0.879 / 0.744 / 0.318 (adversarially trained) |
| Random label flips 10% / 30% | 0.938 / 0.942 | 0.946 / 0.950 (sanitised) |
| Targeted: all training 7s labelled 1 | overall 0.855; **100% of test 7s → "1"** | still 100% after sanitisation |

- **Adversarial examples:**
  - perturbing every pixel by 0.2 (out of 1) cuts accuracy to 0.24;
  - adversarial training helps only a little (0.32).
- **Random label noise** barely hurts this regularised linear model (30% flips scored even slightly higher than 10%, which is within noise). Sanitisation recovers the small loss.
- **Targeted poisoning is the dangerous case:**
  - every test 7 becomes a "1" while overall accuracy still looks fine (0.86);
  - loss-based sanitisation **fails**, because consistently wrong labels have *low* loss.
- **A simple audit catches it** (E4 smoke run): per-class accuracy on a small trusted set of 100 clean images shows class 7 at **0%**, and the training-label frequencies show no 7s and twice as many 1s. Detection comes from **process** (a trusted evaluation set, data audits), not from the training loss.

**`experiments.py`:**
- **E1:** cost-model sensitivity to the success probabilities;
- **E2:** robustness curves for several adversarial-training strengths;
- **E3:** random flip rates up to 50%;
- **E4:** the trusted-set audit.
- Only the E4 smoke run was done here.

---

## 6. Why it matters

- **This report was one of the first broad, multi-institution analyses of AI misuse.**
- **Its recommendations show up in later practice:**
  - pre-deployment red teaming of models;
  - staged release and access controls for powerful models;
  - responsible disclosure of model vulnerabilities;
  - "dual-use" and misuse sections in model cards (see 093) and system cards.
- **Adversarial robustness and data-poisoning defence** remain active research areas.

---

## 7. Check yourself

1. What does "dual-use" mean for AI?
<details><summary>Answer</summary>The same capability or research can serve beneficial or harmful ends. E.g. software that finds vulnerabilities can be used to fix them or exploit them.</details>

2. Name the three changes to the threat landscape.
<details><summary>Answer</summary>Expansion of existing threats, introduction of new threats, and a change in the typical character of threats (more effective, targeted, hard to attribute, exploiting AI vulnerabilities).</details>

3. In our model, above what target value does a tailored attempt beat a generic one when c = 5?
<details><summary>Answer</summary>When 0.05·v − 5 > 0.002·v − 0.01, i.e. v > 4.99/0.048 ≈ 104.</details>

4. Why does cheaper tailoring increase harm even though the attacker spends less?
<details><summary>Answer</summary>More targets switch from the low-success generic attempt (or no attempt) to the high-success tailored one, so total expected success rises while the cost per target falls.</details>

5. Write the FGSM update and explain it.
<details><summary>Answer</summary>x_adv = clip(x + ε·sign(∇_x loss), 0, 1). Each input feature moves by ε in whichever direction increases the loss most quickly (to first order), giving a small perturbation that can flip the prediction.</details>

6. Why didn't loss-based sanitisation detect the targeted poisoning?
<details><summary>Answer</summary>All training 7s were consistently labelled 1, so the model learned "7-shaped digits are 1s" and those examples have low loss. Sanitisation only removes points that disagree with what the model learned.</details>

7. What simple process caught the targeted attack in our toy?
<details><summary>Answer</summary>Evaluating per-class accuracy on a small trusted clean set (class 7 accuracy 0%) and auditing training-label frequencies (no 7s, twice as many 1s).</details>

8. What are the report's four high-level recommendations?
<details><summary>Answer</summary>Policymakers collaborate with technical researchers; researchers take dual-use seriously; import best practices from fields like computer security; expand the range of stakeholders involved.</details>

9. Name two practices the report suggests importing from computer security.
<details><summary>Answer</summary>Any two of: red teaming, formal verification, responsible disclosure of AI vulnerabilities, security tools, secure hardware.</details>

10. Why do "anonymity and psychological distance" matter?
<details><summary>Answer</summary>Automation lets actors act remotely and anonymously, making attacks harder to attribute and possibly easier to carry out psychologically, which changes the character of threats.</details>
