# KG-FPQ — discussion notes

## Summary

| | |
|--|--|
| **What** | **False-premise questions** (FPQs) that smuggle a wrong KG fact into the question to induce factuality hallucination |
| **Size** | ~178k FPQs (paper); local JSON ≈14.8k base records × 6 FPQ levels × {YN, WH} across art/people/place |
| **KG** | **KoPL / KQA Pro Wikidata subset** (frozen; not live Wikidata) |
| **Gold fields** | `Ttriple`, `Ftriple_*`, `TPQ`, `FPQ_1`…`FPQ_6`, optional `path_*` / `hop_*` |
| **For our RQs** | Strong for detecting/resisting **premise-induced** hallucinations and KG-vs-question conflict |
| **Need `baselines/`?** | No such folder; JSON under `KG-FPQ-data/` is enough |

Paper: *KG-FPQ: Evaluating Factuality Hallucination in LLMs with Knowledge Graph-based False Premise Questions* (COLING 2025; [arXiv:2407.05868](https://arxiv.org/abs/2407.05868)).  
Repo: [yanxuzhu/KG-FPQ](https://github.com/yanxuzhu/KG-FPQ) (dataset released; **code “coming soon”** at time of install).

---

## Example data entry

Random item from `KG-FPQ-data/people_YN.json` (`id = 1f6f3514-0627-11ef-b8cb-e8611f487d64`). True premise: Batman appears in *The Dark Knight*; FPQs replace the work with confusable false objects.

```json
{
  "Ttriple": ["Batman", "present in work", "The Dark Knight"],
  "Ftriple_6": ["Batman", "present in work", "Citizen Kane"],
  "path_6": [
    ["Batman", "performer", "George Clooney"],
    ["George Clooney", "award received", "Golden Globe Cecil B. DeMille Award"],
    ["Golden Globe Cecil B. DeMille Award", "award disciplines or subjects", "film"],
    ["film", "model item", "Citizen Kane"]
  ],
  "hop_6": 4,
  "Ftriple_5": ["Batman", "present in work", "Denmark"],
  "Ftriple_1": ["Batman", "present in work", "Mel Gibson"],
  "head": "Batman",
  "domain": "people",
  "counts_of_head": 104,
  "subgraph_size": 16,
  "id": "1f6f3514-0627-11ef-b8cb-e8611f487d64",
  "TPQ": "Is Batman present in the work The Dark Knight?",
  "FPQ_6": "Is Batman present in the work Citizen Kane?",
  "FPQ_5": "Is Batman present in the work Denmark?",
  "FPQ_4": "Is Batman present in the work The Specialist?",
  "FPQ_3": "Is Batman present in the work playback singer?",
  "FPQ_2": "Is Batman present in the work Ghost Rider: Spirit of Vengeance?",
  "FPQ_1": "Is Batman present in the work Mel Gibson?"
}
```

Ideal behavior: answer **Yes** on `TPQ`, **No** / refuse / correct on `FPQ_*` (do not invent how Batman appears in *Citizen Kane*).

---

## What it is

**KG-FPQ** = **Knowledge Graph–based False Premise Questions**.

- Goal: measure how often LLMs suffer **factuality hallucination** when the **question itself smuggles in a wrong fact**.
- Scale (paper): ~**178k** FPQs from ~**14.8k** base triples.
- Local JSON (under `KG-FPQ-data/`):

  | File | Domain | Format | Items (each with 6 FPQ levels) |
  |------|--------|--------|--------------------------------|
  | `art_YN.json` / `art_WH.json` | Art | Yes/No & WH | 4,969 |
  | `people_YN.json` / `people_WH.json` | People | Yes/No & WH | 4,897 |
  | `place_YN.json` / `place_WH.json` | Place | Yes/No & WH | 4,994 |

- **Two task formats**
  - **YN**: discriminative yes/no (“Did Spielberg receive …?”).
  - **WH**: generative (“For what achievement was Spielberg awarded …?”).
- Each record stores (exact JSON keys): `Ttriple`, `Ftriple_1`…`Ftriple_6`, optional `path_5`/`path_6` + `hop_5`/`hop_6`, `head`, `domain`, `counts_of_head`, `subgraph_size`, `id`, `TPQ`, `FPQ_1`…`FPQ_6`.

---

## False Premise Questions (FPQ) — how this works (extended)

### Intuition

Everyday questions quietly assume facts. A **premise** is that assumed fact.

- Honest question (true premise): “When did **Paris** become the capital of France?”  
  Premise: Paris *is* the capital of France → true.
- **False premise question (FPQ):** “When did **Lyon** become the capital of France?”  
  Premise: Lyon *is* the capital of France → **false**.

An ideal model should **notice the bad assumption** (refuse, correct, or ask for clarification). Many LLMs instead **play along** and invent a fluent answer → that is the **factuality hallucination** KG-FPQ targets.

So KG-FPQ is **not** mainly “does the model know random trivia?” It is: **when we deliberately poison the question with a KG-false fact, does the model get fooled?**

### Pipeline (authors’ construction)

```
 true KG triple  →  edit object  →  false triple  →  GPT verbalizes questions
      Ttriple            Ftriple_*              TPQ / FPQ_*
```

1. **Take a true triple** from the KG, e.g.  
   `Ttriple = (Steven Spielberg, award received, Order of the British Empire)`  
   This *is* in the KG.

2. **Edit only the object** into something else that is **not** the true award for that subject under that relation, e.g.  
   `Ftriple_6 = (Steven Spielberg, award received, United States Department of the Treasury)`  
   Same subject + relation, wrong object → **false premise**.

3. **Generate natural language** with GPT:
   - From the **true** triple → **TPQ** (true-premise question).  
     YN example: *“Did Steven Spielberg receive the Order of the British Empire award?”*
   - From the **false** triple → **FPQ** (same template, swapped object).  
     YN example: *“Did Steven Spielberg receive the United States Department of the Treasury award?”*

4. Repeat the object edit with **six different strategies** (see below) → six FPQs per base triple (`FPQ_1` … `FPQ_6`), graded by how **confusable** the false object is.

5. Optionally keep a **path** in the KG from the subject to the false object (`path_6`, `hop_6`, …) when the edit uses a graph neighbor — useful later for “how far is the distractor?”

### What “correct behavior” looks like

| Input | Desired model behavior | Hallucination (bad) |
|-------|------------------------|---------------------|
| **TPQ** | Answer using the true fact (YN: Yes / WH: real award context) | Wrong denial or wrong object |
| **FPQ** | Spot that the premise is false: say No, refuse, or correct the entity | Treat the false award as real and invent details |

Paper-style evaluation often **pre-checks** that the model can handle the **TPQ** (it “knows” the topic), then measures how badly it fails on the matched **FPQ**.

### Confusability levels (why six FPQs?)

False objects are not random strings. They are chosen from the KG so some look **plausibly related** to the subject:

| Key | Name (README) | Rough idea | Harder? |
|-----|---------------|------------|---------|
| FPQ_6 | Neighbor–Same–Concept (NSC) | Object is near the subject in the graph **and** same concept type as the true object | Often hardest |
| FPQ_5 | Neighbor–Different–Concept (NDC) | Near subject, different concept | Hard |
| FPQ_4 | Not-Neighbor–Same–Concept (NNSC) | Not a close neighbor; same concept as true object | Medium |
| FPQ_3 | Not-Neighbor–Different–Concept (NNDC) | Farther; different concept | Easier |
| FPQ_2 | Not-Neighbor–Same–Relation (NNSR) | Linked via similar relation patterns | Medium/varies |
| FPQ_1 | Not-Neighbor–Different–Relation (NNDR) | Looser association | Often easiest |

**Takeaway:** FPQ_6 is like swapping in a **nearby, same-type** entity (easy to confuse); FPQ_1 is a more **obviously weird** swap.

### Tiny walkthrough (YN, people)

1. KG says: Spielberg —award received→ Order of the British Empire.  
2. Authors replace the object with “US Department of the Treasury” (a multi-hop neighbor in their edit scheme).  
3. Ask the LLM the FPQ.  
4. If it answers “Yes” or elaborates how he got that “award,” it **accepted the false premise** → counts as hallucination under this benchmark.  
5. A KG summary that lists his **real** awards (and not Treasury as an award) should let a **verifier** mark that answer unsupported / contradicted.

### YN vs WH

- **YN:** model should often answer **No** (or equivalent) on FPQs; easy automatic scoring.  
- **WH:** question asks for details *as if* the false fact were true (“For what achievement was he awarded X?”). Refusing or correcting is still ideal; scoring free text needs a judge (paper: FPQ-Judge; not in this folder yet).

---

## Which KG? Which version?

KG-FPQ builds triples from what the paper calls **“KoPL, a high-quality subset of Wikidata.”** That wording mixes two related things (see next section). Practically:

- Gold triples in the JSON are **frozen** from that **Wikidata-derived KB** used with KoPL / **KQA Pro**.
- **Not** “whatever live Wikidata returns today,” and **no dump timestamp** is given beyond that fixed subset.
- For experiments: use **`Ttriple` / `Ftriple_*` as ground truth** for the premise. Live Wikidata is optional for larger neighborhoods around `head`.

---

## KoPL primer (what it is, download, contents, usage)

### Name collision (read this first)

| Name | What it actually is |
|------|---------------------|
| **KoPL** | **Knowledge-oriented Programming Language** — a small functional language (≈27 operators) to query a structured KB (find entity, follow relation, filter by concept, count, compare, …). Repo: [THU-KEG/KoPL](https://github.com/THU-KEG/KoPL). Docs: [kopl.xlore.cn](https://kopl.xlore.cn). |
| **KQA Pro KB** | A **dense Wikidata subset** released with the **KQA Pro** dataset (Cao et al.), stored as `kb.json`. Seeded from FB15k-237 entities aligned to Wikidata (+ extras). This is the **graph of facts**. Baselines repo: [shijx12/KQAPro_Baselines](https://github.com/shijx12/KQAPro_Baselines). |

KG-FPQ’s paper says they use “KoPL … as our KG.” In practice they mean: **triples taken from the KoPL-executable Wikidata subset (KQA Pro–style KB)**, not “the programming language is a graph.” For your work you care about **`kb.json` facts** (and/or live Wikidata); KoPL the language is only needed if you want to **run KoPL programs** over that KB.

### What the KB contains (KQA Pro `kb.json`)

Typical structure (simplified):

- **`concepts`**: concept id → name, instance lists, subclass links.  
- **`entities`**: entity id → name, attributes, relations, **qualifiers**.  
- Richer than plain `(s,p,o)` only: literals/attributes and qualifier annotations appear (Wikidata-like).  
- Design goals of KQA Pro KB: smaller than full Wikidata, but **dense** and multi-knowledge-type (relational + literal + qualifier), good for compositional QA.

Approximate scale (KQA Pro paper order of magnitude): tens of thousands of entities/concepts (not full Wikidata’s millions). Exact counts are in the KQA Pro paper tables once you unzip `kb.json`.

**Domains used inside KG-FPQ:** they further filter this KB to **Art / People / Place** concepts and selected relations (see paper appendix), then extract true triples and edit objects.

### How to download

**A. Dataset triples only (enough for most KG-FPQ experiments)**  
Already local: `benchmarks/KG-FPQ-main/KG-FPQ-data/*.json`.  
You do **not** need KoPL installed to read TPQ/FPQ and `Ttriple`/`Ftriple`.

**B. Full KQA Pro KB + questions (if you want the underlying graph file)**

1. Download the official bundle (from KQA Pro baselines README):  
   https://cloud.tsinghua.edu.cn/f/04ce81541e704a648b03/?dl=1  
2. Unzip into e.g. `./dataset/` so you have:

```text
dataset/
  kb.json      ← the knowledge graph
  train.json
  val.json
  test.json
```

3. Optional HF mirrors of the QA splits exist (e.g. community `kqa_pro` datasets); the **canonical KB file** is still that `kb.json` from the Tsinghua package / KQA Pro release.

**C. KoPL engine (only if you want to execute KoPL programs)**

```bash
# lighter package
pip install KoPL tqdm

# or high-performance engine (Linux/macOS; see docs)
pip install kopl-engine
```

Docs / install: https://kopl.xlore.cn/en/doc/3_install.html  
GitHub: https://github.com/THU-KEG/KoPL  

### How to use it (practical recipes)

**1. Use KG-FPQ JSON alone (recommended starting point)**  
- Load `people_YN.json` (etc.).  
- Q = `FPQ_k` or `TPQ`; gold premise facts = `Ttriple` vs `Ftriple_k`.  
- Build your own summary from live Wikidata **or** from a local graph keyed by entity names/ids.

**2. Use `kb.json` as a local closed-world KG**  
- Parse entities/relations into NetworkX, RDF, or a triple store.  
- Look up `head` / object labels from KG-FPQ in `kb.json` and extract k-hop neighborhoods for Summary(S)/Summary(Q).  
- Advantage: **stable**, same world the authors edited; no live Wikidata drift.

**3. Use KoPL language on `kb.json`**  
- `engine.init("kb.json")` (kopl-engine) or follow KoPL “hello world” KB format.  
- Write/parse a KoPL program (Find → Relate → FilterConcept → …) and `forward` to get answers.  
- Useful for KBQA baselines; **not required** for summarization + LLM hallucination studies if you only need triples.

**4. Use live Wikidata instead**  
- Map labels/QIDs and SPARQL around the subject.  
- Faster to plug into existing Wikidata tooling, but **may disagree** with frozen `Ttriple`/`Ftriple` if Wikidata changed.

### Do you need KoPL for our research problems?

| Goal | Need KoPL language? | Need `kb.json`? | Need KG-FPQ JSON? |
|------|---------------------|-----------------|-------------------|
| Detect FPQ hallucinations with a KG summary | No | Optional (nice closed world) | **Yes** |
| Controlled LLM context from KG summary | No | Optional | **Yes** |
| Reproduce KQA Pro-style program execution | **Yes** | **Yes** | No |

**Bottom line:** For summarization + hallucination work, treat **KG-FPQ JSON as the benchmark** and treat **KoPL/KQA Pro `kb.json` as an optional local Wikidata subset** behind those triples—not as something you must install on day one.

### Useful links

| Resource | URL |
|----------|-----|
| KoPL language repo | https://github.com/THU-KEG/KoPL |
| KoPL docs / site | https://kopl.xlore.cn |
| KQA Pro paper | https://arxiv.org/abs/2007.03875 |
| KQA Pro baselines (+ dataset download link in README) | https://github.com/shijx12/KQAPro_Baselines |
| KG-FPQ paper | https://arxiv.org/abs/2407.05868 |
| KG-FPQ repo | https://github.com/yanxuzhu/KG-FPQ |

---

## Baselines folder?

- **There is no `baselines/` folder** in this clone.
- Upstream README: dataset is public; **generation/eval code “coming soon”** (including FPQ-Judge for WH).
- For new experiments you only need **`KG-FPQ-data/*.json`** (and optionally KQA Pro `kb.json` or live Wikidata for summaries). No author baseline code required.

---

## Other points worth mentioning

1. **Hallucination type is specific:** induced by **accepting a false premise**, not open-ended unsupported invention alone. A good system should **reject or correct** the premise, not answer as if it were true.
2. **Paired TPQ/FPQ:** same surface pattern; only the edited object changes → clean ablations on confusability and summary seeds.
3. **Entities are often famous** (Spielberg, etc.), unlike CoLoTa’s long-tail focus — parametric knowledge is stronger; FPQs still fool models.
4. **Paths for neighbor edits** (`path_6`, `path_5`, …) are useful as **structural hints** for graph distance / summary radius studies.
5. **WH scoring is harder** than YN; the paper uses an automatic **FPQ-Judge** (not shipped here yet).
6. Duplicate copy may exist at `benchmarks/KG-FPQ-data/`; prefer **`benchmarks/KG-FPQ-main/`** as the canonical tree.
7. Scale is large (~178k FPQs) — usually **subsample** by domain and confusability level for pilots.

---

## Fit to research problems

### Problem 1 — KG summarization for hallucination detection (post-generation)

**Setup reminder:** `LLM(Q)=A` first; then `Summary(X)` with `X ∈ {A, S, Q}`; check whether `A` is supported; correct only if hallucination detected.

#### Example use

Item (people, YN), true triple `(Steven Spielberg, award received, Order of the British Empire)`:

- **TPQ:** “Did Steven Spielberg receive the Order of the British Empire award?”
- **FPQ_6:** “Did Steven Spielberg receive the United States Department of the Treasury award?”
- **S:** `Steven Spielberg` (and optionally the false/true object entities)

Protocol:

1. Ask the LLM the **FPQ** (no KG) → get **A** (e.g. invents a justification for a non-award).
2. Build summaries:
   - **Summary(Q):** retrieve Wikidata/KoPL facts around entities mentioned in the question (Spielberg + “US Treasury”).
   - **Summary(S):** neighborhood of subject only.
   - **Summary(A):** ground entities/claims in the model’s answer, then summarize.
3. Verify: does Summary(X) **support**, **contradict**, or **leave open** A? Gold: the premise is false (`Ftriple` ∉ KG; `Ttriple` is the true fact) → answers that accept the premise are hallucinations.
4. Compare Q vs S vs A as summary seeds; ablate summary size vs `subgraph_size` / paths; sweep **FPQ_1…6** to see if harder confusability needs larger summaries.
5. Oracle: feed only `Ttriple` (+ maybe path) as a minimal summary — upper bound on “how little KG is enough to reject the premise.”

#### How good is it? Why?

**Strong fit for post-generation detection of premise-induced hallucinations.**

| Strengths | Limitations |
|-----------|-------------|
| Explicit false vs true triples = clear gold for “unsupported premise.” | Less about open-ended long-form claim splitting than about **premise validity**. |
| Six confusability levels = controlled hardness for “how much KG is needed.” | Famous entities → parametric priors may dominate unless you force KG-only verification. |
| Paths / hops support distance-aware summarization studies. | KoPL ≠ full Wikidata; live summaries may include extra noise or missing KoPL edges. |
| YN easy to score; huge scale for stats. | WH needs a judge/rubric; author judge code not in-repo yet. |
| Directly matches “detect then correct” overhead story. | Correction target is often **reject/rewrite the question**, not fill a missing entity. |

**Verdict:** Excellent for RQ *“Can a Q/A/S-specific KG summary detect that A accepted a false premise?”* and for comparing summary seeds and KG size. Pair with CoLoTa if you also need **long-tail + commonsense** detection.

---

### Problem 2 — KG summaries as controlled knowledge for LLM QA

**Setup reminder:** compare Baseline `Q→LLM→A`, Summary-only `Q→Summary→LLM→A`, and Context augmentation `Q+Summary→LLM→A`.

#### Example use

Same Spielberg FPQ / TPQ pair:

1. **Baseline:** answer FPQ with parameters only (expect many hallucinations that accept the false award).
2. **Summary-only:** give a compact KG summary around S (or Q entities)—e.g. Spielberg’s real `award received` tails from KoPL/Wikidata—and require the model to answer **only** from that summary (should refuse or say the premise is false).
3. **Augmentation:** Q + same summary without forbidding parameters → measure **knowledge conflict** (does the model stick to the false premise in Q or follow the KG?).
4. Controls:
   - Summary = oracle `{Ttriple}` only vs larger k-hop neighborhood vs neighborhood that **accidentally includes** confusing neighbors (related to FPQ_5/6 paths).
   - Run matched **TPQ** to ensure the model *can* answer when the premise is true (paper-style knowledge pre-check).
5. Metrics: premise rejection rate, answer accuracy on TPQ, conflict rate on FPQ under augmentation, summary size.

#### How good is it? Why?

**Very good for controlled-context / conflict studies on false premises; narrower for general open QA.**

| Strengths | Limitations |
|-----------|-------------|
| Natural stress test for “does the model obey the KG summary or the false question text?” | Questions are **adversarial by construction**; gains may not transfer to ordinary factual QA. |
| True triple = minimal sufficient summary for many YN items. | Binary/short answers; limited study of long grounded generation. |
| Confusability levels ≈ difficulty of ignoring distractors in a summary. | Need careful prompting so “summary-only” is actually enforced. |
| TPQ/FPQ pairs separate “has knowledge” vs “resists false premise.” | Domain coverage only Art / People / Place via KoPL filters. |

**Verdict:** Strong benchmark for *whether restricting or augmenting with a question-specific KG summary reduces premise-induced hallucinations*, and for studying **how much** external evidence is needed vs parametric trust. Less ideal as the sole benchmark if the main goal is open-ended long-tail QA without adversarial premises (use CoLoTa / OKGQA there).

---

## Cross-problem takeaway

| Dimension | KG-FPQ |
|-----------|--------|
| KG | KoPL ⊂ Wikidata (frozen subset; not “latest Wikidata”) |
| Core phenomenon | False-premise factuality hallucination |
| Best for | Summary-based **detection** and **controlled generation** under adversarial Q |
| Data needed | `KG-FPQ-data/*.json` only; no baselines folder |
| Complements | CoLoTa (long-tail + commonsense, non-adversarial premises) |

Interpret “A exists in Summary(X)” here as: **Summary(X) supports accepting the premise / the answer**—for FPQs, a faithful summary of the true KG should typically **not** support A if A endorses the false premise.
