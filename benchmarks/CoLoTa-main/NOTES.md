# CoLoTa — discussion notes

## Summary

| | |
|--|--|
| **What** | Long-tail **commonsense** QA / claim verification over obscure entities (yes/no or true/false) |
| **Size** | 3,300 items (`CoLoTa_qa.json` ≈1.6k + `CoLoTa_cv.json` 1,650) |
| **KG** | **Wikidata** (QIDs + gold triples; no dump date pinned) |
| **Gold fields** | `query`, `answer`, `kg_entities`, `kg_triples`, `inference_rule`, `reasoning_steps` |
| **For our RQs** | Strong for post-gen support checks & controlled KG context on **binary** long-tail+inference items; not open-ended |
| **Need `baselines/`?** | No — JSON only is enough |

Notes from our walkthrough of this benchmark (for KG-summary / LLM hallucination experiments).

---

## Example data entry

Random coherent QA item from `CoLoTa_qa.json` (`id = S87`):

```json
{
  "id": "S87",
  "query": "Could Paolo Cannavaro have suffered from middle child syndrome?",
  "answer": false,
  "kg_entities": {
    "Paolo Cannavaro": "Q311956"
  },
  "kg_triples": [
    "(Paolo Cannavaro, sibling, Fabio Cannavaro)"
  ],
  "inference_rule": "If Paolo Cannavaro has at least one older and one younger sibling, he could have suffered from middle child syndrome.",
  "reasoning_strategy": ["biological", "sports", "cultural"],
  "reasoning_steps": [
    {
      "step": "Paolo Cannavaro has only one brother, Fabio Cannavaro.",
      "evidence": "(Paolo Cannavaro, sibling, Fabio Cannavaro)"
    },
    {
      "step": "Since didn't have at least one older and one younger sibling, he couldn't have suffered from middle child syndrome."
    }
  ]
}
```

Claim-verification example from `CoLoTa_cv.json` (`id = C41`), including a **qualifier**:

```json
{
  "id": "C41",
  "query": "Francesco Totti and Ilary Blasi are experiencing a successful marriage.",
  "answer": false,
  "kg_entities": {
    "Francesco Totti": "Q20110",
    "Ilary Blasi": "Q287186"
  },
  "kg_triples": [
    "(Francesco Totti, spouse, Ilary Blasi {end time, 2022})"
  ],
  "inference_rule": "There must be no evidence of Francesco Totti and Ilary Blasi's romantic relationship to have ended to make the claim correct.",
  "reasoning_strategy": ["cultural"],
  "reasoning_steps": [
    {
      "step": "Based on the tiple 1, Francesco Totti and Ilary Blasi's romantic relationship ended in 2022, so the claim is false.",
      "evidence": ["(Francesco Totti, spouse, Ilary Blasi {end time, 2022})"]
    }
  ]
}
```

---

## What CoLoTa is

- Full name framing: **Commonsense reasoning over Long-Tail entities**.
- Paper / repo: *CoLoTa: A Dataset for Entity-based Commonsense Reasoning over Long-Tail Knowledge* ([arXiv:2504.14462](https://arxiv.org/abs/2504.14462), [D3Mlab/CoLoTa](https://github.com/D3Mlab/CoLoTa)).
- **3,300** queries total:
  - **QA** (`CoLoTa_qa.json`): 1,650 yes/no questions, rewritten from **StrategyQA**.
  - **Claim verification** (`CoLoTa_cv.json`): 1,650 true/false claims, rewritten from **CREAK**.
- Goal: stress-test LLMs (and KGQA methods) on **obscure entities**, where parametric memory is weak and **hallucinations / reasoning errors** rise.

## Knowledge graph

- Grounded in **Wikidata** (not DBpedia).
- Each item includes:
  - Natural-language **query**
  - **`kg_entities`**: labels → Wikidata **QIDs**
  - **`kg_triples`**: small gold supporting subgraph
  - **`answer`**, **`inference_rule`**, **`reasoning_steps`**, **`reasoning_strategy`**
- Some triples are **hyper-relational** (Wikidata **qualifiers**), not only plain `(s, p, o)`.

## “Latest Wikidata?”

- **No dump / version is pinned.**
- Facts were checked as present in Wikidata **at curation time**.
- For experiments that match the benchmark, prefer the **dataset’s frozen QIDs + `kg_triples`** as gold evidence.
- **Live Wikidata** (endpoint / rebuild neighborhoods) is fine for GSS retrieval, but may **drift** from the annotated triples.

## How we plan to use it (GSS experiment)

- We **do not need** the `baselines/` folder for our setup.
- We **do need** `CoLoTa_qa.json` and `CoLoTa_cv.json` (queries, QIDs, gold answers).
- Pipeline idea: take **query + QIDs** → query **Wikidata live** (e.g. build a GSS subgraph) → validate the LLM answer against gold **`answer`**.
- Optional: keep author **`kg_triples`** as a compact gold evidence set for comparison with larger live GSS neighborhoods.

## What `baselines/` is

- Authors’ **paper experiment code** (zero-/few-shot CoT LLM runs, Wikidata helpers, CSV variants under `data/`).
- Useful only to **reproduce their reported baselines**, not required to use the JSON dataset.

## Other properties worth remembering

1. **Not pure factoid QA** — needs Wikidata facts **plus** a commonsense inference rule (geo, temporal, numeric, occupations, etc.).
2. **Binary labels** — easier to score than open-ended generation benchmarks.
3. **Long-tail by design** — popular StrategyQA/CREAK entities were replaced with obscure Wikidata counterparts (popularity ≈ fewer Wikidata triples + search-hit checks).
4. **Gold subgraphs are tiny** — live GSS graphs will usually be much larger/noisier than `kg_triples`.
5. **Parallel originals** — paper contrasts popular vs long-tail versions; that contrast is available if we want it.
6. **Two tasks, same schema** — one pipeline can cover QA and CV.
7. For GSS framing: better as *“does the ACE support this yes/no claim about obscure entities?”* than *“retrieve the answer entity.”*

---

## Fit to research problems (general KG summarization; not GSS-specific)

### Problem 1 — KG summarization for hallucination detection (post-generation)

**Idea:** LLM answers first (`LLM(Q)=A`). Only then build `Summary(X)` with `X ∈ {A, S, Q}` from Wikidata and check whether `A` is supported by that summary. Correction runs only when a hallucination is detected.

#### 1. Example of how CoLoTa could be used

Take a CoLoTa QA item, e.g. roughly:

- **Q:** “Could you travel from Gujan to Aousserd only by car?”
- **S:** entities `{Gujan, Aousserd}` with QIDs from `kg_entities`
- **Gold:** `answer = false`, plus author `kg_triples` / reasoning steps

Experiment sketch:

1. Ask an LLM **Q** with no KG → get free-text **A** (e.g. “Yes, both are in the same region…”).
2. Build three candidate summaries from Wikidata (or from a larger retrieved neighborhood), conditioned on:
   - **Summary(Q)** — retrieve around entities/relations suggested by the question  
   - **Summary(S)** — retrieve around anchor subject QIDs only  
   - **Summary(A)** — ground entities/claims mentioned in the generated answer, then summarize  
3. Run a verifier: *Is A supported / contradicted / underspecified given Summary(X)?*
4. Score detection against CoLoTa gold: if gold is `false` and A said `true` (or invented wrong geography), that is a hallucination; a good summary should enable detection. Compare **Q vs S vs A** as summary seeds on accuracy, false alarms, and summary size (#triples / #tokens).
5. Optional oracle: use CoLoTa’s frozen `kg_triples` as an upper-bound “perfect tiny summary” to see how much of detection failure is **retrieval/summarization** vs **verification**.

Same protocol works on **CV**: treat the claim as Q (or as A if you first rewrite/expand it), verify support in Summary(X).

#### 2. How good is CoLoTa for this problem, and why?

**Overall: strong fit for post-generation detection, with caveats.**

| Strength | Why it helps this RQ |
|----------|----------------------|
| Long-tail entities | Parametric knowledge is weak → many real hallucinations; detection is not trivial “memorized fact” matching. |
| Wikidata QIDs + gold triples | Clear KG grounding; gold `kg_triples` = natural oracle for “how much KG is enough?” |
| Binary gold answers | Easy end-to-end labels for supported vs hallucinated decisions. |
| Annotated reasoning steps | Lets you analyze *where* detection fails (missing fact vs failed inference). |
| Parallel popular originals | Can show that detection/summarization difficulty rises on long-tail vs head entities. |

| Limitation | Impact |
|------------|--------|
| Answers are yes/no (or claim true/false), not long open prose | Claim extraction from long A is easier than on essay-style OKGQA; under-stresses “decompose A into atomic claims.” |
| Needs **commonsense** beyond triples | Support is not always “A appears as a triple”; Summary(X) must enable inference (continents, occupations, …). Pure triple-containment checks will underperform. |
| Gold subgraphs are tiny | Great for “minimum KG” studies; less realistic as the *only* retrieved neighborhood unless you also pull live Wikidata around QIDs. |
| Frozen triples vs live KG | Live summarization may miss or alter gold facts → confound detection with KG drift. |

**Verdict for Problem 1:** Very good benchmark to ask *which of Q / S / A yields the best summary for detecting hallucinations* and *how small a summary can still work*, especially under long-tail stress. Pair it with an explicit inference/verification step, not string match alone.

---

### Problem 2 — KG summaries as controlled knowledge for LLM QA (in-/context generation)

**Idea:** Instead of (or before) trusting parametric knowledge, give the LLM a **question-specific KG summary** and compare:

- Baseline: `Q → LLM → A`
- Summary-only: `Q → Summary(Q|S) → LLM(restricted) → A`
- Context augmentation: `Q + Summary → LLM → A` (parametric + external; knowledge conflict possible)

#### 1. Example of how CoLoTa could be used

Same Gujan / Aousserd-style item:

1. **Baseline:** prompt LLM with Q only; record A and whether it matches gold `true`/`false` (and optional hallucination rate vs gold triples / steps).
2. **KG Summary Only:** build Summary from Wikidata around **S** (QIDs) or **Q**; prompt the LLM to answer **using only** the provided triples (verbalized or structured). Measure accuracy and contradiction with gold.
3. **Context Augmentation:** provide Q + the same Summary without forbidding parametric knowledge; measure whether the model follows the KG, ignores it, or blends (knowledge conflict).
4. Ablate summary size (k-hop, top-n triples, oracle CoLoTa `kg_triples` vs larger live neighborhoods) to study *how much* external evidence is needed and *how much* the model should be allowed to lean on parameters.
5. Use **CV** as a second regime: “verify this claim given only Summary” vs “claim + Summary as soft context.”

#### 2. How good is CoLoTa for this problem, and why?

**Overall: good-to-strong fit for controlled KG context, especially long-tail + conflict analysis.**

| Strength | Why it helps this RQ |
|----------|----------------------|
| Long-tail focus | Baseline parametric answers should often fail → room to show Summary-only / augmentation gains. |
| Small gold subgraphs | Ideal “minimal controlled knowledge” condition: can the LLM answer correctly if given exactly the author triples (+ inference rule)? |
| QA + CV | Two prompting styles (answer vs verify) for the same schema. |
| Diverse reasoning strategies | Tests whether summaries help not just lookup but multi-step reasoning when verbalized carefully. |
| Popular vs long-tail parallels | Isolates “needs external KG” vs “already in parameters.” |

| Limitation | Impact |
|------------|--------|
| Binary targets | Measures factual correctness of a decision well; less suited to studying fluency / open-ended hallucination in long answers. |
| Commonsense gap | Even a perfect factual summary may omit the **inference rule**; you may need to provide triples only, or triples + rule, as separate conditions. |
| Not a classical SPARQL/KGQA logical-form set | Strong for NL+summary prompting; weaker if the research assumes executable queries as the summary form. |
| Knowledge conflict is real here | On long-tail items, augmentation may still defer to wrong parametric guesses—CoLoTa is good for measuring that, but needs careful conflict metrics (not only accuracy). |

**Verdict for Problem 2:** Well suited to compare baseline vs summary-only vs augmentation on Wikidata-grounded, long-tail yes/no (and claim) questions. Best use: treat CoLoTa `kg_triples` as the **controlled minimal context**, then scale up live summaries to study efficiency vs conflict.

---

### Cross-problem takeaway

CoLoTa is **particularly aligned** with both RQs because it was built to expose **hallucinations on long-tail Wikidata entities** while shipping **QIDs**, **gold subgraphs**, and **reasoning annotations**.  

- Problem 1 (detect after generation): use Q/S/A → Summary → verify A; gold answer + gold triples evaluate detectors.  
- Problem 2 (generate with controlled KG): use Summary(Q|S) as the only or additional context; gold answer evaluates whether restriction/augmentation reduces errors.  

Main shared caveat: success often requires **facts + inference**, so “A exists in Summary(X)” should be interpreted as **support under reasoning**, not literal triple membership.
