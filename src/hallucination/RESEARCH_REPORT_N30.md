# Research Report: GSS-Based Hallucination Detection — n=30 Stability Study

**Working title:** *Validating LLM Factual Answers with Query-Driven Knowledge Graph Summaries*  
**Companion to:** [`RESEARCH_PROPOSAL.md`](RESEARCH_PROPOSAL.md) (final proposal synthesizing **n=10 + n=30**).  
This file keeps the detailed n=30-only narrative; prefer the proposal for the combined research framing.

**Status:** n=30 stratified T-REx run completed (DBpedia, three GSS query modes).  
**Prior run:** archived under `out/hallucination/archive_n10_pilot/`.  
**Codebase:** GSSplusplus — experiment wrappers around **stock** Graph Semantic Summarization (GSS); algorithm unchanged.

---

### Abstract

This report presents a **30-item stability study** of post-generation hallucination detection that uses GSS answer-centered explanatory (ACE) subgraphs as symbolic evidence. An Ollama **Llama 3.1 8B** model answers FactEval-style T-REx WH-questions; stock GSS is queried in **answer**, **subject**, and **claim** modes over DBpedia; a post-hoc validator labels supported / weak_support / unsupported / uncertain. Exact gold match is ~**17/30**; claim mode yields the most strict supports (**3**), including one granularity case, while subject yields **2** strict + **2** weak true supports. **Answer mode collapses** (0 supports, **21/30** empty ACEs). Across all modes, **strict false_support remains 0**. Compared with the archived n=10 pilot, the qualitative story holds—subject/claim beat answer hubs—but **recall is low** and **uncertain rates are high**, so GSS-as-validator is precise when it fires and incomplete as a standalone detector. The study motivates subject/claim-first protocols, allowlist expansion, empty-ACE diagnostics, and baselines before further scale-up.

---

## 1. Motivation and research question

LLMs produce fluent answers that may be wrong or only loosely grounded. Knowledge graphs can supply explicit triples for post-generation checks. GSS builds compact, query-focused ACE subgraphs; we ask whether those subgraphs can **support, weakly support, or refuse to validate** a free-text answer **without modifying GSS**.

**Research question (unchanged).**  
*Given a natural-language question and an LLM free-text answer, can a stock GSS ACE subgraph provide reliable, interpretable evidence for claim validation?*

The n=10 pilot suggested subject/claim ≫ answer and zero false supports. This n=30 run tests whether that pattern **survives a larger, stratified sample**.

---

## 2. Approach (unchanged protocol)

### 2.1 Pipeline

```
Question Q  +  LLM answer A
        │
        ▼
  GSS query string ∈ {answer, subject, claim}
        │
        ▼
  Stock GSS → ACE subgraph S
        │
        ▼
  Post-hoc validator (outside GSS)
        │
        ▼
  supported | weak_support | unsupported | uncertain
  (+ detection labels vs gold)
```

**Constraint:** only the query string and post-hoc interpretation change; scoring, retrieval, and top-k selection inside GSS stay fixed.

### 2.2 Query modes

| Mode | GSS query | Intent |
|------|-----------|--------|
| **answer** | LLM answer text | Summarize the claimed answer entity |
| **subject** | Question subject | Summarize the asked entity; look for the claim |
| **claim** | Verbalized statement | Steer GSS with subject + answer wording |

### 2.3 Validation decisions

| Decision | Definition |
|----------|------------|
| **supported** | Subject + answer co-occur on an allowlisted relation predicate |
| **weak_support** | Subject + answer co-occur; predicate not allowlisted |
| **unsupported** | Non-empty ACE; no subject–answer co-occurrence |
| **uncertain** | Empty ACE / empty answer / pipeline failure |

**Gold (analysis only):** `gold_match` (fuzzy exact) ∨ granularity in ACE → `gold_acceptable`.  
**Detection labels:** `true_support`, `true_support_granularity`, `weak_true_support`, `missed_support`, `false_support`, `weak_false_support`, `not_validated`, `uncertain`.  
`not_validated` ≠ proven hallucination.

---

## 3. Experimental setup (n=30)

### 3.1 Data

| Item | Choice |
|------|--------|
| Benchmark | LAMA T-REx / FactEval-style WH questions |
| Sample | `dataset/facteval_trex/sample_30.json` |
| Construction | Round-robin over 10 relations (P19, P20, P27, P36, P106, P17, P159, P495, P740, P937), **3 items each**, seed **2026** |
| Mistakes | Natural LLM errors only |
| Archive of n=10 | `out/hallucination/archive_n10_pilot/` |

Sampler update: `prepare_trex_sample.py` now supports n>10 via stratified round-robin (previously capped at one item per relation).

### 3.2 Models and endpoints

| Role | Choice |
|------|--------|
| QA LLM + GSS entity extraction | Ollama `llama3.1:8b` @ `http://127.0.0.1:11434` |
| Similarity | `sentence-transformers` `all-MiniLM-L6-v2` |
| KG | DBpedia SPARQL `https://dbpedia.org/sparql` |
| GSS entry | `pipeline.pipeline(query)` via `run_stock_gss` |

### 3.3 Procedure and wall time

1. Stage 1: `answer_facteval.py` → `out/hallucination/llm_answers.json` (~45 s).  
2. Stage 2: `validate_with_gss.py --modes answer subject claim` → `validation_results_{mode}.json`.  
3. Wall time Stage 2: **~106 minutes** for 90 GSS runs (30×3).

---

## 4. Stage 1 results (LLM answers)

Approximate gold agreement (fuzzy / containment-style string match used for reporting): **17/30**.

Examples of correct short answers: Nebraska→Lincoln, Schneerson→Rabbi, Mackenzie→Edinburgh, Strambi→Rome, Akhmetov→Kazakhstan, Ginza→Japan, Lindenberg→Hamburg.  
Natural errors: Grymalska born→Poland (gold Kiev), Spitzer citizenship→Hungary (gold Israel), Munich Residence→Austria (gold Germany), Pulse created→Canada (gold Japan), Dirge Within→United Kingdom (gold Chicago).  
Messy free text: Crossley Motors answer was a full sentence naming Rochdale rather than Manchester.

Pilot LLM accuracy rose vs n=10 exact rate (4/10), but the item set differs (new seed), so rates are not directly comparable.

---

## 5. Evaluation metrics (reminder)

| Metric | Meaning |
|--------|---------|
| `#supported` / `#weak_support` / `#unsupported` / `#uncertain` | Validator decisions on ACE |
| `true_support` | Exact gold + strict supported |
| `true_support_granularity` | Granularity-acceptable + strict supported |
| `weak_true_support` | Gold-acceptable + weak_support |
| `missed_support` | Gold-acceptable but unsupported (coverage miss) |
| `false_support` | Not gold-acceptable but strict supported (**dangerous**) |
| `not_validated` | Wrong/unacceptable answer and no ACE support |
| `uncertain` | No usable ACE |

---

## 6. Results (n=30 × 3 modes)

### 6.1 Summary table

| Metric | answer | subject | claim |
|--------|--------|---------|-------|
| supported | 0 | 2 | 3 |
| weak_support | 0 | 2 | 0 |
| unsupported | 9 | 13 | 10 |
| uncertain | **21** | 13 | 17 |
| true_support | 0 | 2 | 2 |
| true_support_granularity | 0 | 0 | 1 |
| weak_true_support | 0 | 2 | 0 |
| missed_support | 6 | 7 | 6 |
| **false_support** | **0** | **0** | **0** |
| weak_false_support | 0 | 0 | 0 |
| not_validated | 3 | 6 | 4 |
| gold_match | 17 | 17 | 17 |
| gold_acceptable | 17 | 17 | 18 |
| granularity_match | 0 | 2 | 3 |

### 6.2 Comparison with archived n=10 (subject mode)

| | n=10 subject | n=30 subject |
|--|--------------|--------------|
| supported | 3 | 2 |
| weak_support | 1 | 2 |
| uncertain | 1 | **13** |
| false_support | 0 | 0 |
| missed_support | 1 | 7 |
| gold_acceptable | 5 | 17 |

The **zero false_support** pattern replicates. Absolute support counts do not scale linearly: empty ACEs and missed links dominate the larger set.

### 6.3 Qualitative highlights

**Strict successes (subject):** Vincent Strambi died in Rome; Revolverheld formed in Hamburg — subject ACE retained allowlisted deathPlace / formation links.

**Weak successes (subject):** Serik Akhmetov–Kazakhstan and Always Greener–Australia — entities co-occur, predicate outside allowlist (`weak_true_support`).

**Claim successes:** Henry Mackenzie born Edinburgh; Revolverheld–Hamburg; Crossley Motors HQ coded `true_support_granularity` despite a verbose/Rochdale-flavored LLM string (gold Manchester) — treat as an **audit case** for granularity rules.

**Answer-mode failure mode:** 21/30 empty subgraphs. Hub or ambiguous answer strings (France, Japan, Rome, Hungary, Coventry, …) often yield no retained triples after pruning, so the validator correctly returns **uncertain**, not false accept.

**Persistent misses:** Correct answers with non-empty ACEs that lack subject–answer co-occurrence (e.g. Ginza–Japan under subject; Udo Lindenberg–Hamburg under subject/claim) → `missed_support`.

**Honest abstention on wrong answers:** Wrong answers without linking evidence → `not_validated` (e.g. Poland vs Kiev; Monaco vs Naples under subject).

---

## 7. Discussion

### 7.1 What the n=30 study confirms

1. **Query formulation dominates.** Answer-centered queries are a poor validation protocol (0 supports, mostly uncertain). Subject and claim remain the only viable modes in this design.  
2. **Precision when firing is high.** Still **no strict false_support** at n=30.  
3. **Coverage is the bottleneck.** High `uncertain` + `missed_support` means many gold-correct answers are never confirmed — GSS summarization / linking / allowlists, not over-acceptance, is the main failure.

### 7.2 What changed vs n=10

- Empty ACE rate is much higher, especially in answer mode (and still substantial in subject/claim). Long-tail subjects and brittle DBpedia retrieval matter more at scale.  
- Weak_support remains useful for citizenship / country-of-origin style predicates outside tight allowlists.  
- Absolute “wins” stay small (handful of supports); the method is a **conservative filter**, not a high-recall detector.

### 7.3 Is it still worth pursuing?

**Yes, with a narrowed protocol:**

| Keep | Drop / demote |
|------|----------------|
| Subject + claim as primary | Answer mode as main detector |
| Conservative decisions + granularity auditing | Treating unsupported as hallucination |
| Allowlist + weak tier | Expecting high recall from stock GSS alone |

**Next experiments (priority order):**

1. **Diagnose empty ACE** (URI resolution failures vs pruning) on the 13–21 uncertain cases.  
2. **Expand relation allowlists** (especially P17/P27/P495) to convert weak_true_support → true_support where appropriate.  
3. **Baseline:** raw 1-hop SPARQL for subject–answer–predicate vs GSS ACE (isolates summarization value).  
4. Optional: freeze answer mode as ablation only to cut runtime ~⅓ on future runs.

### 7.4 Limits

- n=30 still small for confidence intervals.  
- Public DBpedia latency/variance.  
- One LLM (`llama3.1:8b`).  
- Granularity on verbose answers needs manual spot-checks.  
- No comparison yet to NLI / LLM-as-judge.

---

## 8. Conclusion

On a stratified 30-item T-REx sample, stock GSS ACE subgraphs remain a **precise but low-coverage** post-hoc evidence source for LLM answers when queries use **subject** or **claim** formulations. **Answer mode is not viable** as a detector (mostly empty ACEs). **False support stays at zero**, while missed support and uncertainty dominate—exactly the profile expected of a conservative symbolic validator wrapped around an unchanged summarizer. The research direction remains interesting if framed as **explainable abstention / support tagging**, not as a complete hallucination oracle; the next milestone should be empty-ACE diagnostics, allowlist refinement, and a non-GSS retrieval baseline rather than another blind increase in n.

---

## 9. Artifact map

| Path | Description |
|------|-------------|
| `dataset/facteval_trex/sample_30.json` | Stratified 30 questions (seed 2026) |
| `out/hallucination/llm_answers.json` | Stage-1 Llama answers |
| `out/hallucination/validation_results_{answer,subject,claim}.json` | Per-mode validation |
| `out/hallucination/run_n30_validation.log` | Full Stage-2 console log |
| `out/hallucination/archive_n10_pilot/` | Archived n=10 sample + results |
| `src/hallucination/prepare_trex_sample.py` | Stratified sampler (supports n>10) |
| `src/hallucination/validate_with_gss.py` | Experiment runner |
| `src/hallucination/validation.py` | Post-hoc decisions |
| `src/pipeline/claim_validation.py` | Query construction + stock `pipeline()` |
