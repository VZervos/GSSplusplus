# Research Proposal: KG-Grounded Hallucination Detection via Answer-Centered Explanatory Subgraphs (GSS)

**Working title:** *Validating LLM Factual Answers with Query-Driven Knowledge Graph Summaries*

**Status:** Two empirical stages completed — **n=10 pilot** (archived) and **n=30 stratified stability study**.  
**Codebase:** GSSplusplus — hallucination experiment wrappers around **stock** Graph Semantic Summarization (GSS); the GSS algorithm is unmodified.  
**Experiment log:** [`README.md`](README.md)  
**Detailed n=30 companion:** [`RESEARCH_REPORT_N30.md`](RESEARCH_REPORT_N30.md)

---

### Abstract

Large language models often produce fluent but ungrounded answers. This proposal studies **post-generation hallucination detection** that treats Graph Semantic Summarization (GSS) **answer-centered explanatory (ACE)** subgraphs as interpretable symbolic evidence, **without changing the GSS algorithm**. Given a factual question and a free-text LLM answer, stock GSS is run under three query formulations—**answer**, **subject**, and **claim**—over DBpedia; a post-hoc validator then labels ACE evidence as supported, weak_support, unsupported, or uncertain. Evaluation separates exact gold match from granularity-aware KG agreement and distinguishes missed support from false support.

We report two FactEval-style T-REx experiments with Llama 3.1 8B (Ollama): an exploratory **n=10** pilot and a stratified **n=30** stability study (3 items × 10 relations). Across both, **subject and claim modes dominate answer-hub queries**, and **strict false_support remains 0**. The larger study shows the method is **precise when it fires but low-coverage**: empty ACEs and missed linking triples dominate, especially in answer mode (21/30 uncertain). We conclude that GSS-ACE validation is research-worthy as an **explainable support / abstention signal**, not as a high-recall hallucination oracle, and outline next steps: empty-ACE diagnostics, allowlist refinement, non-GSS retrieval baselines, and downstream abstention in QA.

---

## 1. Motivation and problem statement

Large language models achieve strong performance on open-domain question answering, but they still **hallucinate**: they produce fluent answers that are factually wrong, underspecified, or only loosely related to grounded knowledge. This is especially problematic where answers should be checkable against structured, trustworthy sources.

Knowledge graphs (KGs) such as DBpedia and Wikidata provide explicit triples that can, in principle, support or refute a claimed answer. Recent surveys on reducing LLM hallucination with KGs organize methods by when the KG intervenes (training, inference, or **post-generation**). This proposal focuses on **post-generation detection**: the LLM answers first; a KG-derived structure is then used to assess whether the answer is supported.

**Gap.** Most KG-based detectors either (i) retrieve raw triples / subgraphs without a summarization objective, or (ii) treat summarization as a separate problem (workload-based or template-based). GSS (Graph Semantic Summarization) was designed to build **ACE subgraphs**—compact, query-focused summaries that preserve explanatory context. We ask whether those summaries can serve as **symbolic evidence** for validating free-text LLM answers.

**Research question.**  
*Given a natural-language question and an LLM’s free-text answer, can a stock GSS ACE subgraph—produced without changing the GSS algorithm—provide reliable, interpretable evidence to support, weakly support, or refuse to validate that answer?*

**Empirical program.**  
1. **Pilot (n=10):** establish protocol, metrics, and early mode comparisons.  
2. **Stability study (n=30):** test whether pilot patterns hold under stratified sampling and more long-tail entities.

---

## 2. Proposed approach

### 2.1 High-level idea

We do **not** use GSS to answer the question. We use it to **construct a small explanatory subgraph**, then apply a **post-hoc symbolic validator** outside GSS:

```
Question Q  +  LLM answer A
        │
        ▼
  Choose GSS query string
  (answer | subject | claim)
        │
        ▼
  Stock GSS → ACE subgraph S
        │
        ▼
  Post-hoc check:
  Does S contain subject–relation–answer evidence?
        │
        ▼
  Decision: supported | weak_support | unsupported | uncertain
  (+ analysis labels vs gold)
```

**Constraint (methodological):** the GSS algorithm (entity extraction → SPARQL retrieval → one-hop expansion → importance/similarity scoring → top-k selection) remains **unchanged**. Experiment logic only (1) selects the natural-language string passed into `pipeline(query)`, and (2) interprets the returned triples.

### 2.2 Query modes

Because GSS is query-driven, *what* we pass as the query strongly affects which entity the summary is about:

| Mode | GSS query | Intent |
|------|-----------|--------|
| **answer** | LLM answer text (e.g. `France`) | Summarize the claimed answer entity |
| **subject** | Question subject (e.g. `Catherine Tasca`) | Summarize the asked entity; see if the claim appears |
| **claim** | Verbalized claim (e.g. `Catherine Tasca was born in France.`) | Steer GSS with the full statement |

**Hypothesis (confirmed by both experiments):** subject and claim are better suited to claim validation than answer, because answer-hub summaries (London, Canada, France, Japan, …) rarely retain long-tail subjects and often collapse to empty ACEs after pruning.

### 2.3 Post-hoc validation (outside GSS)

Let \(s\) be the question subject, \(a\) the claimed answer, and \(R\) the FactEval/LAMA relation (e.g. P19 place of birth), with a DBpedia predicate allowlist for \(R\).

| Decision | Definition |
|----------|------------|
| **supported** | Some ACE triple mentions both \(s\) and \(a\), and its predicate is in the allowlist for \(R\) |
| **weak_support** | \(s\) and \(a\) co-occur in a triple, but the predicate is **not** allowlisted |
| **unsupported** | ACE non-empty, but no subject–answer co-occurrence |
| **uncertain** | Empty ACE, empty answer, or pipeline failure |

**Gold acceptance (evaluation only):**

- **Exact match** (`gold_match`): fuzzy string match to T-REx gold label(s).  
- **Granularity match:** ACE shows multi-valued or hierarchical agreement (e.g. both `birthPlace→France` and `birthPlace→Lyon`), so a coarser KG-true answer is not scored as a false accept.  
- **gold_acceptable** = exact ∨ granularity.

This separation is intentional: *absence of ACE evidence is not proof of hallucination*; we label such cases **not_validated** when the answer disagrees with gold.

---

## 3. Experimental setup

### 3.1 Shared stack

| Role | Choice |
|------|--------|
| QA LLM (Stage 1) | Meta **Llama 3.1 8B** via Ollama (`http://127.0.0.1:11434`) |
| GSS entity extraction | Same Llama 3.1 8B (Ollama) |
| Semantic similarity | `sentence-transformers` **`all-MiniLM-L6-v2`** |
| Knowledge graph | **DBpedia** SPARQL `https://dbpedia.org/sparql` |
| GSS entry point | `pipeline.pipeline(query)` via `run_stock_gss` |
| Mistakes | **Natural** LLM errors only (no injected wrong entities) |

### 3.2 Stock GSS (black box)

For query string \(Q_{\mathrm{gss}}\):

1. LLM entity extraction with importance scores (1–5).  
2. URI grounding to DBpedia.  
3. SPARQL batch retrieval of triples for seed URIs.  
4. One-hop expansion (priority-limited).  
5. Pruning of unproductive URIs.  
6. Triple scoring \(w(t) = 0.6\, w_{\mathrm{imp}}(t) + 0.4\, w_{\mathrm{sim}}(t)\).  
7. Top-k selection and deduplication → ACE \(S\).

Wrappers never alter these steps; they only set \(Q_{\mathrm{gss}}\).

### 3.3 End-to-end procedure

1. **Answer:** `answer_facteval.py` → `llm_answers.json`.  
2. **Validate:** `validate_with_gss.py --modes answer subject claim` → `validation_results_{mode}.json`.  
3. Post-hoc checks in `validation.py`.

### 3.4 Two corpora

| | **Pilot (n=10)** | **Stability (n=30)** |
|--|------------------|----------------------|
| Sample file | `sample_10.json` (archived) | `sample_30.json` |
| Sampling | One item per preferred relation (seed 42 family) | Round-robin: **3 × 10** relations, seed **2026** |
| Relations | P19, P20, P27, P36, P106, P17, P159, P495, P740, P937 | Same ten relations |
| LLM ≈ gold | **4/10** exact | **~17/30** fuzzy/containment match |
| Stage-2 wall time | ~**90–95 min** (30 GSS runs) | ~**106 min** (90 GSS runs) |
| Artifacts | `out/hallucination/archive_n10_pilot/` | `out/hallucination/` (current) |

The n=30 sampler (`prepare_trex_sample.py`) was extended to support stratified n>10; the n=10 set is a different draw and is **not** a subset of n=30, so rates are compared for **pattern stability**, not paired item deltas.

---

## 4. Evaluation protocol and metrics

### 4.1 What we evaluate

1. **Validator decision** on the ACE (support under relation \(R\)).  
2. **Agreement with gold** (exact / granularity).  

We never equate every “unsupported” case with a detected hallucination.

### 4.2 Decision metrics

| Metric | Meaning |
|--------|---------|
| `#supported` | Strict relation-compatible evidence |
| `#weak_support` | Entity link found; predicate not allowlisted |
| `#unsupported` | No subject–answer co-occurrence in ACE |
| `#uncertain` | No usable ACE (empty / error) |

### 4.3 Detection metrics (vs gold)

| Metric | Meaning |
|--------|---------|
| `true_support` | Exact gold + strict supported |
| `true_support_granularity` | Granularity-acceptable + strict supported |
| `weak_true_support` | Gold-acceptable + weak_support |
| `missed_support` | Gold-acceptable but unsupported (coverage miss) |
| `false_support` | Not gold-acceptable but strict supported (**dangerous**) |
| `weak_false_support` | Not gold-acceptable + weak_support |
| `not_validated` | Not gold-acceptable + unsupported (not proven false) |
| `uncertain` | No decision possible |

### 4.4 Aggregate classifier view

Treating **supported** as accept and `gold_acceptable` as positive:

- **Precision** ≈ fraction of strict accepts that are gold-acceptable.  
- **Recall** ≈ fraction of gold-acceptable answers that are strictly supported.  

Weak_support is reported separately and **not** counted as full accept in the strict view.

### 4.5 Graph diagnostics

`subject_in_graph`, `answer_in_graph`, `subject_answer_linked`, `relation_linked`, ACE size (`n_triples`).

---

## 5. Experiment I — Pilot (n=10)

### 5.1 Summary table

| Metric | answer | subject | claim |
|--------|--------|---------|-------|
| supported | 1 | 3 | 3 |
| weak_support | 1 | 1 | 0 |
| unsupported | 8 | 5 | 6 |
| uncertain | 0 | 1 | 1 |
| true_support | 1 | 2 | 2 |
| true_support_granularity | 0 | 1 | 1 |
| weak_true_support | 0 | 1 | 0 |
| missed_support | 3 | 1 | 2 |
| **false_support** | **0** | **0** | **0** |
| weak_false_support | 1 | 0 | 0 |
| not_validated | 5 | 4 | 4 |
| gold_match (exact) | 4 | 4 | 4 |
| gold_acceptable | 4 | 5 | 5 |

### 5.2 Qualitative highlights

- **Clear success:** Albania → Tirana (`capital`) is `true_support` in all three modes.  
- **Subject/claim > answer:** Roger Dubuis → Geneva missed in answer mode (hub drops the company) but supported in subject/claim.  
- **Granularity:** Catherine Tasca → France (gold Lyon) → `true_support_granularity` when ACE contains both birthPlace facts.  
- **Weak tier:** Yuquot → Canada via non-allowlisted `location` → `weak_true_support` (subject).  
- **Persistent miss:** Pink Fairies → London often `missed_support` (band-centric ACE).  
- Low empty-ACE rate on this small, relatively well-covered set.

### 5.3 Pilot takeaway

Query formulation is as important as summarization quality. Subject/claim look promising; answer mode is high-precision / very low-recall. Zero strict false_support is encouraging but inconclusive at n=10.

---

## 6. Experiment II — Stability study (n=30)

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

### 6.2 Qualitative highlights

- **Subject strict wins:** Vincent Strambi–Rome; Revolverheld–Hamburg.  
- **Subject weak wins:** Akhmetov–Kazakhstan; Always Greener–Australia (predicate mismatch).  
- **Claim wins:** Mackenzie–Edinburgh; Revolverheld–Hamburg; Crossley Motors HQ tagged `true_support_granularity` despite verbose/Rochdale-flavored text (gold Manchester)—**audit case** for granularity rules.  
- **Answer-mode collapse:** 21/30 empty ACEs on hubs/ambiguous strings → uncertain, not false accept.  
- **Misses:** correct answers with non-empty ACEs lacking subject–answer links (e.g. Ginza–Japan; Lindenberg–Hamburg).  
- **Honest abstention:** wrong answers without evidence → `not_validated`.

### 6.3 Stability takeaway

The pilot’s qualitative ranking **replicates**: subject/claim ≫ answer; false_support stays **0**. Absolute support counts do **not** scale with n; **coverage** (empty ACE + missed_support) becomes the dominant failure mode.

---

## 7. Cross-experiment synthesis

### 7.1 Side-by-side (subject mode)

| | n=10 subject | n=30 subject |
|--|--------------|--------------|
| supported | 3 | 2 |
| weak_support | 1 | 2 |
| uncertain | 1 | **13** |
| false_support | 0 | 0 |
| missed_support | 1 | 7 |
| gold_acceptable | 5 | 17 |
| Strict recall (≈ supported÷gold_acceptable) | ~0.6 | ~0.12 |

| | n=10 answer | n=30 answer |
|--|-------------|-------------|
| supported | 1 | 0 |
| uncertain | 0 | **21** |
| false_support | 0 | 0 |

### 7.2 Findings that hold in both experiments

1. **Query formulation dominates outcomes.** Answer-centered GSS is a poor validator; subject/claim are the only viable primary modes.  
2. **High precision when the validator fires.** Across 40×3 = 120 mode-item GSS validations in the two studies’ primary tables, **strict false_support = 0** in every reported mode aggregate.  
3. **Conservative semantics are necessary.** Empty ACE → uncertain; wrong answer without link → not_validated ≠ proven hallucination.  
4. **Weak_support is useful.** Citizenship / country-style facts often link under non-allowlisted predicates; demoting them to “miss” would understate KG agreement.  
5. **GSS is not a high-recall detector out of the box.** Missed linking triples and empty subgraphs dominate at n=30.

### 7.3 Findings that emerge mainly at n=30

1. **Empty-ACE rate explodes** under answer mode and becomes material under subject/claim—long-tail entities and brittle DBpedia resolution/pruning matter more than the pilot suggested.  
2. **Support counts stay in the single digits** even with 3× more items → the system behaves as a **sparse confirmation filter**.  
3. **Granularity rules need auditing** on verbose free-text answers (Crossley Motors case).  
4. Blindly increasing n further, without diagnostics and baselines, has **diminishing scientific return**.

### 7.4 Revised product framing

| Role for GSS-ACE validation | Fit |
|-----------------------------|-----|
| High-recall hallucination detector | **Poor** (coverage) |
| High-precision support tagger | **Strong** (no false supports observed) |
| Explainable abstention / “needs check” signal | **Promising** |
| Replacement for NLI / LLM-as-judge alone | **Not yet**—needs baselines |

---

## 8. Discussion: is this worth investigating further?

### 8.1 Why it remains interesting

1. **New use of KG summarization.** ACE subgraphs as evidence objects for claim validation differ from answer retrieval and from workload-based single-triple summaries.  
2. **Neuro-symbolic fit.** LLMs propose; symbolic structure constrains what counts as support.  
3. **Interpretability.** Decisions cite explicit triples—unlike opaque judges.  
4. **Informative negatives.** Answer-mode failure is a design result for KG-RAG and claim verification.  
5. **Empirical replication.** Core ranking and zero false_support survive the jump from 10 → 30.

### 8.2 Risks and limitations (now evidenced)

- Public SPARQL latency/variance; single LLM; n=30 still small for CIs.  
- Incomplete predicate allowlists.  
- Multi-valued KG facts vs single gold labels.  
- High empty-ACE rates limit practical recall.  
- No 1-hop SPARQL / NLI / LLM-judge baselines yet.  
- Granularity edge cases on long answers.

### 8.3 Research plan (updated after both experiments)

| Phase | Status | Work | Goal |
|-------|--------|------|------|
| **A. Scale to ~30** | **Done** | Stratified T-REx; subject/claim primary | Pattern stability |
| **A′. Empty-ACE diagnostics** | Next | Classify uncertain cases (URI fail vs prune vs SPARQL) | Recoverable coverage |
| **B. Evaluation hardening** | Next | Expand allowlists (P17/P27/P495); audit granularity; optional human ACE ratings | Fewer metric artifacts |
| **C. Baselines** | Priority | (i) raw 1-hop SPARQL, (ii) embedding entailment, (iii) LLM-as-judge ± ACE triples | Isolate GSS value |
| **D. KG coverage** | Later | DBpedia + Wikidata; long-tail vs popular | Completeness vs summarization |
| **E. Downstream abstention** | Later | Drop / flag unsupported answers in QA; measure effective accuracy | Practical benefit |
| **Out of scope** | — | Gold-answer oracle as GSS query at test time | Ceiling only |

**Recommended operating protocol going forward:** subject + claim as primary; answer mode ablation only (or skip to cut ~⅓ runtime).

### 8.4 Expected contributions

1. A **post-generation framework** reusing stock GSS ACEs without modifying the summarizer.  
2. An empirical study of **query formulations** across two sample sizes (n=10, n=30).  
3. An evaluation protocol with **supported / weak / unsupported / uncertain** and **granularity-aware** gold.  
4. Evidence that the method is a **precise sparse filter**, with coverage—not over-acceptance—as the main failure.  
5. A concrete roadmap (diagnostics → allowlists → baselines → abstention).

---

## 9. Conclusion

Across a 10-item pilot and a 30-item stratified stability study, **GSS-produced ACE subgraphs can serve as interpretable evidence for validating LLM answers**, provided that:

- the GSS query is centered on the **subject or full claim**, not only the answer hub;  
- validation remains **post-hoc and conservative** (`not_validated` ≠ proven hallucination; empty ACE → uncertain);  
- evaluation accounts for **weak predicates** and **multi-valued / coarser KG facts**;  
- expectations match the empirical profile: **high precision when firing, low recall / high uncertainty at scale**.

The direction is research-worthy if framed as **explainable support tagging and abstention**, not as a complete hallucination oracle. Phase A (scale to ~30) is complete. The next milestones are **empty-ACE diagnostics**, **allowlist/granularity hardening**, and **non-GSS baselines**—not another blind increase in sample size alone.

---

## 10. References (seed)

- Zervos et al., *GSS: Graph Semantic Summarization Using Answer-Centered Explanatory Subgraphs*, EDBT/ICDT Workshops 2026.  
- Survey literature on KG-supported hallucination reduction (training / inference / post-generation taxonomy), including FactEval, LTGen, KG-FPQ, OKGQA, etc.  
- Vassiliou et al., *iSummary* (workload-based KG summaries)—strong retrieval baseline, limited explanatory structure.  
- LAMA / T-REx resources used by FactEval-style factual probing.

---

## 11. Appendix: artifact map

| Path | Description |
|------|-------------|
| `dataset/facteval_trex/sample_30.json` | Current stratified 30-question set |
| `out/hallucination/llm_answers.json` | Stage-1 answers (n=30) |
| `out/hallucination/validation_results_{answer,subject,claim}.json` | n=30 validation outputs |
| `out/hallucination/run_n30_validation.log` | n=30 Stage-2 log |
| `out/hallucination/archive_n10_pilot/` | Archived n=10 sample, answers, validations |
| `src/hallucination/RESEARCH_REPORT_N30.md` | Extended n=30 write-up |
| `src/hallucination/prepare_trex_sample.py` | Stratified sampler |
| `src/hallucination/answer_facteval.py` | Stage 1 |
| `src/hallucination/validate_with_gss.py` | Experiment runner |
| `src/hallucination/validation.py` | Post-hoc decisions & metrics |
| `src/pipeline/claim_validation.py` | Query construction + stock `pipeline()` |
| `src/pipeline/pipeline.py` | Unmodified GSS |
