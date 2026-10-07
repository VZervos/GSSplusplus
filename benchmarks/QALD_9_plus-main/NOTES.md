# QALD-9-plus — discussion notes

## Summary

| | |
|--|--|
| **What** | Multilingual **KGQA** benchmark: natural-language questions with gold **SPARQL** and answer bindings over DBpedia / Wikidata |
| **Size** | Train ≈408 (DBpedia) / 371 (Wikidata); Test 150 (DBpedia) / 136 (Wikidata); many languages + paraphrases |
| **KG** | **DBpedia** and **Wikidata** (parallel question sets); endpoints/dumps as used by QALD-9-plus release — treat gold SPARQL/answers as frozen |
| **Gold fields** | `id`, multilingual `question[]`, `query.sparql`, `answers` (SPARQL-result bindings) |
| **For our RQs** | Good for **factoid** Summary(X) checks & KG-grounded QA with executable gold; weaker for open-ended long-form / false-premise / long-tail commonsense |
| **Need `baselines/`?** | No — use JSON under `data/` |

Dataset: [KGQA/QALD-9-plus](https://github.com/KGQA/QALD-9-plus) (extension of QALD-9 with extra languages/paraphrases). Local folder: `QALD_9_plus-main/`.

---

## Example data entry

Random test item from `data/qald_9_plus_test_dbpedia.json` (`id = 158`), trimmed to English + SPARQL + a few answer URIs:

```json
{
  "id": "158",
  "question": [
    {
      "language": "en",
      "string": "Give me all writers that won the Nobel Prize in literature."
    },
    {
      "language": "de",
      "string": "Liste alle Schriftsteller auf, die den Nobelpreis gewonnen haben."
    }
  ],
  "query": {
    "sparql": "SELECT DISTINCT ?uri WHERE { ?uri a <http://dbpedia.org/ontology/Writer> ; <http://dbpedia.org/ontology/award> <http://dbpedia.org/resource/Nobel_Prize_in_Literature> }"
  },
  "answers": [
    {
      "head": { "vars": ["uri"] },
      "results": {
        "bindings": [
          { "uri": { "type": "uri", "value": "http://dbpedia.org/resource/Aleksandr_Solzhenitsyn" } },
          { "uri": { "type": "uri", "value": "http://dbpedia.org/resource/Harold_Pinter" } },
          { "uri": { "type": "uri", "value": "http://dbpedia.org/resource/Hermann_Hesse" } },
          { "uri": { "type": "uri", "value": "http://dbpedia.org/resource/Ernest_Hemingway" } }
        ]
      }
    }
  ]
}
```

(Full entry includes many more languages and a long answer list.)

---

## What it is

- Classic **Knowledge Graph Question Answering** set: NL question → SPARQL → entity/list/boolean answers.
- Built on **QALD-9**, extended with native-speaker translations and alternative phrasings (paraphrase robustness).
- Languages include en, de, ru, fr, es, hy, be, lt, ba, uk (coverage varies by split).
- Parallel files for **DBpedia** and **Wikidata** train/test under `data/`.

---

## Which KG? Which version?

- **DBpedia** (`*_dbpedia.json`) and **Wikidata** (`*_wikidata.json`).
- Gold **SPARQL + answer bindings** are the frozen reference; live endpoints may drift (entities renamed, triples changed). Prefer dataset answers for scoring unless you intentionally re-execute SPARQL on a chosen dump/endpoint.

---

## Baselines folder?

- **No `baselines/`** in this dataset repo (data + docs).
- Evaluation in the wild often uses **GERBIL QA**; not required to use the JSON for your own KG-summary experiments.

---

## Other notes

1. Answers can be **sets of URIs**, literals, or booleans — not only single entities.
2. Multilingual + paraphrase variants help robustness tests, but English-only is enough for most KG-summary pilots.
3. Factoid/list questions: summarization target is usually the subgraph that makes the SPARQL true, not a long essay.
4. Complements CoLoTa (commonsense long-tail), KG-FPQ (false premises), OKGQA (open-ended), llm-facteval (atomic cloze/QA from triples).

---

## Fit to research problems (brief)

### Problem 1 — post-generation hallucination detection

- Ask LLM the English question (no SPARQL) → get A (names/list).
- Build Summary(Q/S/A) from DBpedia/Wikidata around entities in Q or A.
- Check whether A’s entities are supported / match gold bindings.
- **Fit:** solid for **factoid/list** support checking; less for long-form atomic-claim decomposition.

### Problem 2 — KG summary as controlled knowledge

- Baseline vs Summary-only (verbalize triples/subgraph that answer the SPARQL) vs Q+Summary.
- Oracle summary can be derived from gold SPARQL neighborhood.
- **Fit:** strong for executable, closed-world KGQA-style grounding; weaker if the research focus is open-ended trustworthiness (prefer OKGQA) or adversarial premises (KG-FPQ).
