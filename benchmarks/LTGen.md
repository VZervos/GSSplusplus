# LTGen

## What it is

**LTGen** (Long-Tail Generation) is a benchmark from Huang et al., *Prompting Large Language Models with Knowledge Graphs for Question Answering Involving Long-tail Facts* ([arXiv:2405.06524](https://arxiv.org/abs/2405.06524)).

It evaluates LLMs on open-ended generation that needs **long-tail** (rare) factual knowledge, typically grounded in Wikidata. It has two parts:

| Split | Task |
|-------|------|
| **LTGen-QA** | Single-turn question answering from KG triples (template-free) |
| **LTGen-Conv** | Multi-turn conversational QA, often needing multiple relations/triples |

Entities are stratified by Wikipedia page-view rarity (levels I–IV). The paper compares parametric-only LLMs vs prompting with KG triples, retrieved passages, or both, and reports that KG prompting often helps more than passage RAG on this setting, while combining both can reduce hallucinations.

## What we found (install status)

Attempted source: [https://github.com/hwy9855/LTGen](https://github.com/hwy9855/LTGen)

| Check | Result |
|-------|--------|
| Given GitHub URL | **404 Not Found** |
| Author `hwy9855` public repos | No LTGen (or similarly named) release |
| Hugging Face / GitHub search for LTGen | No matching long-tail QA benchmark; other “LTGen” repos are unrelated |
| Paper / arXiv | Describes LTGen and claims a release, but **no working public code or dataset URL** was located |

**Conclusion:** LTGen is **not installed** under `benchmarks/`. The original linked repository appears unavailable, and no substitute official dump was found. Revisit if the authors publish a new URL.
