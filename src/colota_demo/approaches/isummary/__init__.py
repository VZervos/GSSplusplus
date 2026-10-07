"""iSummary as a CoLoTa answering approach.

Runs the official isummary.jar (paper Algorithm 1) on the authors' Wikidata
query log. The JAR's main picks one seed from a node-ranking file, builds the
summary from the training log, and prints it. We only choose the seed
(CoLoTa kg_entities as Wikidata URIs) and parse that SUMMARY output.
Empty log match → empty subgraph.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from utils.llm_client import TF_OR_INSUFFICIENT_SCHEMA, chat, parse_tf

URI_RE = re.compile(r"<([^>\s]+)>")
PARTI_RE = re.compile(r"^---------- PARTI\s+(.+)$")

PROMPT = """You are given a yes/no question and an iSummary workload-based subgraph.
The triples may be incomplete or include distractors. Use only the triples;
if they are not enough to decide, answer INSUFFICIENT.

Question: {question}

iSummary triples:
{triples}

Return JSON: {{"answer": "TRUE"}}, {{"answer": "FALSE"}}, or {{"answer": "INSUFFICIENT"}}.
"""


def _local_name(uri: str) -> str:
    text = (uri or "").strip().strip("<>")
    if not text or text.startswith("?"):
        return text
    if "#" in text:
        return text.rsplit("#", 1)[-1].replace("_", " ")
    return text.rstrip("/").rsplit("/", 1)[-1].replace("_", " ")


def format_triple_text(s: str, p: str, o: str) -> str:
    return f"({_local_name(s)}, {_local_name(p)}, {_local_name(o)})"


def seed_uris(item: dict) -> list[str]:
    """CoLoTa kg_entities → Wikidata entity URIs (same form as the iSummary log)."""
    seen: set[str] = set()
    out: list[str] = []
    entities = item.get("kg_entities") or {}
    for _name, qid in entities.items():
        ident = str(qid or "").strip()
        candidates: list[str] = []
        if ident.startswith("http://") or ident.startswith("https://"):
            candidates.append(ident.replace("https://", "http://"))
        elif ident.upper().startswith("Q") and ident[1:].isdigit():
            candidates.append(f"http://www.wikidata.org/entity/{ident.upper()}")
            if ident != ident.upper():
                candidates.append(f"http://www.wikidata.org/entity/{ident}")
        for uri in candidates:
            if uri and uri not in seen:
                seen.add(uri)
                out.append(uri)
    return out


def _hop_triple(hops: list[str]) -> dict | None:
    if len(hops) < 3:
        return None
    s, p, o = hops[0], hops[1], hops[2]
    return {
        "s": s.strip("<>"),
        "p": p.strip("<>"),
        "o": o.strip("<>"),
        "s_label": _local_name(s),
        "p_label": _local_name(p),
        "o_label": _local_name(o),
        "triple": format_triple_text(s, p, o),
        "path": "->".join(hops),
    }


def parse_jar_summary(stdout: str) -> list[dict]:
    """Keep the JAR's SUMMARY path fragments (`---------- PARTI` lines)."""
    ranked: list[dict] = []
    seen: set[str] = set()
    for line in stdout.splitlines():
        match = PARTI_RE.match(line.strip())
        if not match:
            continue
        hops = [part.strip() for part in match.group(1).split("->") if part.strip()]
        for start in range(0, max(0, len(hops) - 2), 2):
            hit = _hop_triple(hops[start : start + 3])
            if hit is None:
                continue
            key = hit["path"]
            if key in seen:
                continue
            seen.add(key)
            hit["score"] = 1.0 / (len(ranked) + 1)
            hit["rank"] = len(ranked) + 1
            ranked.append(hit)
    return ranked


class IsummaryApproach:
    name = "isummary"
    needs_corpus = False

    def __init__(self) -> None:
        self.kappa = 12
        self.timeout_sec = 600
        self.jar: Path | None = None
        self.train: Path | None = None
        self.test: Path | None = None
        self.log_uris: set[str] = set()
        self.cache_dir: Path | None = None
        self.work_dir: Path | None = None

    def prepare(self, corpus: list[dict], config: dict) -> None:
        from utils.dataset import ROOT

        cfg = config.get("isummary") or {}
        self.kappa = int(cfg.get("kappa", 12))
        self.timeout_sec = int(cfg.get("timeout_sec", 600))
        self.jar = Path(cfg.get("jar") or "third_party/iSummary/isummary.jar")
        self.train = Path(
            cfg.get("train") or "third_party/iSummary/data/wikidata/train.txt"
        )
        self.test = Path(
            cfg.get("test")
            or "src/colota_demo/approaches/isummary/java/dummy_test.tsv"
        )
        for path_attr in ("jar", "train", "test"):
            path = getattr(self, path_attr)
            if not path.is_absolute():
                path = ROOT / path
                setattr(self, path_attr, path)
            if not path.exists():
                raise FileNotFoundError(f"iSummary {path_attr} not found: {path}")
        print(f"iSummary: indexing query log {self.train}")
        self.log_uris = set()
        with self.train.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                self.log_uris.update(URI_RE.findall(line))
        print(f"iSummary: {len(self.log_uris)} URIs in training log")
        out_dir = Path(config.get("out_dir") or "out/colota_demo")
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        self.cache_dir = out_dir / "isummary_cache"
        self.work_dir = out_dir / "isummary_work"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _pick_seed(self, item: dict) -> str | None:
        for uri in seed_uris(item):
            if uri in self.log_uris:
                return uri
        return None

    def summarize(self, question: str, item: dict | None = None, item_id: str | None = None) -> list[dict]:
        qid = item_id or (item or {}).get("id")
        cached = self._read_cache(qid)
        if cached is not None:
            return cached
        seed = self._pick_seed(item or {})
        if not seed:
            ranked: list[dict] = []
            self._write_cache(qid, ranked)
            return ranked
        ranked = self._run_jar(seed, qid)
        self._write_cache(qid, ranked)
        return ranked

    def _run_jar(self, seed: str, item_id: str | None) -> list[dict]:
        assert self.jar and self.train and self.test and self.work_dir
        run_dir = self.work_dir / str(item_id or "item")
        run_dir.mkdir(parents=True, exist_ok=True)
        nodes = run_dir / "nodes.txt"
        nodes.write_text(
            f"{seed}\t1\nhttp://www.wikidata.org/entity/__isummary_pad__\t0\n",
            encoding="utf-8",
        )
        cmd = [
            "java",
            "-Xmx2g",
            "-jar",
            str(self.jar),
            str(self.test),
            str(self.train),
            str(nodes),
            str(self.kappa),
            "2",
        ]
        print(f"iSummary: java -jar seed={seed} kappa={self.kappa}")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            print(f"iSummary: timed out for seed {seed}")
            return []
        stdout = proc.stdout or ""
        (run_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
        if proc.stderr:
            (run_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            print(f"iSummary: jar exit {proc.returncode} for seed {seed}")
        if "BROKER FOR" in stdout:
            print(f"iSummary: original size gate skipped seed {seed}")
        return parse_jar_summary(stdout)

    def evaluate_ranking(
        self,
        gold_triples: list[str],
        ranking: list[dict],
        ks: list[int] | tuple[int, ...],
    ) -> dict:
        from approaches.rag_transformers.retriever import ranking_metrics

        docs = []
        for index, hit in enumerate(ranking, start=1):
            docs.append({
                "triple": hit.get("triple") or "",
                "rank": hit.get("rank") or index,
                "score": float(hit.get("score") or 0.0),
            })
        return ranking_metrics(gold_triples, docs, ks=tuple(ks))

    def predict(
        self,
        question: str,
        *,
        model: str,
        k: int,
        ranking: list[dict] | None = None,
        item: dict | None = None,
        item_id: str | None = None,
        **_kwargs,
    ) -> str:
        hits = (
            ranking
            if ranking is not None
            else self.summarize(question, item=item, item_id=item_id)
        )[:k]
        prompt = PROMPT.format(
            question=question,
            triples="\n".join(
                f"- {hit.get('triple')}  (score={float(hit.get('score', 0.0)):.3f})"
                for hit in hits
            )
            or "(none)",
        )
        raw = chat(prompt, model=model, schema=TF_OR_INSUFFICIENT_SCHEMA)
        return parse_tf(raw, allow_insufficient=True)

    def _cache_path(self, item_id: str | None) -> Path | None:
        if not self.cache_dir or not item_id:
            return None
        return self.cache_dir / f"{item_id}_k{self.kappa}.json"

    def _read_cache(self, item_id: str | None) -> list[dict] | None:
        path = self._cache_path(item_id)
        if path is None or not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def _write_cache(self, item_id: str | None, triples: list[dict]) -> None:
        path = self._cache_path(item_id)
        if path is None:
            return
        path.write_text(json.dumps(triples, indent=2, ensure_ascii=False), encoding="utf-8")
