"""Lightweight BLAST-style helpers used by the Flask API.

The goal here is not to faithfully re-implement BLAST, but to supply a
simple, explainable scoring routine so the UI can contrast the classical
search with the Grover-based experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

MATCH_REWARD = 2
MISMATCH_PENALTY = -1


@dataclass
class MatchWindow:
    """Container used to describe a single genome window."""

    index: int
    window: str
    identity: float
    matches: int
    score: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "window": self.window,
            "identity": self.identity,
            "matches": self.matches,
            "score": self.score,
            "is_perfect": self.matches == len(self.window),
        }


def _clean_sequence(seq: str, name: str) -> str:
    if seq is None:
        raise ValueError(f"{name} nao pode ser vazio")

    cleaned = seq.strip().upper()
    if not cleaned:
        raise ValueError(f"{name} nao pode ser vazio")
    return cleaned


def _score_window(window: str, motif: str) -> MatchWindow:
    matches = sum(1 for a, b in zip(window, motif) if a == b)
    identity = (matches / len(motif)) * 100
    mismatches = len(motif) - matches
    score = matches * MATCH_REWARD + mismatches * MISMATCH_PENALTY
    return MatchWindow(
        index=-1,  # will be set by caller
        window=window,
        identity=identity,
        matches=matches,
        score=score,
    )


def blast_search(genome: str, motif: str, threshold: float = 70.0) -> Dict[str, object]:
    genome = _clean_sequence(genome, "Genoma")
    motif = _clean_sequence(motif, "Motif")

    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("Threshold deve ser numerico.") from exc

    if len(motif) > len(genome):
        raise ValueError("Motif precisa ser menor ou igual ao genoma.")

    windows: List[Dict[str, object]] = []
    hits: List[Dict[str, object]] = []

    for idx in range(len(genome) - len(motif) + 1):
        window = genome[idx : idx + len(motif)]
        scored = _score_window(window, motif)
        scored.index = idx
        scored_dict = scored.to_dict()
        windows.append(scored_dict)
        if scored.identity >= threshold_value:
            hits.append(scored_dict)

    hits.sort(key=lambda item: (item["identity"], item["score"], -item["index"]), reverse=True)

    return {
        "threshold": threshold_value,
        "genome_length": len(genome),
        "motif_length": len(motif),
        "hit_count": len(hits),
        "hits": hits,
        "top_hit": hits[0] if hits else None,
        "all_windows": windows,
    }


def compare_methods(
    genome: str,
    motif: str,
    grover_results: Dict[str, object],
    blast_threshold: float = 70.0,
) -> Dict[str, object]:
    blast_results = blast_search(genome, motif, blast_threshold)

    grover_hits = set(grover_results.get("good_indices", []))
    blast_hits = {hit["index"] for hit in blast_results["hits"]}

    overlap = sorted(grover_hits & blast_hits)
    grover_only = sorted(grover_hits - blast_hits)
    blast_only = sorted(blast_hits - grover_hits)

    agreement_ratio = 0.0
    union = grover_hits | blast_hits
    if union:
        agreement_ratio = len(overlap) / len(union)

    analysis = {
        "overlap_indices": overlap,
        "grover_only": grover_only,
        "blast_only": blast_only,
        "agreement_ratio": agreement_ratio,
        "message": _build_analysis_message(overlap, grover_only, blast_only),
    }

    return {
        "grover": {
            "accuracy": grover_results.get("accuracy"),
            "good_indices": grover_results.get("good_indices", []),
            "total_windows": grover_results.get("total_windows"),
        },
        "blast": blast_results,
        "analysis": analysis,
    }


def _build_analysis_message(
    overlap: List[int], grover_only: List[int], blast_only: List[int]
) -> str:
    if overlap and not (grover_only or blast_only):
        return "Os dois metodos concordaram totalmente nas posicoes encontradas."
    if overlap:
        return (
            "Metodos possuem intersecao parcial; investigue indices exclusivos "
            "para entender diferencas nos criterios."
        )
    if grover_only and not blast_only:
        return "Somente Grover encontrou padroes; considere reduzir o limiar do BLAST."
    if blast_only and not grover_only:
        return "Somente o BLAST passou do limiar; verifique a precisao quantica."
    return "Nenhum dos metodos encontrou o motif com os parametros atuais."
