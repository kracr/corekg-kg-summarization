"""
Evaluation metrics for COREKG summaries.

- compute_coverage: structural coverage per paper's definition (nodes + edges)
- compute_f1_score: computes F1 (with precision=1 assumption from the paper, so F1 == recall)
- utility helpers to extract nodes/edges from triples and query patterns
"""

from typing import Set, Tuple, List, Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

Triple = Tuple[str, str, str]


def _triples_to_nodes_edges(triples: Set[Triple]) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """
    Return node set and edge set representation from triples.
    Edges are represented as (subject, predicate, object) tuples (same as triples).
    """
    nodes = set()
    edges = set()
    for s, p, o in triples:
        nodes.add(s)
        nodes.add(o)
        edges.add((s, p, o))
    return nodes, edges


def compute_coverage(
    summary_triples: Set[Triple],
    query_to_triples: Dict[int, Set[Triple]],
    wn: float = 0.5,
    wp: float = 0.5,
) -> float:
    """
    Compute coverage as described in the paper:
    Coverage(Q, S, s) = (1/n) sum_{q_i in Q} ( wn * snodes(S,q_i)/nodes(q_i) + wp * sedges(S,q_i)/edges(q_i) )

    query_to_triples: mapping query_index -> set of ground-truth triples for that query (T_q)
    summary_triples: set of triples in the summary S

    Returns coverage in [0,1].
    """
    n = len(query_to_triples)
    if n == 0:
        return 0.0
    summary_nodes, summary_edges = _triples_to_nodes_edges(summary_triples)
    total = 0.0
    for q_idx, Tq in query_to_triples.items():
        if not Tq:
            continue
        nodes_q, edges_q = _triples_to_nodes_edges(Tq)
        snodes = len(nodes_q.intersection(summary_nodes))
        sedges = len(set(edges_q).intersection(summary_edges))
        nodes_frac = snodes / len(nodes_q) if len(nodes_q) > 0 else 0.0
        edges_frac = sedges / len(edges_q) if len(edges_q) > 0 else 0.0
        total += wn * nodes_frac + wp * edges_frac
    coverage = total / n
    logger.info(f"Computed coverage: {coverage:.4f}")
    return coverage


def compute_f1_score(
    summary_triples: Set[Triple],
    query_to_full_answers: Dict[int, Set[Tuple[str, ...]]],
    query_to_summary_answers: Dict[int, Set[Tuple[str, ...]]],
) -> float:
    """
    Compute F1 across queries.
    The paper describes an evaluation where retrieved answers from the summary are always present in the original KG (TP=answers_in_both),
    FP=0, so F1 reduces to recall (if this holds). We still implement general F1:
      precision = TP / (TP + FP)
      recall = TP / (TP + FN)
      F1 = 2 * precision * recall / (precision + recall)

    query_to_full_answers: map q -> set of ground-truth answer tuples (e.g., result rows)
    query_to_summary_answers: map q -> set of summary-produced answer tuples
    """
    precisions = []
    recalls = []
    f1s = []
    n = len(query_to_full_answers)
    for q_idx in query_to_full_answers:
        full = query_to_full_answers.get(q_idx, set())
        summ = query_to_summary_answers.get(q_idx, set())
        tp = len(full.intersection(summ))
        fp = len(summ - full)
        fn = len(full - summ)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    # macro-average F1
    avg_f1 = sum(f1s) / max(1, n)
    logger.info(f"Computed F1 (macro-average): {avg_f1:.4f}")
    return avg_f1
