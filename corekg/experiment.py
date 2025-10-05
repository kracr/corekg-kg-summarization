"""
End-to-end experiment runner for COREKG.

Typical usage:
    python -m corekg.experiment --kg data/graph.ttl --queries data/queries.sparql --m 1000 --endpoint http://localhost:3030/ds/query

This script:
- Loads triples and queries
- Extracts seed nodes (rudimentary)
- Filters queries to those containing the seeds (optional)
- Builds coreset via COREKG
- Evaluates coverage and F1 (best-effort)
- Writes summary to TTL if rdflib available
"""

import argparse
import logging
from typing import List, Set, Tuple, Dict
from rdflib import Graph, URIRef, Literal, BNode
from .data_utils import (
    load_triples_from_ttl,
    load_queries,
    extract_seed_nodes,
    filter_queries_by_seeds,
    execute_sparql_endpoint,
)
from .coreset import COREKG
from .metrics import compute_coverage, compute_f1_score
from . import config
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def save_summary_to_ttl(triples: Set[Tuple[str, str, str]], path: str):
    """
    Save summary triples to TTL using rdflib.
    """
    g = Graph()
    for s, p, o in triples:
        try:
            sterm = URIRef(s)
        except Exception:
            sterm = Literal(s)
        try:
            pterm = URIRef(p)
        except Exception:
            pterm = Literal(p)
        try:
            oterm = URIRef(o)
        except Exception:
            oterm = Literal(o)
        g.add((sterm, pterm, oterm))
    g.serialize(destination=path, format="turtle")
    logger.info(f"Saved summary ({len(triples)} triples) to {path}")


def build_query_to_triples_local(triples: Set[Tuple[str, str, str]], queries: List[str]) -> Dict[int, Set[Tuple[str, str, str]]]:
    """
    Build a mapping query_index -> T_q using the same heuristic as COREKG local matching.
    """
    model = COREKG(triples=triples, queries=queries, m=1)  # m unused
    return model.compute_query_triples()


def example_run(kg_path: str, queries_path: str, m: int, endpoint: str = None, sample_seed: int = 42):
    # Load data
    triples = load_triples_from_ttl(kg_path)
    queries = load_queries(queries_path)

    # extract seeds and optionally filter (paper uses seed-driven workloads)
    seeds = extract_seed_nodes(queries)
    # For the purpose of the example we keep all queries, but you can filter by seeds
    filtered_queries = filter_queries_by_seeds(queries, seeds) if seeds else queries

    logger.info(f"Using {len(filtered_queries)} queries (filtered) and KG with {len(triples)} triples")

    # Build COREKG coreset
    model = COREKG(triples=triples, queries=filtered_queries, m=m, endpoint=endpoint, random_seed=sample_seed)
    summary_triples, weights = model.build_summary()

    logger.info(f"Coreset produced with {len(summary_triples)} unique triples")

    # Evaluate: compute T_q (ground-truth triples for each query)
    query_to_triples = build_query_to_triples_local(triples, filtered_queries)

    coverage = compute_coverage(summary_triples, query_to_triples, wn=0.5, wp=0.5)

    # For F1, we need answer sets. We attempt to run SELECT queries on full graph and summary:
    # NOTE:for local rdflib we would need to create an in-memory graph and run SPARQL.
    # Here we will compute "answers" as the set of triple tuples present in T_q (i.e., treat triples as answers).
    query_to_full_answers = {i: set(Tq) for i, Tq in query_to_triples.items()}
    # For summary answers, intersect T_q with summary triples
    query_to_summary_answers = {i: (Tq.intersection(summary_triples) if Tq else set()) for i, Tq in query_to_triples.items()}

    f1 = compute_f1_score(summary_triples, query_to_full_answers, query_to_summary_answers)

    logger.info(f"Results -> Coverage: {coverage:.4f}, F1 (macro): {f1:.4f}")

    # Save summary to TTL
    save_summary_to_ttl(summary_triples, config.OUTPUT_SUMMARY_TTL)

    # Return results
    return {
        "coverage": coverage,
        "f1": f1,
        "coreset_size": len(summary_triples),
        "weights_count": len(weights),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run COREKG summarization experiment (minimal runner).")
    p.add_argument("--kg", type=str, default=config.DEFAULT_KG_PATH, help="Path to KG TTL/RDF file")
    p.add_argument("--queries", type=str, default=config.DEFAULT_QUERIES_PATH, help="Path to queries file")
    p.add_argument("--m", type=int, default=config.DEFAULT_SAMPLE_SIZE, help="Number of samples for coreset")
    p.add_argument("--endpoint", type=str, default=config.SPARQL_ENDPOINT, help="SPARQL endpoint URL (optional)")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()
    example_run(kg_path=args.kg, queries_path=args.queries, m=args.m, endpoint=args.endpoint, sample_seed=args.seed)


if __name__ == "__main__":
    main()
