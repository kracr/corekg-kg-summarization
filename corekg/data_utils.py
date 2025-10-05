"""
Data loading and SPARQL utilities.

Functions:
- load_triples_from_ttl: load triples into set[(s,p,o)] via rdflib
- load_queries: read SPARQL queries from file
- normalize_query: simple prefix-expansion (best-effort)
- extract_seed_nodes: get URIs from queries
- filter_queries_by_seeds: keep queries containing at least one seed
- execute_sparql_endpoint: run query against endpoint and return triples/results
"""

import re
from typing import List, Set, Tuple, Dict, Optional
import rdflib
from rdflib import Graph, URIRef, BNode, Literal
from SPARQLWrapper import SPARQLWrapper, JSON
import logging

logger = logging.getLogger(__name__)


Triple = Tuple[str, str, str]


def load_triples_from_ttl(path: str, format: str = "ttl") -> Set[Triple]:
    """
    Load an RDF file via rdflib and return a set of triples (s,p,o) as strings.
    Accepts formats recognized by rdflib (e.g., "ttl", "xml", "nt").
    """
    g = Graph()
    logger.info(f"Parsing RDF graph from {path} (format={format}) ...")
    g.parse(path, format=format)
    triples = set()
    for s, p, o in g:
        triples.add((str(s), str(p), str(o)))
    logger.info(f"Loaded {len(triples)} triples from {path}")
    return triples


def load_queries(path: str) -> List[str]:
    """
    Load SPARQL queries from a file. The file may contain one or more queries.
    This function splits queries by closing '}' at top-level heuristically.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Naive splitting for files with multiple queries; keep braces balanced
    queries = []
    buf = []
    depth = 0
    for line in content.splitlines():
        buf.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0 and buf:
            q = "\n".join(buf).strip()
            if q:
                queries.append(q)
            buf = []
    # If leftover
    if buf:
        q = "\n".join(buf).strip()
        if q:
            queries.append(q)

    logger.info(f"Loaded {len(queries)} queries from {path}")
    return queries


_prefix_pattern = re.compile(r"PREFIX\s+([^\s:]+):\s*<([^>]+)>", flags=re.IGNORECASE)


def normalize_query(q: str, prefix_map: Optional[Dict[str, str]] = None) -> str:
    """
    Very small helper to expand prefixes if prefix_map provided or infer from query.
    Returns the original query if no substitution is done.
    """
    if prefix_map is None:
        prefix_map = {}
        for m in _prefix_pattern.finditer(q):
            prefix_map[m.group(1)] = m.group(2)

    def expand_match(match):
        pref, local = match.group(1).split(":")
        if pref in prefix_map:
            return f"<{prefix_map[pref]}{local}>"
        return match.group(0)

    # Expand occurrences like foaf:Person (very naive, only token-level)
    expanded = re.sub(r"([A-Za-z0-9_\-]+:[A-Za-z0-9_\-]+)", lambda m: expand_match(m), q)
    return expanded


_uri_pattern = re.compile(r"<([^>]+)>|((?:http|https)://[^\s\)\.]+)|([A-Za-z0-9_\-]+:[A-Za-z0-9_\-]+)")


def extract_seed_nodes(queries: List[str]) -> Set[str]:
    """
    Extract candidate seed URIs from queries. Returns set of strings (URIs or prefixed tokens).
    This is a heuristic: finds tokens that look like URIs or prefixed names.
    """
    seeds = set()
    for q in queries:
        for m in _uri_pattern.finditer(q):
            token = m.group(1) or m.group(2) or m.group(3)
            if token:
                # Filter out common SPARQL keywords
                if token.upper() in {"SELECT", "WHERE", "FILTER", "OPTIONAL", "UNION"}:
                    continue
                seeds.add(token)
    logger.info(f"Extracted {len(seeds)} seed candidates from queries")
    return seeds


def filter_queries_by_seeds(queries: List[str], seeds: Set[str]) -> List[str]:
    """
    Keep only queries that mention at least one seed token.
    """
    filtered = []
    for q in queries:
        if any(seed in q for seed in seeds):
            filtered.append(q)
    logger.info(f"Filtered queries: {len(filtered)} kept out of {len(queries)}")
    return filtered


def execute_sparql_endpoint(endpoint: str, query: str, timeout: int = 60) -> List[Dict]:
    """
    Execute a SPARQL SELECT query against an endpoint and return result JSON bindings.
    The caller should post-process bindings into triples or answer sets.
    """
    sparql = SPARQLWrapper(endpoint, agent="COREKG/1.0")
    sparql.setQuery(query)
    sparql.setTimeout(timeout)
    sparql.setReturnFormat(JSON)
    logger.debug(f"Executing query on endpoint {endpoint} ...")
    results = sparql.query().convert()
    bindings = results.get("results", {}).get("bindings", [])
    logger.debug(f"Endpoint returned {len(bindings)} bindings")
    return bindings
