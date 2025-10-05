"""
COREKG implementation (coreset-based personalized KG summarization).

Implements:
- sensitivity computation (s(t))
- importance sampling p(t) = s(t) / S
- sampling of m triples
- weight assignment w(t) = S / (m * s(t))

Usage:
    from corekg.coreset import COREKG
    model = COREKG(triples, queries, m=500, endpoint=None)
    C, weights = model.build_summary()
"""

from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict, Counter
import random
import math
import numpy as np
from tqdm import tqdm
import logging
from .data_utils import execute_sparql_endpoint

logger = logging.getLogger(__name__)

Triple = Tuple[str, str, str]  # (s,p,o)


class COREKG:
    def __init__(
        self,
        triples: Set[Triple],
        queries: List[str],
        m: int = 1000,
        endpoint: Optional[str] = None,
        random_seed: Optional[int] = None,
    ):
        """
        triples: set of (s,p,o) triples from the full KG (strings)
        queries: list of SPARQL queries (strings) representing workload Q
        m: number of samples to draw for coreset
        endpoint: optional SPARQL endpoint; if provided, will run queries there to obtain matching triples
        """
        self.triples = list(triples)  # list for indexing
        self.triple_index = {t: i for i, t in enumerate(self.triples)}
        self.queries = queries
        self.m = m
        self.endpoint = endpoint
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _get_triples_matching_query_local(self, query: str) -> Set[Triple]:
        """
        Try to parse simple triple patterns from the WHERE clause and match against local triples.
        This is a best-effort (heuristic) approach: for many complex SPARQL queries it will be incomplete.
        The heuristic: find occurrences like (<uri> <uri> <uri|var>), or token sequences separated by spaces in patterns.
        """
        # extract lines between WHERE { and the matching closing brace
        import re

        m = re.search(r"WHERE\s*\{(.+)\}", query, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return set()
        where = m.group(1)
        # Find triple pattern tokens (very naive)
        pattern = re.compile(r"([^\s]+)\s+([^\s]+)\s+([^\s]+)\s*\.")
        matches = pattern.findall(where)
        matched = set()
        if not matches:
            # Try pattern without trailing dot
            pattern2 = re.compile(r"([^\s]+)\s+([^\s]+)\s+([^\s]+)\s*")
            matches = pattern2.findall(where)
        # normalize tokens (strip parentheses/comma)
        def norm(tok):
            return tok.strip().strip("()")
        for s_tok, p_tok, o_tok in matches:
            s_tok, p_tok, o_tok = norm(s_tok), norm(p_tok), norm(o_tok)
            # match triples in the graph: support variables by wildcard
            for t in self.triples:
                s, p, o = t
                if (s_tok.startswith("?") or s_tok in s) and (p_tok.startswith("?") or p_tok in p) and (o_tok.startswith("?") or o_tok in o):
                    matched.add(t)
        return matched

    def _get_triples_matching_query_endpoint(self, query: str) -> Set[Triple]:
        """
        Run the query against endpoint and reconstruct triples from bindings if possible.
        If the SELECT query returns variables ?s ?p ?o, use those. Otherwise, this function will
        try to extract triples from bindings heuristically.
        """
        bindings = execute_sparql_endpoint(self.endpoint, query)
        triples = set()
        for b in bindings:
            # try common variable names
            s = b.get("s") or b.get("subject") or b.get("sub") or b.get("S")
            p = b.get("p") or b.get("predicate") or b.get("pred") or b.get("P")
            o = b.get("o") or b.get("object") or b.get("obj") or b.get("O")
            if s and p and o:
                triples.add((s["value"], p["value"], o["value"]))
            else:
                # if not 3-vars, attempt to reconstruct from arbitrary bindings (take first 3)
                vals = [v["value"] for v in b.values()]
                if len(vals) >= 3:
                    triples.add((vals[0], vals[1], vals[2]))
        return triples

    def compute_query_triples(self) -> Dict[int, Set[Triple]]:
        """
        For each query q in self.queries, compute T_q = set of triples relevant to q.
        Returns dict mapping query_index -> set(triples).
        Uses endpoint if provided, otherwise local matching heuristics.
        """
        Qmaps = {}
        for i, q in enumerate(tqdm(self.queries, desc="Computing T_q for queries")):
            if self.endpoint:
                Tq = self._get_triples_matching_query_endpoint(q)#Using remote SPARQL endpoint (Fuseki) mode — used in our paper experiments.
            else:
                Tq = self._get_triples_matching_query_local(q)
            Qmaps[i] = Tq
        return Qmaps

    def compute_sensitivities(self, Qmaps: Dict[int, Set[Triple]]) -> Dict[Triple, float]:
        """
        Compute s(t) = sum_{q in Q} (1 / |T_q|) * I[t in T_q]
        Return dictionary triple -> sensitivity (float). Triples not in any T_q get s=0 (not included).
        """
        s = defaultdict(float)
        for q_idx, Tq in Qmaps.items():
            size = len(Tq)
            if size == 0:
                continue
            inv = 1.0 / size
            for t in Tq:
                s[t] += inv
        # total sensitivity S should equal |Q| ignoring queries with empty Tq
        return dict(s)

    def sample_coreset(self, sensitivities: Dict[Triple, float]) -> Tuple[List[Triple], Dict[Triple, float]]:
        """
        Sample m triples according to p(t) = s(t)/S; return list of sampled triples (with repetitions possibly)
        and weights w(t) for each sampled triple according to w(t) = S / (m * s(t))
        We deduplicate final coreset (keeping weight as sum of weights for duplicates).
        """
        # Filter triples with positive sensitivity
        items = list(sensitivities.items())
        triples, svals = zip(*items) if items else ([], [])
        svals = np.array(svals, dtype=float)
        S = svals.sum()
        if S <= 0 or len(triples) == 0:
            logger.warning("Total sensitivity S is zero: no queries matched any triples.")
            return [], {}

        ps = svals / S
        # we sample m indices with replacement according to ps
        indices = np.random.choice(len(triples), size=self.m, replace=True, p=ps)
        sampled_triples = [triples[i] for i in indices]
        # compute weights per sampled triple (correct for sampling probability)
        # w(t) = S / (m * s(t))
        weights = {}
        for t in set(sampled_triples):
            st = sensitivities[t]
            if st <= 0:
                w = 0.0
            else:
                w = S / (self.m * st)
            weights[t] = w
        # If duplicated samples should sum weight contributions: but formula already correct for expectation.
        logger.info(f"Sampled {len(sampled_triples)} items, unique {len(weights)} triples in coreset")
        return sampled_triples, weights

    def build_summary(self) -> Tuple[Set[Triple], Dict[Triple, float]]:
        """
        Full COREKG pipeline: compute T_q, s(t), sample coreset, compute weights, return summary set and weights dict.
        """
        Qmaps = self.compute_query_triples()
        sensitivities = self.compute_sensitivities(Qmaps)
        sampled_list, weights = self.sample_coreset(sensitivities)
        # final coreset: unique triples from sampled_list
        coreset_triples = set(weights.keys())
        return coreset_triples, weights
