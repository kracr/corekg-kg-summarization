"""
Configuration and default parameters for COREKG pipeline.
Modify these values or import and override them in experiments.
"""

# SPARQL endpoint (if using Fuseki or other service). If None, use local rdflib graph matching.
#Using remote SPARQL endpoint (Fuseki) mode — used in our paper experiments.
SPARQL_ENDPOINT = None  # e.g., "http://localhost:3030/ds/query"

# Default sample size (m) used by COREKG if not specified elsewhere
DEFAULT_SAMPLE_SIZE = 1000

# Default epsilon/delta for theoretical guidance (not required to run)
DEFAULT_EPSILON = 0.1
DEFAULT_DELTA = 1e-3

# File paths: set to your local data locations before running experiments
DEFAULT_KG_PATH = "data/graph.ttl"       # turtle/ttl or rdf/xml file
DEFAULT_QUERIES_PATH = "data/queries.sparql"  # one SPARQL per line or file containing queries

# Experiment output path
OUTPUT_SUMMARY_TTL = "data/summary.ttl"

# Random seed for reproducibility
RANDOM_SEED = 42
