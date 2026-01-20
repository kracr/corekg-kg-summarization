# **OVERVIEW**

COREKG is a coreset-based framework for personalized knowledge graph summarization.
It uses sensitivity-based importance sampling to create compact, user-centric knowledge graph (KG) summaries that preserve query-relevant information with provable guarantees.

  
SPARQL Query

A structured query language used to retrieve and manipulate data stored in Resource Description Framework (RDF) format within knowledge graphs.

Example query:

SELECT ?person ?organization
WHERE {

?person [http://schema.org/worksFor](http://schema.org/worksFor) ?organization .

?organization [http://schema.org/founder](http://schema.org/founder) ?founder .
}

  
# **DEPENDENCIES AND INSTALLATION**

Required Python libraries:

numpy

tqdm

rdflib

SPARQLWrapper

Install using pip:

```
pip install numpy tqdm rdflib SPARQLWrapper
```

# **EVALUATION METRICS**

F1 Score – Accuracy of answers compared to the full knowledge graph.

Coverage – Proportion of query-relevant triples retained.

# **DATASETS AND QUERY WORKLOADS:**

Wikidata Dataset – [https://dumps.wikimedia.org/wikidatawiki/entities/](https://dumps.wikimedia.org/wikidatawiki/entities/)

Wikidata Query Dataset – [https://huggingface.co/datasets/mohnish/lc_quad/blob/main/data.zip](https://huggingface.co/datasets/mohnish/lc_quad/blob/main/data.zip)

DBpedia Dataset – [https://databus.dbpedia.org/dbpedia/collections/latest-core](https://databus.dbpedia.org/dbpedia/collections/latest-core) 

DBpedia Query Dataset (LSQ) – [http://lsq.aksw.org/](http://lsq.aksw.org/) 

LSQ-Clean Toolkit – [https://github.com/sparqeology/lsq-clean](https://github.com/sparqeology/lsq-clean) 

Freebase Dataset – [https://developers.google.com/freebase](https://developers.google.com/freebase)

Freebase Query Dataset (WebQSP) – [https://aclanthology.org/P16-2033.pdf](https://aclanthology.org/P16-2033.pdf)

# **SOFTWARE AND TOOLS USED**

Apache Jena Fuseki – Used as a SPARQL endpoint for query execution. Website: [https://jena.apache.org/documentation/fuseki2/](https://jena.apache.org/documentation/fuseki2/)

LSQ-Clean Toolkit – For preprocessing SPARQL logs and creating cleaned query workloads.
Website: [https://github.com/sparqeology/lsq-clean](https://github.com/sparqeology/lsq-clean)
