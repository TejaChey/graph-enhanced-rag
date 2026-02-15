# Baseline RAG prompt template
BASELINE_QA_TEMPLATE = """You are a helpful assistant answering questions about documentation.

Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""


# Baseline RAG prompt with source citations
BASELINE_QA_WITH_SOURCES_TEMPLATE = """You are a helpful assistant answering questions about documentation.

Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
When providing an answer, mention which source(s) the information came from.

Context:
{context}

Question: {question}

Answer (with sources):"""


# Graph RAG prompt template
GRAPH_QA_TEMPLATE = """You are a helpful assistant answering questions about documentation using both document context and knowledge graph information.

Document Context:
{context}

Knowledge Graph Information:
{graph_context}

Question: {question}

Synthesize information from both sources to provide a comprehensive answer. If you don't know the answer, say so.

Answer:"""


# Evaluation prompt for answer quality
EVAL_ANSWER_QUALITY_TEMPLATE = """You are evaluating the quality of an answer to a documentation question.

Question: {question}
Expected Answer: {expected_answer}
Generated Answer: {generated_answer}

Rate the generated answer on a scale of 1-5 for:
1. Accuracy (does it match the expected answer?)
2. Completeness (does it cover all important points?)
3. Clarity (is it well-explained?)

Provide your ratings and brief justification:"""


# Entity extraction prompt
ENTITY_EXTRACTION_PROMPT = """Extract key entities and their relationships from the following text.

Text: {text}

Extract:
- Technical terms and concepts
- Products and tools
- Organizations
- Important relationships between entities

Format as JSON with entities and relationships."""


# Relationship classification prompt
RELATIONSHIP_CLASSIFICATION_PROMPT = """Given two entities from documentation, classify their relationship.

Entity 1: {entity1}
Entity 2: {entity2}
Context: {context}

Classify the relationship as one of:
- uses
- part_of
- related_to
- implements
- extends
- depends_on

Relationship type:"""
