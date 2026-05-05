# Graph-Enhanced RAG

A Retrieval-Augmented Generation (RAG) system that extends standard semantic vector search with a structured **knowledge graph** to produce richer, more accurate answers from technical documentation.

## How It Works

Two RAG pipelines are available, both backed by the same ingested data:

| Pipeline | Retrieval Sources | Best For |
|---|---|---|
| **Vector RAG** | Semantic vector search (ChromaDB) | General Q&A |
| **Graph RAG** | Semantic search **+** knowledge graph (NetworkX) | Entity-relationship reasoning |

### Graph-Enhanced Pipeline (in detail)

```text
Query
  │
  ├─► Vector Store (ChromaDB)   →  top-k semantically similar passages
  │
  └─► Knowledge Graph (NetworkX)
        │  SpaCy NER on query
        └► entity lookup → typed relationships
               e.g. 'NumPy' --[uses]--> Python

Both contexts fed into a combined prompt → LLM (API or Local) → Answer
```

### Knowledge Graph Construction

During ingestion, each document chunk is processed by:
1. **SpaCy NER** (`en_core_web_sm`) — extracts entities (PERSON, ORG, PRODUCT, GPE, EVENT, TECH)
2. **Relationship Classifier** — classifies pairs of entities in the same chunk into typed relations: `uses`, `part_of`, `extends`, `depends_on`, `implements`, `related_to`
3. **NetworkX graph** — entities become nodes; relationships become weighted, typed edges. Persisted as `knowledge_graph.graphml`.

---

## Project Structure

```text
graph-enhanced-rag/
├── pipelines/
│   ├── ingest.py            # Ingestion: loads docs, chunks, embeds + builds knowledge graph
│   ├── query_baseline.py    # Vector-only RAG (interactive & single-query)
│   └── query_graph_rag.py   # Graph-Enhanced RAG (interactive & single-query)
│
├── scripts/
│   └── visualize_graph.py   # Generates an interactive HTML visualization of the graph
│
├── src/
│   ├── data_processing/     # DocumentLoader, TextCleaner, DocumentChunker
│   ├── embeddings/          # EmbeddingModel (all-MiniLM-L6-v2 via sentence-transformers)
│   ├── graph/
│   │   ├── entity_extractor.py        # SpaCy NER over document chunks
│   │   ├── relationship_classifier.py # Classifies entity pairs into typed relations
│   │   └── knowledge_graph.py         # NetworkX graph: build, persist, query
│   ├── retrieval/
│   │   ├── base_retriever.py   # ChromaDB similarity search
│   │   └── graph_retriever.py  # Entity lookup over the knowledge graph
│   ├── generation/
│   │   └── llm.py              # LLMGenerator (Supports HuggingFace API and Local CPU models)
│   └── utils/
│
├── tests/
│   └── ...                  # Pytest suite and quick tests
│
├── config/
│   └── settings.py          # All configuration (paths, models, chunk sizes, etc.)
│
├── data/                    # Place your source documents here
│   └── sample/              # Example documentation
│
├── storage/
│   ├── vectorstore/         # Persisted ChromaDB vector store
│   └── graph/               # Persisted knowledge_graph.graphml
│
└── requirements.txt
```

---

## Setup

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/vinayak-ktp/graph-enhanced-rag.git
cd graph-enhanced-rag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

**LLM Models:**
The project supports two modes for text generation:
1. **API Mode (Default)**: Served via the HuggingFace Inference API (model: `Qwen/Qwen2.5-7B-Instruct`). A free or Pro HF account with API access is required.
2. **Local Mode**: Uses a CPU-friendly local model (`google/flan-t5-base`). This does not require API credits but will download ~990MB of weights on first run. Triggered using the `--local` flag.

All other settings (chunk size, embedding model, top-k, etc.) are in [`config/settings.py`](config/settings.py) and can also be overridden via `.env`.

---

## Usage

### 1. Ingest your documents

Place your files (PDF, Markdown, TXT) in `data/` (subdirectories like `data/sample/` are supported), then run:

```bash
python -m pipelines.ingest
```

This will:
- Load and clean all documents
- Chunk them (default: 1000 tokens, 200 overlap)
- Embed chunks into the ChromaDB vector store (downloads `all-MiniLM-L6-v2` locally on first run)
- Extract entities and relationships → build and save the knowledge graph

### 2. Visualize the Graph

Generate a fully interactive HTML visualization of your constructed knowledge graph:

```bash
python scripts/visualize_graph.py
```
This will open `knowledge_graph_viz.html` in your browser.

### 3. Query

**Vector RAG** (semantic search only):

```bash
# Interactive mode
python -m pipelines.query_baseline

# Single question
python -m pipelines.query_baseline -q "What is the recommended tire pressure?"
```

**Graph-Enhanced RAG** (semantic + knowledge graph):

```bash
# Interactive mode
python -m pipelines.query_graph_rag

# Single question
python -m pipelines.query_graph_rag -q "What does NumPy use internally?"
```

**Local CPU Mode**:
Append `--local` to any query command to run entirely offline without API calls:
```bash
python -m pipelines.query_graph_rag --local
```

---

## Configuration Reference

Key settings in `config/settings.py` (all overridable via `.env`):

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |
| `HF_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | LLM served via HF Inference API |
| `CHUNK_SIZE` | `1000` | Token chunk size for document splitting |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `RETRIEVAL_TOP_K` | `4` | Number of chunks retrieved per query |
| `SPACY_MODEL` | `en_core_web_sm` | SpaCy model for NER |
| `LLM_TEMPERATURE` | `0.1` | LLM generation temperature |
| `LLM_MAX_TOKENS` | `512` | Maximum tokens in LLM response |

---

## Contributing

- Fork the repo and create a new branch for your changes
- Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages
- Open a pull request against `main`
