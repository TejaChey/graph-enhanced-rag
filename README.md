# RAG Documentation Q&A System

A scalable RAG (Retrieval-Augmented Generation) system for documentation Q&A, with support for both baseline vector retrieval and knowledge graph-enhanced retrieval.

## Project Structure

```
rag-documentation-qa/
├── config/              # Configuration files
├── data/               # Documentation storage
├── storage/            # Vector store and graph storage
├── src/                # Source code modules
├── pipelines/          # Main execution pipelines
├── evaluation/         # Evaluation scripts and metrics
├── notebooks/          # Jupyter notebooks for experimentation
└── scripts/            # Utility scripts
```

## Features

- **Baseline RAG**: Traditional vector similarity-based retrieval
- **Graph RAG**: Knowledge graph-enhanced retrieval (Phase 2)
- **Modular Architecture**: Easy to swap components and compare approaches
- **Evaluation Framework**: Built-in metrics and comparison tools

## Setup

### Prerequisites

1. **Install Ollama**
   - Download from https://ollama.com
   - Pull a model: `ollama pull llama3.2`

2. **Python 3.9+**

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (for graph phase)
python -m spacy download en_core_web_sm
```

### Get Documentation

```bash
# Download LangChain documentation
bash scripts/download_docs.sh

# Or manually:
git clone https://github.com/langchain-ai/langchain.git
cp -r langchain/docs/* data/raw/
```

## Quick Start

### Phase 1: Baseline RAG

```bash
# 1. Ingest documentation
python pipelines/ingest_baseline.py

# 2. Query the system
python pipelines/query_baseline.py
```

### Phase 2: Graph RAG (Coming Soon)

```bash
# 1. Build knowledge graph
python pipelines/ingest_graph.py

# 2. Query with graph
python pipelines/query_graph.py
```

## Configuration

Edit `config/settings.py` to customize:
- Chunk size and overlap
- Embedding model
- LLM parameters
- Retrieval settings

## Evaluation

```bash
# Run evaluation on baseline
python evaluation/evaluator.py --mode baseline

# Compare baseline vs graph (Phase 2)
python evaluation/compare.py
```

## Development Roadmap

- [x] Project structure setup
- [ ] Baseline RAG implementation
- [ ] Evaluation framework
- [ ] Knowledge Graph RAG
- [ ] Hybrid retrieval
- [ ] Performance optimization

## Notes

- Start with a small subset of documentation to test quickly
- Tune chunk size based on your specific documentation
- Create evaluation questions early to track progress

## Resources

- LangChain Documentation: https://python.langchain.com/
- Graph RAG Reference: [Notebook link]
- Ollama: https://ollama.com
