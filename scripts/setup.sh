#!/bin/bash

# Setup script for RAG Documentation Q&A project

echo "=========================================="
echo "RAG Documentation Q&A - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Error: Could not find virtual environment activation script"
    exit 1
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Download spaCy model
echo ""
echo "Downloading spaCy model (for Phase 2)..."
python -m spacy download en_core_web_sm

# Create .env file
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ".env file created. Please edit it with your configuration."
else
    echo ".env file already exists."
fi

# Create directory structure
echo ""
echo "Setting up directory structure..."
python -c "from config.settings import setup_directories; setup_directories()"

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch storage/vectorstore/.gitkeep
touch storage/graph/.gitkeep
touch storage/cache/.gitkeep

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check Ollama is running: ollama list"
echo "2. Download documentation: bash scripts/download_docs.sh"
echo "3. Run ingestion: python pipelines/ingest_baseline.py"
echo "4. Query the system: python pipelines/query_baseline.py"
echo ""
echo "Note: Remember to activate the virtual environment:"
echo "  source venv/bin/activate  (Linux/Mac)"
echo "  venv\\Scripts\\activate    (Windows)"
