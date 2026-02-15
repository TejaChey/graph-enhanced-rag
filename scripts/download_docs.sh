#!/bin/bash

# Script to download LangChain documentation
# This script clones the LangChain repository and copies documentation to data/raw/

echo "Downloading LangChain documentation..."

# Create temporary directory
TEMP_DIR="temp_langchain"
DATA_DIR="data/raw"

# Clone LangChain repository
if [ -d "$TEMP_DIR" ]; then
    echo "Removing existing temporary directory..."
    rm -rf "$TEMP_DIR"
fi

echo "Cloning LangChain repository..."
git clone --depth 1 https://github.com/langchain-ai/langchain.git "$TEMP_DIR"

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Copy documentation files
echo "Copying documentation files..."
if [ -d "$TEMP_DIR/docs" ]; then
    cp -r "$TEMP_DIR/docs/"* "$DATA_DIR/"
    echo "Documentation copied to $DATA_DIR"
else
    echo "Warning: docs directory not found in repository"
fi

# Cleanup
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

# Count files
FILE_COUNT=$(find "$DATA_DIR" -type f | wc -l)
echo "Done! Downloaded $FILE_COUNT files to $DATA_DIR"
echo ""
echo "Next steps:"
echo "1. Run: python pipelines/ingest_baseline.py"
echo "2. Then: python pipelines/query_baseline.py"
