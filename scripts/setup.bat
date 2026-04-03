@echo off
REM Windows setup script for graph-enhanced-rag
REM Run this once to set up the environment

echo Setting up graph-enhanced-rag...

python -m venv venv
call venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

echo.
echo Setup complete! Now run:
echo   python pipelines/ingest_baseline.py
echo   python pipelines/query_graph_rag.py
