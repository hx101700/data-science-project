#!/bin/bash
fuser -k 8501/tcp
echo "1. Running environment: Linux, Conda, Python 3.11"
echo "2. Installing dependencies from requirements_for_linux_conda_py311.txt..."
echo "   pip install -r ../requirements/requirements_for_linux_conda_py311.txt"

streamlit run ../main.py

echo "All tasks completed successfully."
