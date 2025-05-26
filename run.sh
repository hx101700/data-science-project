#!/bin/bash

# force kill any process using port 8501 which is used by streamlit
fuser -k 8501/tcp

echo "run project star...."

streamlit run ./code/main.py

echo "All tasks completed successfully."
