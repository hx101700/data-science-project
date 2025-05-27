@echo off
echo 1. Running environment: Windows, Conda, Python 3.10
echo 2. Installing dependencies from requirements_for_win_conda_py310.txt...
echo    pip install -r ../requirements/requirements_for_win_conda_py310.txt
echo 3. Dependencies installed successfully.
echo 4. Launching application-cli_predict...
python ../cli_predict.py
pause