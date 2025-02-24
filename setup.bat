@echo off
REM setup.bat - for Windows

echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
echo Setup complete. Activate the environment with "venv\Scripts\activate"
pause

