@echo off
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 (
 echo Python 3.11 was not found. Install Python 3.11 first.
 pause
 exit /b 1
)
call .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m ipykernel install --user --name deepfake-noise-wavelet --display-name "DeepFake Noise Wavelet ML"
echo Setup complete.
pause
