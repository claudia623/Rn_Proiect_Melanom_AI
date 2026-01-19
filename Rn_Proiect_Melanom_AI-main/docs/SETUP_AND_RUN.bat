@echo off
REM ============================================
REM  SETUP & RUN Melanom AI Aplikatie
REM ============================================

cd /d "%~dp0"

echo.
echo ===================================
echo  MELANOM AI - Setup & Run
echo ===================================
echo.

REM Check Python
echo [1/4] Verificare Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python NU este instalat!
    echo.
    echo Trebuie sa instalezi Python 3.11+ de pe:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Bifează "Add Python to PATH" la instalare!
    echo.
    pause
    exit /b 1
)

python --version
echo [OK] Python gasit!
echo.

REM Install dependencies
echo [2/4] Instalare dependente in mediul virtual...
echo         (Aceasta poate dura 10-30 minute prima data)
echo.
if not exist .venv (
    echo [*] Creare mediu virtual...
    python -m venv .venv
)
.\.venv\Scripts\python.exe -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Instalare pachete esuata!
    pause
    exit /b 1
)
echo [OK] Pachete instalate!
echo.

REM Verify key packages
echo [3/4] Verificare pachete-cheie...
.\.venv\Scripts\python.exe -c "import streamlit; print('[OK] Streamlit gasit')" 2>nul || (echo ERROR: Streamlit nu s-a instalat! & pause & exit /b 1)
.\.venv\Scripts\python.exe -c "import tensorflow; print('[OK] TensorFlow gasit')" 2>nul || (echo ERROR: TensorFlow nu s-a instalat! & pause & exit /b 1)
echo.

REM Run application
echo [4/4] Pornire aplicatie...
echo         (Browserul se va deschide automat)
echo.
echo ===================================
echo  APLICATIA PORNESTE...
echo ===================================
echo.
echo Deschid browser la: http://localhost:8501
echo.
echo Pentru a opri aplicatia: Apasa CTRL+C in aceasta fereastra
echo.

timeout /t 2 /nobreak

.\.venv\Scripts\python.exe -m streamlit run src/app/streamlit_ui.py

pause
