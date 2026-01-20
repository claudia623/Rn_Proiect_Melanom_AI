@echo off
cd /d "%~dp0Rn_Proiect_Melanom_AI-main"
".\.venv\Scripts\python.exe" -m streamlit run "src/app/streamlit_ui.py"
pause
