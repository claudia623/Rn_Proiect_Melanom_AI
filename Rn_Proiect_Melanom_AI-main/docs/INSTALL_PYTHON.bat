@echo off
REM ============================================
REM  PYTHON INSTALLER - Automated Setup
REM ============================================

echo.
echo ===================================
echo  PYTHON 3.12 INSTALLER
echo ===================================
echo.

REM Check if Python is already installed
python --version >nul 2>&1
if errorlevel 0 (
    echo Python is already installed!
    python --version
    pause
    exit /b 0
)

echo [*] Python NOT found. Attempting to install...
echo.

REM Option 1: Try to install from Microsoft Store
echo Trying to install Python from Microsoft Store...
echo Please wait...
echo.

REM This will prompt Windows to open Microsoft Store
start ms-windows-store://pdp/?productid=9NRWMJP3717K

echo.
echo ================================================
echo  PLEASE FOLLOW THESE STEPS:
echo ================================================
echo.
echo 1. Browser/Microsoft Store should have opened
echo 2. Click "Get" or "Install" button
echo 3. Wait for installation to complete
echo 4. Close Microsoft Store
echo 5. Close PowerShell/Command Prompt
echo 6. Restart PowerShell
echo 7. Run this script again to verify
echo.
echo OR download and install from:
echo https://www.python.org/downloads/
echo.
echo IMPORTANT: Enable "Add Python to PATH"
echo.
pause
