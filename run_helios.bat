@echo off
echo 🌞 HELIOS Weather System - Startup
echo =====================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python.
    pause
    exit /b
)

echo ✅ Python found.

if exist requirements.txt (
    echo 📦 Checking dependencies...
    pip install -r requirements.txt -q
)

echo 🚀 Starting HELIOS Web Server on port 8000...
echo    (Press Ctrl+C to stop)
echo.

python -m uvicorn web_server:app --port 8000 --reload --log-level info

pause
