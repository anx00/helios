
Write-Host "HELIOS Weather System - Startup" -ForegroundColor Yellow
Write-Host "=====================================" -ForegroundColor Yellow

# 1. Check Python
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python not found! Please install Python 3.10+" -ForegroundColor Red
    exit
}
Write-Host "Python found." -ForegroundColor Green

# 2. Check Dependencies
if (Test-Path "requirements.txt") {
    Write-Host "Checking dependencies..." -ForegroundColor Cyan
    pip install -r requirements.txt -q
}

# 3. Start Server
Write-Host "Starting HELIOS Web Server on port 8000..." -ForegroundColor Cyan
Write-Host "   (Press Ctrl+C to stop)" -ForegroundColor DarkGray

# Run Uvicorn
python -m uvicorn web_server:app --port 8000 --reload --log-level info
