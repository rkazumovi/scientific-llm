# ===========================================================================
# scientific-llm - Step 8a: FastAPI serving layer verification.
# ===========================================================================
# New packages this step: fastapi, uvicorn (see requirements-step8a.txt).
#
# This step starts a real server subprocess and talks to it over real
# HTTP, exactly like curl would - including the full model load Step
# 2/4/5/7 already do, so it needs the same several minutes those steps
# needed, plus a couple of real generations. It automatically shuts the
# server down when finished either way.
#
# If it finds a trained Step 4 adapter (outputs\checkpoints\step4_verify
# by default), the API serves that fine-tuned model. If not, it falls
# back to serving the base model - that is expected, not an error, if
# you have not run Step 4 in this folder.
#
# Prerequisite: Steps 1-7 already passed in this same folder.
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_step8a.ps1
# ===========================================================================

$ErrorActionPreference = "Stop"

function Section($title) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
}

Section "Activating venv"
if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "No venv found here. Run setup_step1.ps1 first, in this same folder." -ForegroundColor Red
    exit 1
}
. .\venv\Scripts\Activate.ps1
python --version

Section "Installing Step 8a requirements (fastapi, uvicorn)"
pip install -r requirements-step8a.txt

Section "Running Step 8a verification (real server subprocess + real HTTP requests)"
python scripts\verify_step8a.py
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Step 8a: ALL CHECKS PASSED." -ForegroundColor Green
    Write-Host "Send the full output above back and we will move to Step 8b (Docker)." -ForegroundColor Green
} else {
    Write-Host "Step 8a: SOME CHECKS FAILED (see above)." -ForegroundColor Red
    Write-Host "Paste the output back and I will tell you exactly what to fix." -ForegroundColor Red
}
exit $verifyExit
