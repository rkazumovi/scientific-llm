# ===========================================================================
# scientific-llm - Step 8d: Prometheus metrics verification.
# ===========================================================================
# New package this step: prometheus_client (see requirements-step8d.txt).
#
# Same shape as setup_step8a.ps1 - a real server subprocess, real HTTP
# requests, automatic teardown - extended to also check the new
# GET /metrics endpoint (src/api/main.py) actually reflects the
# requests this run makes, not just that it responds. Needs the same
# several minutes as Step 8a for the full model load.
#
# Prerequisite: Steps 1-8c already passed in this same folder (this
# step only touches src/api/main.py - nothing Docker/Kubernetes-related
# changes here, so Steps 8b/8c do not need to be re-run for this one).
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_step8d.ps1
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

Section "Installing Step 8d requirements (prometheus_client)"
pip install -r requirements-step8d.txt

Section "Running Step 8d verification (real server subprocess + real HTTP requests)"
python scripts\verify_step8d.py
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Step 8d: ALL CHECKS PASSED." -ForegroundColor Green
    Write-Host "Send the full output above back and we will move to Step 8e (Gradio UI)." -ForegroundColor Green
} else {
    Write-Host "Step 8d: SOME CHECKS FAILED (see above)." -ForegroundColor Red
    Write-Host "Paste the output back and I will tell you exactly what to fix." -ForegroundColor Red
}
exit $verifyExit
