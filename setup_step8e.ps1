# ===========================================================================
# scientific-llm - Step 8e: Gradio UI verification.
# ===========================================================================
# New packages this step: gradio, gradio_client (see requirements-step8e.txt).
#
# Same shape as setup_step8a.ps1/setup_step8d.ps1 - activate venv, install
# this step's requirements, run the verify script - but verify_step8e.py
# itself now starts TWO real servers (the Step 8a API and the new Step 8e
# Gradio UI) and drives the UI over real HTTP with gradio_client. Needs
# the same several minutes as Step 8a for the full model load.
#
# Prerequisite: Steps 1-8a already passed in this same folder (this step
# only adds src/ui/gradio_app.py and talks to the existing Step 8a API -
# no Docker/Kubernetes/Prometheus changes needed to try this one).
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_step8e.ps1
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

Section "Installing Step 8e requirements (gradio, gradio_client)"
pip install -r requirements-step8e.txt

Section "Running Step 8e verification (real API server + real Gradio UI server, driven over real HTTP)"
python scripts\verify_step8e.py
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Step 8e: ALL CHECKS PASSED." -ForegroundColor Green
    Write-Host "Send the full output above back and we will move to Step 8f (GitHub Actions CI/CD)." -ForegroundColor Green
} else {
    Write-Host "Step 8e: SOME CHECKS FAILED (see above)." -ForegroundColor Red
    Write-Host "Paste the output back and I will tell you exactly what to fix." -ForegroundColor Red
}
exit $verifyExit
