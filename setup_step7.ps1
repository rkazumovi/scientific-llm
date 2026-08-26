# ===========================================================================
# scientific-llm - Step 7: retrieval-grounded agent (LangGraph) verification.
# ===========================================================================
# New package this step: langgraph (see requirements-step7.txt).
#
# This step fetches real papers from arXiv, builds a real embedding
# index (Step 6), loads the Step 2 base model, and runs a real question
# through the agent graph - up to two full generations (retry included),
# so expect this to take a few minutes.
#
# Prerequisite: Steps 1-6 already passed in this same folder.
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_step7.ps1
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

Section "Installing Step 7 requirements (langgraph)"
pip install -r requirements-step7.txt

Section "Running Step 7 verification (real arXiv fetch + real agent run)"
python scripts\verify_step7.py
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Step 7: ALL CHECKS PASSED." -ForegroundColor Green
    Write-Host "Send the full output above back and we will move to Step 8." -ForegroundColor Green
} else {
    Write-Host "Step 7: SOME CHECKS FAILED (see above)." -ForegroundColor Red
    Write-Host "Paste the output back and I will tell you exactly what to fix." -ForegroundColor Red
}
exit $verifyExit
