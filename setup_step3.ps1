# ===========================================================================
# scientific-llm - Step 3: arXiv data collection pipeline verification.
# ===========================================================================
# No new packages to install this step - everything used (urllib, xml,
# json, re - all Python standard library, plus datasets from Step 1) is
# already in the venv. This script just activates the venv and runs the
# Step 3 verifier, which does a small real fetch against the live arXiv
# API (5 papers) to prove the whole pipeline works end to end.
#
# Prerequisite: Step 1 and Step 2 already passed in this same folder.
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_step3.ps1
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

Section "Running Step 3 verification (fetches 5 real papers from arXiv)"
python scripts\verify_step3.py
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Step 3: ALL CHECKS PASSED." -ForegroundColor Green
    Write-Host "Send the full output above back and we will move to Step 4." -ForegroundColor Green
} else {
    Write-Host "Step 3: SOME CHECKS FAILED (see above)." -ForegroundColor Red
    Write-Host "Paste the output back and I will tell you exactly what to fix." -ForegroundColor Red
}
exit $verifyExit
