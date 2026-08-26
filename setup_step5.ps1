# ===========================================================================
# scientific-llm - Step 5: evaluation suite verification (perplexity,
# ROUGE, SymPy math verification, MATH/SciQ/ARC-Challenge benchmarks).
# ===========================================================================
# No new packages this step (see requirements-step5.txt) - this just
# activates the venv and runs the Step 5 verifier, which fetches a few
# real papers from arXiv, loads the Step 2 base model, and exercises all
# four evaluation modules against real generations.
#
# This step generates text several times (perplexity texts, one summary,
# 6 benchmark examples) so it will take a few minutes, not seconds -
# expected, not a hang. The benchmark harness also tries to download
# real MATH/SciQ/ARC-Challenge examples from the Hugging Face Hub; if
# that fails for any reason it automatically falls back to small
# built-in examples and says so in the output rather than failing.
#
# Prerequisite: Steps 1-4 already passed in this same folder.
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_step5.ps1
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

Section "Installing Step 5 requirements (none new - see requirements-step5.txt)"
pip install -r requirements-step5.txt

Section "Running Step 5 verification (real arXiv fetch + real generations + real benchmarks)"
python scripts\verify_step5.py
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Step 5: ALL CHECKS PASSED." -ForegroundColor Green
    Write-Host "Send the full output above back and we will move to Step 6." -ForegroundColor Green
} else {
    Write-Host "Step 5: SOME CHECKS FAILED (see above)." -ForegroundColor Red
    Write-Host "Paste the output back and I will tell you exactly what to fix." -ForegroundColor Red
}
exit $verifyExit
