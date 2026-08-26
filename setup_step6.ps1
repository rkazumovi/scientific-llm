# ===========================================================================
# scientific-llm - Step 6: RAG retrieval + RAFT dataset construction
# verification.
# ===========================================================================
# New packages this step: faiss-cpu, sentence-transformers (see
# requirements-step6.txt for why, and why they should NOT trigger a
# torch reinstall - your existing CUDA torch build from Step 1 already
# satisfies that dependency).
#
# This step downloads a small embedding model on first run (much smaller
# than the Step 2 7B download) and fetches more real papers from arXiv
# than earlier steps (RAG needs a real corpus to retrieve against, not
# just 3-5 papers), so expect this to take a bit longer than Steps 3/5,
# though nowhere near as long as the Step 2 first run.
#
# Prerequisite: Steps 1-5 already passed in this same folder.
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_step6.ps1
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

Section "Installing Step 6 requirements (faiss-cpu, sentence-transformers)"
pip install -r requirements-step6.txt

Section "Running Step 6 verification (real arXiv fetch + real embeddings + real FAISS index)"
python scripts\verify_step6.py
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Step 6: ALL CHECKS PASSED." -ForegroundColor Green
    Write-Host "Send the full output above back and we will move to Step 7." -ForegroundColor Green
} else {
    Write-Host "Step 6: SOME CHECKS FAILED (see above)." -ForegroundColor Red
    Write-Host "Paste the output back and I will tell you exactly what to fix." -ForegroundColor Red
}
exit $verifyExit
