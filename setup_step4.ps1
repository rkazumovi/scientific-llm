# ===========================================================================
# scientific-llm - Step 4: physics-consistency loss + QLoRA training loop
# verification.
# ===========================================================================
# No new packages this step either (see requirements-step4.txt) - this
# just activates the venv and runs the Step 4 verifier, which fetches a
# few real papers from arXiv, loads the Step 2 base model + LoRA, and
# runs a few real optimizer steps combining cross-entropy with the new
# physics-consistency loss.
#
# This step trains on the GPU for real, so it will take noticeably
# longer than Steps 1-3 (roughly a couple of minutes, not seconds) - that
# is expected, not a hang.
#
# Prerequisite: Steps 1, 2, and 3 already passed in this same folder.
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_step4.ps1
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

Section "Installing Step 4 requirements (none new - see requirements-step4.txt)"
pip install -r requirements-step4.txt

Section "Running Step 4 verification (real arXiv fetch + real training steps on GPU)"
python scripts\verify_step4.py
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Step 4: ALL CHECKS PASSED." -ForegroundColor Green
    Write-Host "Send the full output above back and we will move to Step 5." -ForegroundColor Green
    Write-Host ""
    Write-Host "Optional, separate, slower check (CPU, ~14GB RAM, a few minutes):" -ForegroundColor Yellow
    Write-Host "  python src\training\merge.py --adapter-dir outputs\checkpoints\step4_verify" -ForegroundColor Yellow
} else {
    Write-Host "Step 4: SOME CHECKS FAILED (see above)." -ForegroundColor Red
    Write-Host "Paste the output back and I will tell you exactly what to fix." -ForegroundColor Red
}
exit $verifyExit
