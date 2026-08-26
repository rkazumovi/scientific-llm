# ===========================================================================
# scientific-llm - Step 2: load and 4-bit quantize the base model, attach
# LoRA adapters, verify everything end to end.
# ===========================================================================
# Prerequisite: Step 1 must already be complete (venv exists, all checks
# passed). Run this from the SAME project folder as setup_step1.ps1.
#
# Also, before running this for the first time, Mistral-7B-Instruct-v0.3
# needs a Hugging Face login (it is a gated repo, approval is normally
# instant):
#   1. Create an account: https://huggingface.co/join
#   2. Open https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 and
#      click "Agree and access repository"
#   3. Create a read-access token: https://huggingface.co/settings/tokens
#   4. Run once in this venv: hf auth login
#      and paste the token when prompted
#      (older huggingface_hub versions call this huggingface-cli login -
#      this script uses the current "hf" command name)
#
# First run downloads roughly 14-15GB of model weights, cached afterward
# under your user profile. Make sure you have that much free disk space.
#
# Usage (same as Step 1):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_step2.ps1
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

Section "Installing Step 2 requirements (jupyter, ipykernel)"
pip install -r requirements-step2.txt

Section "Checking Hugging Face login"
# Native commands that print anything to stderr (even a harmless
# deprecation notice) get treated as a terminating error under
# $ErrorActionPreference = "Stop". Relax that locally for this one call
# so a warning cannot kill the script, then restore it.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$whoami = & hf auth whoami 2>&1 | Out-String
$whoamiExit = $LASTEXITCODE
$ErrorActionPreference = $prevEAP

if ($whoamiExit -ne 0 -or $whoami -match "Not logged in") {
    Write-Host "Not logged in to Hugging Face yet." -ForegroundColor Red
    Write-Host "Run: hf auth login" -ForegroundColor Yellow
    Write-Host "(paste an access token from https://huggingface.co/settings/tokens" -ForegroundColor Yellow
    Write-Host "after accepting the license at" -ForegroundColor Yellow
    Write-Host "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 )" -ForegroundColor Yellow
    Write-Host "Then re-run this script." -ForegroundColor Yellow
    exit 1
}
Write-Host "Logged in as: $whoami"

Section "Running Step 2 verification (downloads the model on first run)"
python scripts\verify_step2.py
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Step 2: ALL CHECKS PASSED." -ForegroundColor Green
    Write-Host "Send the full output above back and we will move to Step 3." -ForegroundColor Green
} else {
    Write-Host "Step 2: SOME CHECKS FAILED (see above)." -ForegroundColor Red
    Write-Host "Paste the output back and I will tell you exactly what to fix." -ForegroundColor Red
}
exit $verifyExit