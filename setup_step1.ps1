# ===========================================================================
# scientific-llm - Step 1: Environment Setup (Windows 11, Python 3.13)
# ===========================================================================
# Run this from PowerShell INSIDE the project folder you want to use, e.g.
# C:\Users\<you>\Projects\scientific-llm
#
# Before running:
#   1. Open PowerShell as your normal user (not admin needed) in the project
#      folder: right-click the folder in Explorer > "Open in Terminal", or
#      cd there manually.
#   2. If you have never run a local .ps1 script before, PowerShell will
#      block it. Allow it for this session only (safe, does not change
#      system policy permanently):
#         Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   3. Then run:
#         .\setup_step1.ps1
#
# This script STOPS on the first failure (does not silently continue with a
# broken environment) and prints what to do about it.
# ===========================================================================

$ErrorActionPreference = "Stop"

function Section($title) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 0. Sanity checks: Python 3.13 present, NVIDIA driver present
# ---------------------------------------------------------------------------
Section "Checking Python 3.13"
$pyVersion = & py -3.13 --version 2>$null
if (-not $?) {
    Write-Host "Python 3.13 was not found via the py launcher." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/ (check Add python.exe to PATH)"
    Write-Host "then re-run this script."
    exit 1
}
Write-Host "Found: $pyVersion"

Section "Checking NVIDIA driver (nvidia-smi)"
$smi = & nvidia-smi 2>$null
if (-not $?) {
    Write-Host "nvidia-smi not found or failed. Your NVIDIA driver may be missing or outdated." -ForegroundColor Red
    Write-Host "Install the latest driver from https://www.nvidia.com/Download/index.aspx and re-run."
    exit 1
}
Write-Host $smi
Write-Host ""
Write-Host "Look at the CUDA Version shown top-right of the table above." -ForegroundColor Yellow
Write-Host "That is the MAXIMUM CUDA version your driver supports (not what is currently installed)." -ForegroundColor Yellow
Write-Host "This script installs PyTorch built for CUDA 12.4, which needs driver support" -ForegroundColor Yellow
Write-Host "for CUDA 12.4 or newer. If your nvidia-smi shows something older than 12.4," -ForegroundColor Yellow
Write-Host "stop here and tell me - we will pick a matching build instead." -ForegroundColor Yellow
Write-Host ""
$confirm = Read-Host "Type y to continue with the CUDA 12.4 build, or Ctrl+C to abort"
if ($confirm -ne "y") { exit 1 }

# ---------------------------------------------------------------------------
# 1. Create virtual environment
# ---------------------------------------------------------------------------
Section "Creating venv"
if (Test-Path ".\venv") {
    Write-Host "venv already exists, reusing it."
} else {
    py -3.13 -m venv venv
    Write-Host "Created .\venv"
}

Section "Activating venv"
. .\venv\Scripts\Activate.ps1
python --version

# ---------------------------------------------------------------------------
# 2. Upgrade pip
# ---------------------------------------------------------------------------
Section "Upgrading pip"
python -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# 3. Install PyTorch with CUDA 12.4 support
#    (8GB-class GPU: this is the safest widely-supported build as of now.
#    If verify_environment.py reports torch.cuda.is_available() as False,
#    your driver is likely too old for cu124 - see the message above.)
# ---------------------------------------------------------------------------
Section "Installing PyTorch (CUDA 12.4 build)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------------------
# 4. Install the rest of the Step 1 stack
# ---------------------------------------------------------------------------
Section "Installing transformers, peft, trl, bitsandbytes, accelerate, datasets"
pip install -r requirements-step1.txt

# ---------------------------------------------------------------------------
# 5. Verify everything
# ---------------------------------------------------------------------------
Section "Running verification script"
python scripts\verify_environment.py
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Step 1 environment setup: ALL CHECKS PASSED." -ForegroundColor Green
    Write-Host "Send the full output above back and we will move to Step 2." -ForegroundColor Green
} else {
    Write-Host "Step 1 environment setup: SOME CHECKS FAILED (see above)." -ForegroundColor Red
    Write-Host "Paste the output back and I will tell you exactly what to fix." -ForegroundColor Red
}
exit $verifyExit
