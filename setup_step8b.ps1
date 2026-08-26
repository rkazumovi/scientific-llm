# ===========================================================================
# scientific-llm - Step 8b: build and run the Dockerized API, with GPU
# passthrough, then verify it over real HTTP.
# ===========================================================================
# Prerequisites this step assumes (different from every prior step -
# this one is not just Python/venv):
#   1. Docker Desktop installed and running, with the WSL2 backend and
#      GPU support enabled (Docker Desktop Settings -> Resources ->
#      WSL Integration, and Settings -> General -> "Use the WSL 2 based
#      engine"). Your RTX 4060 needs a driver that supports WSL2 GPU
#      passthrough - if you can already run nvidia-smi inside WSL, you
#      have this.
#   2. Your existing Hugging Face cache (the one setup_step2.ps1 pointed
#      HF_HOME at - F:\huggingface-cache unless you changed it) is
#      reused via a volume mount below, so the container does not
#      re-download the ~14-15GB model. If your HF_HOME is different,
#      set it before running this script: $env:HF_HOME = "your path"
#   3. Step 8a already passed in this same folder (this step Dockerizes
#      exactly that API).
#
# This does NOT stop the container when it finishes - unlike the
# verify_step8a.py subprocess, this one keeps running afterward so you
# can keep poking at it with curl. See the end of this script for how to
# stop it.
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_step8b.ps1
# ===========================================================================

$ErrorActionPreference = "Stop"

function Section($title) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
}

$ContainerName = "scientific-llm-api"
$ImageName = "scientific-llm-api"

Section "Checking Docker is installed and running"
docker version | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker does not appear to be running. Start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}
Write-Host "Docker is running."

$dockerInfo = docker info 2>&1 | Out-String
if ($dockerInfo -notmatch "nvidia") {
    Write-Host "Note: nvidia was not mentioned in the docker info output." -ForegroundColor Yellow
    Write-Host "This is informational, not a hard failure - GPU passthrough is confirmed for real" -ForegroundColor Yellow
    Write-Host "by the health check below (gpu_available). If that check reports false, check" -ForegroundColor Yellow
    Write-Host "Docker Desktop Settings -> Resources -> WSL Integration and GPU support." -ForegroundColor Yellow
}

Section "Determining Hugging Face cache path to mount"
$HfCachePath = $env:HF_HOME
if ([string]::IsNullOrWhiteSpace($HfCachePath)) {
    $HfCachePath = "F:\huggingface-cache"
    Write-Host "HF_HOME is not set in this session - defaulting to $HfCachePath (from Step 2 setup)." -ForegroundColor Yellow
}
if (-not (Test-Path $HfCachePath)) {
    Write-Host "Cache path $HfCachePath does not exist. Set the HF_HOME environment variable to your real cache path and re-run." -ForegroundColor Red
    exit 1
}
Write-Host "Will mount $HfCachePath -> /root/.cache/huggingface (reuses your already-downloaded model)."

$CheckpointPath = Join-Path (Get-Location) "outputs\checkpoints"
$CheckpointMountArgs = @()
if (Test-Path $CheckpointPath) {
    Write-Host "Will mount $CheckpointPath -> /app/outputs/checkpoints (so a trained Step 4 adapter, if present, is served)."
    $CheckpointMountArgs = @("-v", "${CheckpointPath}:/app/outputs/checkpoints")
} else {
    Write-Host "No outputs\checkpoints folder found locally - container will serve the base model only." -ForegroundColor Yellow
}

Section "Building the Docker image ($ImageName)"
docker build -t $ImageName .
if ($LASTEXITCODE -ne 0) {
    Write-Host "docker build failed - see output above." -ForegroundColor Red
    exit 1
}

Section "Removing any previous container with this name"
docker rm -f $ContainerName 2>$null | Out-Null

Section "Starting the container (GPU passthrough enabled)"
$runArgs = @(
    "run", "-d",
    "--name", $ContainerName,
    "--gpus", "all",
    "-p", "8000:8000",
    "-v", "${HfCachePath}:/root/.cache/huggingface"
) + $CheckpointMountArgs + @($ImageName)

docker @runArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "docker run failed - see output above. If this mentions --gpus, GPU passthrough is not set up yet." -ForegroundColor Red
    exit 1
}
Write-Host "Container started as $ContainerName."

Section "Running Step 8b verification (real HTTP against the running container)"
python scripts\verify_step8b.py
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Step 8b: ALL CHECKS PASSED." -ForegroundColor Green
    Write-Host "Send the full output above back and we will move to Step 8c (Kubernetes / minikube)." -ForegroundColor Green
} else {
    Write-Host "Step 8b: SOME CHECKS FAILED (see above)." -ForegroundColor Red
    Write-Host "Check container logs with: docker logs $ContainerName" -ForegroundColor Red
    Write-Host "Paste the output back (including docker logs output if relevant) and I will tell you exactly what to fix." -ForegroundColor Red
}

Write-Host ""
Write-Host "The container is still running so you can keep testing it with curl." -ForegroundColor Cyan
Write-Host "  docker logs $ContainerName        (view server output)" -ForegroundColor Cyan
Write-Host "  docker stop $ContainerName        (stop it, frees the GPU/port)" -ForegroundColor Cyan
Write-Host "  docker rm $ContainerName          (remove it once stopped)" -ForegroundColor Cyan

exit $verifyExit
