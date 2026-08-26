# ===========================================================================
# scientific-llm - Step 8c: deploy the Step 8b image to a local minikube
# cluster, then verify it over real HTTP through the Kubernetes Service.
# ===========================================================================
# Prerequisites this step assumes:
#   1. minikube installed (minikube version). If it is missing, install
#      it first - this script does not install minikube itself, the
#      same way earlier steps assumed Docker Desktop was already
#      installed rather than installing it for you.
#   2. Docker Desktop installed and running (minikube here uses the
#      docker driver, which runs the cluster as a container inside your
#      existing Docker Desktop - the same engine Step 8b already
#      confirmed works, including GPU passthrough).
#   3. Step 8b already passed in this same folder - this step deploys
#      the exact image that built there, just built again directly into
#      the Docker daemon minikube runs internally (see below).
#
# GPU handling: this script starts minikube WITHOUT GPU passthrough,
# deliberately. The official minikube documentation for NVIDIA GPU
# support with the docker driver is explicit that it requires a Linux
# host and does not work on Windows (checked against the official
# minikube NVIDIA tutorial while building this step, not assumed) - so
# attempting it here would just fail, possibly after several minutes
# and a partially started cluster, for no benefit. That is not a step
# backwards: this
# step exists to verify the Kubernetes Deployment/Service mechanics
# around the image, and Step 8b already proved GPU inference works in
# that same image on this same machine. Without a GPU, bitsandbytes
# cannot 4-bit-load the model here, so /health will correctly report a
# degraded status instead of crashing (the same design already used in
# Step 8a and 8b) - expected, not a failure. See the --allow-degraded
# flag on scripts/verify_step8b.py, used below, for how this is
# verified for real rather than just asserted.
#
# This does NOT tear anything down when it finishes - the pod, Service,
# and the kubectl port-forward keep running afterward so you can keep
# poking at it with curl. See the end of this script for how to stop it.
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_step8c.ps1
# ===========================================================================

$ErrorActionPreference = "Stop"

# PowerShell 7.3+ can otherwise treat a native command exiting non-zero
# as an immediate terminating exception, bypassing this script own
# $LASTEXITCODE checks and their explanatory messages below. Setting
# this to false restores the traditional behavior: a native command
# only stops this script via the $LASTEXITCODE checks already written
# for that purpose. Harmless no-op on older PowerShell versions that do
# not know this variable.
$PSNativeCommandUseErrorActionPreference = $false

# A SEPARATE, real PowerShell behavior (confirmed against the
# PowerShell project own issue tracker while fixing this - see
# PowerShell/PowerShell#4002): redirecting a native command stderr with
# 2>&1 makes PowerShell treat each stderr line as an ErrorRecord - and
# $ErrorActionPreference = "Stop" halts the script on ANY such record,
# even an informational one from a command that exits 0 (minikube
# writes exactly that kind of note - "this is taking a while" - to
# stderr, which is what actually tripped this while testing this step).
# Every native command call below (including minikube status, just
# after this comment) deliberately avoids 2>&1 for exactly this reason
# - per the same issue tracker discussion, stderr left unredirected
# just prints to the console normally, the same as every other command
# in this script already does, without ever triggering this.

function Section($title) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
}

function Invoke-Kubectl {
    # Deliberately no param() block - a function with none declared
    # exposes every argument passed to it, unbound, as the automatic
    # $args array, splattable with @args. That sidesteps any risk of
    # PowerShell trying to match something like -f against a declared
    # parameter name and misinterpreting it - every argument just
    # passes through raw, which is what a thin wrapper around a native
    # command needs.
    if (Get-Command kubectl -ErrorAction SilentlyContinue) {
        & kubectl @args
    } else {
        # Falls back to the kubectl bundled with minikube itself if a
        # standalone kubectl is not on PATH - one less separate install
        # to chase.
        & minikube kubectl -- @args
    }
}

Section "Checking minikube is installed"
minikube version | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "minikube does not appear to be installed. Install it, then re-run this script." -ForegroundColor Red
    exit 1
}
Write-Host "minikube is installed."

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
Write-Host "Will mount $HfCachePath into minikube at /mnt/hf-cache (reuses your already-downloaded model)."

Section "Starting minikube (this can take a few minutes on first run)"
# No 2>&1 here - see the comment near the top of this script for why.
# minikube own status text (Running / Stopped / etc.) is on stdout;
# any stderr note it prints along the way (for example a slow-command
# warning) just prints straight to the console below, harmlessly.
$mkStatus = minikube status | Out-String
if ($mkStatus -match "Running") {
    Write-Host "minikube is already running - leaving it as is. If you need a clean cluster instead, run minikube delete first and re-run this script."
} else {
    # minikube documents --mount-string as using forward slashes for a
    # Windows path (for example C:/Users/you/folder), not backslashes -
    # this avoids ambiguity with the colon in the drive letter itself.
    $mountArg = ($HfCachePath -replace "\\", "/") + ":/mnt/hf-cache"
    # 6144MB rather than a rounder 8192MB - Docker Desktop has its own
    # configured memory ceiling (Settings -> Resources -> Advanced),
    # and asking minikube for more than that ceiling fails immediately
    # with a clear MK_USAGE error (seen directly while testing this on
    # a machine where that ceiling was ~7.7GB). 6GB comfortably clears
    # a default-ish Docker Desktop ceiling while still being generous
    # for what this step actually runs - the model load is skipped
    # entirely without a GPU (see the note above), so the pod itself
    # only needs to run FastAPI/uvicorn, not the model.
    minikube start --driver=docker --cpus=4 --memory=6144 --mount --mount-string="$mountArg"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "minikube start failed - see output above." -ForegroundColor Red
        exit 1
    }
}
Write-Host "minikube is running."

Section "Building the image directly into the Docker daemon minikube runs internally"
Write-Host "The docker CLI in this PowerShell session now points at the Docker daemon minikube runs internally, not the usual Docker Desktop one." -ForegroundColor Yellow
Write-Host "Open a new PowerShell window for anything else that should use your regular Docker Desktop (for example, re-running setup_step8b.ps1)." -ForegroundColor Yellow
minikube docker-env --shell powershell | Invoke-Expression
docker build -t scientific-llm-api:latest .
if ($LASTEXITCODE -ne 0) {
    Write-Host "docker build failed - see output above." -ForegroundColor Red
    exit 1
}

Section "Applying the Deployment and Service"
Invoke-Kubectl delete -f k8s\deployment.yaml -f k8s\service.yaml --ignore-not-found | Out-Null
Invoke-Kubectl apply -f k8s\deployment.yaml -f k8s\service.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "kubectl apply failed - see output above." -ForegroundColor Red
    exit 1
}

Section "Waiting for the rollout to become ready"
Invoke-Kubectl rollout status deployment/scientific-llm-api --timeout=600s
$rolloutOk = ($LASTEXITCODE -eq 0)
if (-not $rolloutOk) {
    Write-Host "Rollout did not become ready in time. Diagnostics below:" -ForegroundColor Red
    Invoke-Kubectl get pods -l app=scientific-llm-api
    Write-Host ""
    Invoke-Kubectl describe pod -l app=scientific-llm-api
    Write-Host ""
    Invoke-Kubectl logs -l app=scientific-llm-api --tail=100
    Write-Host ""
    Write-Host "Paste the output above (especially the logs section) back and I will tell you exactly what to fix." -ForegroundColor Red
    exit 1
}
Write-Host "Pod is ready."

Section "Starting kubectl port-forward in the background (localhost:8000 -> Service:8000)"
$portForwardJob = Start-Job -ScriptBlock {
    param($UseMinikubeKubectl)
    if ($UseMinikubeKubectl) {
        minikube kubectl -- port-forward svc/scientific-llm-api 8000:8000
    } else {
        kubectl port-forward svc/scientific-llm-api 8000:8000
    }
} -ArgumentList (-not (Get-Command kubectl -ErrorAction SilentlyContinue))
Start-Sleep -Seconds 5
Write-Host "Port-forward started as background job $($portForwardJob.Id)."

Section "Running the Step 8b verification script against the Kubernetes Service"
Write-Host "Reusing scripts\verify_step8b.py rather than duplicating it - it already does exactly the HTTP checks this step needs, just pointed at a different URL." -ForegroundColor Cyan
Write-Host "Passing --allow-degraded: on this machine, minikube runs without GPU passthrough (see the note near the top of this script), so the model is expected to report a degraded state here rather than fully loading - Step 8b already proved the same image loads and generates correctly with a real GPU." -ForegroundColor Cyan
python scripts\verify_step8b.py --base-url http://localhost:8000 --allow-degraded
$verifyExit = $LASTEXITCODE

Write-Host ""
if ($verifyExit -eq 0) {
    Write-Host "Step 8c: ALL CHECKS PASSED." -ForegroundColor Green
    Write-Host "Send the full output above back and we will move to Step 8d (Prometheus metrics)." -ForegroundColor Green
} else {
    Write-Host "Step 8c: SOME CHECKS FAILED (see above)." -ForegroundColor Red
    Write-Host "Check pod logs with: kubectl logs -l app=scientific-llm-api" -ForegroundColor Red
    Write-Host "Paste the output back (including pod logs if relevant) and I will tell you exactly what to fix." -ForegroundColor Red
}

Write-Host ""
Write-Host "The pod, Service, and port-forward are still running so you can keep testing." -ForegroundColor Cyan
Write-Host "  kubectl logs -l app=scientific-llm-api        (view server output)" -ForegroundColor Cyan
Write-Host "  Stop-Job $($portForwardJob.Id); Remove-Job $($portForwardJob.Id)        (stop the port-forward)" -ForegroundColor Cyan
Write-Host "  kubectl delete -f k8s\deployment.yaml -f k8s\service.yaml        (remove the pod and Service)" -ForegroundColor Cyan
Write-Host "  minikube stop        (stop the whole cluster)" -ForegroundColor Cyan

exit $verifyExit
