$ErrorActionPreference = 'Stop'
# One-time environment: pinned Python 3.11 / torch 2.7.1 CUDA 12.8 (Blackwell-capable) + V10 extras.
$root = $PSScriptRoot
Set-Location -LiteralPath $root
$runtime = Join-Path $root '.runtime'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
Start-Transcript -Path (Join-Path $root 'setup.log') -Append
try {
    if (-not [Environment]::Is64BitProcess) { throw 'Use 64-bit PowerShell.' }
    $uv = Join-Path $runtime 'uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $archive = Join-Path $runtime 'uv.zip'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/astral-sh/uv/releases/download/0.9.5/uv-x86_64-pc-windows-msvc.zip' -OutFile $archive
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne '515dc53d7553f1357d0abc1f70acd921fbb9e30230b1d9a08737236daa6ee920') {
            throw 'Official uv archive checksum mismatch.'
        }
        Expand-Archive -LiteralPath $archive -DestinationPath $runtime -Force
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $runtime 'python'
    $env:UV_CACHE_DIR = Join-Path $runtime 'uv-cache'
    & $uv python install 3.11
    if ($LASTEXITCODE -ne 0) { throw 'Python download failed.' }
    $python = Join-Path $root '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        & $uv venv --seed --python 3.11 (Join-Path $root '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
    }
    # Blackwell GPUs (e.g. RTX PRO 4000 Blackwell, sm_120) need a CUDA 12.8 build; torch 2.5.1/cu124 cannot run on them.
    & $uv pip install --python $python 'torch==2.7.1' 'torchvision==0.22.1' --index-url 'https://download.pytorch.org/whl/cu128'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    & $uv pip install --python $python -r (Join-Path $root 'workflow\requirements.txt') -r (Join-Path $root 'requirements_v10.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', ok, torch.cuda.get_device_name(0) if ok else ''); x=torch.ones(2,device='cuda')*2 if ok else None; print('GPU kernel test', x.sum().item() if ok else 'skipped'); sys.exit(0 if ok else 1)"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA not available in torch.' }
    Write-Host 'Setup complete. Next: .\run.ps1 0 --v9 <path to V9 best.pth>'
} finally { Stop-Transcript }
