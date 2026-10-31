$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location -LiteralPath $root
$runtime = Join-Path $root '.runtime'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
Start-Transcript -Path (Join-Path $root 'setup.log') -Append
try {
    if (-not [Environment]::Is64BitProcess) { throw 'Use 64-bit PowerShell.' }
    $disk = Get-PSDrive -Name ([IO.Path]::GetPathRoot($root).Substring(0,1))
    if (-not (Test-Path (Join-Path $root 'outputs\full\best.pth')) -and $disk.Free -lt 80GB) {
        throw 'Use a drive with at least 80 GB free for the dataset, caches, environment and checkpoints.'
    }
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
    $env:KAGGLEHUB_CACHE = Join-Path $runtime 'kaggle-cache'
    $taskTemp = Join-Path $runtime 'tmp'
    New-Item -ItemType Directory -Force -Path $taskTemp | Out-Null
    $env:TEMP = $taskTemp
    $env:TMP = $taskTemp
    $env:PYTHONUNBUFFERED = '1'
    & $uv python install 3.11
    if ($LASTEXITCODE -ne 0) { throw 'Python download failed.' }
    $python = Join-Path $root '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        & $uv venv --seed --python 3.11 (Join-Path $root '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
    }
    & $uv pip install --python $python 'torch==2.5.1' 'torchvision==0.20.1' --index-url 'https://download.pytorch.org/whl/cu124'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    & $uv pip install --python $python -r (Join-Path $root 'workflow\requirements.txt') 'kagglehub==0.3.13'
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $python -u (Join-Path $root 'workflow\lab_run.py') --root $root
    if ($LASTEXITCODE -ne 0) { throw 'Pipeline stopped. Read run.log, fix the error, then run START_TRAINING.cmd again to resume.' }
} finally { Stop-Transcript }
