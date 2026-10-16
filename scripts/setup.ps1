$ErrorActionPreference='Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try { uv sync --extra dev; if ($LASTEXITCODE) { exit $LASTEXITCODE }; & './.venv/Scripts/python.exe' -m communication_gym doctor --no-load } finally { Pop-Location }
