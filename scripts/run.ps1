$ErrorActionPreference='Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try { & './.venv/Scripts/python.exe' -m communication_gym serve @args } finally { Pop-Location }
