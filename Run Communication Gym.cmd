@echo off
setlocal
pushd "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m communication_gym serve %*
popd
