@echo off
setlocal

conda run -n Soft-phys-CFC-Informer python scripts\verify_local_smoke.py
if errorlevel 1 exit /b %errorlevel%

echo.
echo === Local CPU smoke verification passed ===
