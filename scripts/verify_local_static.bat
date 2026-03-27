@echo off
setlocal

conda run -n Soft-phys-CFC-Informer python scripts\verify_local_static.py
if errorlevel 1 exit /b %errorlevel%

echo.
echo === Local static verification passed ===
