@echo off
setlocal

conda run -n Soft-phys-CFC-Informer python scripts\verify_local_all.py
if errorlevel 1 exit /b %errorlevel%

echo === Local end-to-end verification passed ===
