@echo off
setlocal

conda run -n Soft-phys-CFC-Informer python scripts\verify_local_baseline_matrix.py
if errorlevel 1 exit /b %errorlevel%

echo.
echo === Local baseline matrix verification passed ===
