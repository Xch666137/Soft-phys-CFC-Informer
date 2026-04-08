@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"
cd /d "%ROOT_DIR%"

if not defined CONDA_ENV set "CONDA_ENV=Soft-phys-CFC-Informer"
if not defined NEXTGEN_DIR set "NEXTGEN_DIR=data_raw\nextgen"
if not defined RYE_DIR set "RYE_DIR=data_raw\rye"
if not defined ERA5_DIR set "ERA5_DIR=data_raw\era5"
if not defined OUTPUT_DIR set "OUTPUT_DIR=data_processed"
if not defined ACT_WEATHER_CSV set "ACT_WEATHER_CSV=%ERA5_DIR%\act_canberra_hourly.csv"
if not defined RYE_WEATHER_CSV set "RYE_WEATHER_CSV=%ERA5_DIR%\rye_template_hourly.csv"
if not defined RYE_GENERATION_CSV set "RYE_GENERATION_CSV=%RYE_DIR%\rye_generation_and_load.csv"
if not defined ACT_START_DATE set "ACT_START_DATE=2018-01-01"
if not defined ACT_END_DATE set "ACT_END_DATE=2018-12-31"
if not defined RYE_START_DATE set "RYE_START_DATE=2020-01-01"
if not defined RYE_END_DATE set "RYE_END_DATE=2020-12-31"
if not defined WIND_PENETRATION_TARGET set "WIND_PENETRATION_TARGET=0.15"

if not defined FETCH_NEXTGEN set "FETCH_NEXTGEN=1"
if not defined FETCH_RYE set "FETCH_RYE=1"
if not defined FETCH_ERA5_ACT set "FETCH_ERA5_ACT=1"
if not defined FETCH_ERA5_RYE set "FETCH_ERA5_RYE=1"
if not defined BUILD_DATASET set "BUILD_DATASET=1"
if not defined DRY_RUN set "DRY_RUN=0"

echo [semi-synthetic-vpp] root=%ROOT_DIR%
echo [semi-synthetic-vpp] conda_env=%CONDA_ENV%
echo [semi-synthetic-vpp] dry_run=%DRY_RUN%

where conda >nul 2>nul
if errorlevel 1 (
  echo [semi-synthetic-vpp] ERROR: conda was not found in PATH.
  echo [semi-synthetic-vpp] Open an Anaconda Prompt or add conda to PATH first.
  exit /b 1
)

call conda activate "%CONDA_ENV%"
if errorlevel 1 (
  echo [semi-synthetic-vpp] ERROR: failed to activate conda environment "%CONDA_ENV%".
  exit /b 1
)

if not exist "%NEXTGEN_DIR%" mkdir "%NEXTGEN_DIR%"
if not exist "%RYE_DIR%" mkdir "%RYE_DIR%"
if not exist "%ERA5_DIR%" mkdir "%ERA5_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

if "%FETCH_NEXTGEN%"=="0" (
  echo [semi-synthetic-vpp] skip fetch_nextgen
) else if "%DRY_RUN%"=="1" (
  echo [semi-synthetic-vpp] dry-run fetch_nextgen: python tools/fetch_nextgen.py --output-dir "%NEXTGEN_DIR%"
) else (
  echo [semi-synthetic-vpp] run fetch_nextgen
  python tools/fetch_nextgen.py --output-dir "%NEXTGEN_DIR%"
  if errorlevel 1 exit /b 1
)

if "%FETCH_RYE%"=="0" (
  echo [semi-synthetic-vpp] skip fetch_rye
) else if "%DRY_RUN%"=="1" (
  echo [semi-synthetic-vpp] dry-run fetch_rye: python tools/fetch_rye.py --output-dir "%RYE_DIR%"
) else (
  echo [semi-synthetic-vpp] run fetch_rye
  python tools/fetch_rye.py --output-dir "%RYE_DIR%"
  if errorlevel 1 exit /b 1
)

if "%FETCH_ERA5_ACT%"=="0" (
  echo [semi-synthetic-vpp] skip fetch_era5_act
) else if "%DRY_RUN%"=="1" (
  echo [semi-synthetic-vpp] dry-run fetch_era5_act: python tools/fetch_era5.py --site-key act_canberra --start-date "%ACT_START_DATE%" --end-date "%ACT_END_DATE%" --output-csv "%ACT_WEATHER_CSV%"
) else (
  echo [semi-synthetic-vpp] run fetch_era5_act
  python tools/fetch_era5.py --site-key act_canberra --start-date "%ACT_START_DATE%" --end-date "%ACT_END_DATE%" --output-csv "%ACT_WEATHER_CSV%"
  if errorlevel 1 exit /b 1
)

if "%FETCH_ERA5_RYE%"=="0" (
  echo [semi-synthetic-vpp] skip fetch_era5_rye
) else if "%DRY_RUN%"=="1" (
  echo [semi-synthetic-vpp] dry-run fetch_era5_rye: python tools/fetch_era5.py --site-key rye_template --start-date "%RYE_START_DATE%" --end-date "%RYE_END_DATE%" --output-csv "%RYE_WEATHER_CSV%"
) else (
  echo [semi-synthetic-vpp] run fetch_era5_rye
  python tools/fetch_era5.py --site-key rye_template --start-date "%RYE_START_DATE%" --end-date "%RYE_END_DATE%" --output-csv "%RYE_WEATHER_CSV%"
  if errorlevel 1 exit /b 1
)

if "%BUILD_DATASET%"=="0" (
  echo [semi-synthetic-vpp] skip build_dataset
) else if "%DRY_RUN%"=="1" (
  echo [semi-synthetic-vpp] dry-run build_dataset: python tools/build_semisynthetic_vpp.py --nextgen-dir "%NEXTGEN_DIR%" --act-weather-csv "%ACT_WEATHER_CSV%" --rye-generation-csv "%RYE_GENERATION_CSV%" --rye-weather-csv "%RYE_WEATHER_CSV%" --output-dir "%OUTPUT_DIR%" --wind-penetration-target "%WIND_PENETRATION_TARGET%"
) else (
  echo [semi-synthetic-vpp] run build_dataset
  python tools/build_semisynthetic_vpp.py --nextgen-dir "%NEXTGEN_DIR%" --act-weather-csv "%ACT_WEATHER_CSV%" --rye-generation-csv "%RYE_GENERATION_CSV%" --rye-weather-csv "%RYE_WEATHER_CSV%" --output-dir "%OUTPUT_DIR%" --wind-penetration-target "%WIND_PENETRATION_TARGET%"
  if errorlevel 1 exit /b 1
)

echo [semi-synthetic-vpp] completed
exit /b 0
