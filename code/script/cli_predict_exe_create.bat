@echo off
REM Change directory to the script location
cd /d %~dp0

REM Define output paths
set "APP_EXE_ROOT=..\..\executable\cli_predict_exe"
set "DIST_DIR=%APP_EXE_ROOT%\dist"
set "BUILD_DIR=%APP_EXE_ROOT%\build"

REM Create target directories if they do not exist
if not exist "%APP_EXE_ROOT%" (mkdir "%APP_EXE_ROOT%")
if not exist "%DIST_DIR%" (mkdir "%DIST_DIR%")
if not exist "%BUILD_DIR%" (mkdir "%BUILD_DIR%")

REM Clean previous build/dist folders if they exist
if exist "%DIST_DIR%" (rmdir /s /q "%DIST_DIR%")
if exist "%BUILD_DIR%" (rmdir /s /q "%BUILD_DIR%")

REM Run PyInstaller using the spec file from the parent directory
pyinstaller ..\cli_predict.spec --distpath "%DIST_DIR%" --workpath "%BUILD_DIR%" --clean

echo ========================================
echo Build complete!
echo The executable is at: %DIST_DIR%\cli_predict\cli_predict.exe
echo ========================================
pause

