@echo off
setlocal
set "TRAK_SHARED_CONTROL=%~dp0..\..\_TRAK_pipeline\tools\shared_control.ps1"
if not exist "%TRAK_SHARED_CONTROL%" (
  echo Shared pipeline is unavailable. Connect the complete My Book directory structure.
  pause
  exit /b 2
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TRAK_SHARED_CONTROL%" -Molecule "TRAK1" -Action "plan" %*
set "TRAK_PIPELINE_EXIT=%ERRORLEVEL%"
echo.
echo Exit code: %TRAK_PIPELINE_EXIT%
pause
exit /b %TRAK_PIPELINE_EXIT%
