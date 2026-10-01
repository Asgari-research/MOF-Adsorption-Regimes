@echo off
call conda activate mofenv
cd /d "%~dp0"
python check_environment.py
if errorlevel 1 goto :fail
python run_all_figures.py
if errorlevel 1 goto :fail
echo.
echo SUCCESS. Main and SI figures are in outputs\main and outputs\si
pause
exit /b 0
:fail
echo.
echo Figure regeneration failed. Read the error above.
pause
exit /b 1
