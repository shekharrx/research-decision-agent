@echo off
title Research Agent
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto ready
echo First-time setup, this takes a minute...
py -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo Could not find Python. Install it from python.org and tick "Add Python to PATH".
  pause
  exit /b
)
".venv\Scripts\python.exe" -m pip install -q -r requirements_free.txt
echo Setup done.
echo.

:ready
".venv\Scripts\python.exe" decide_free.py
echo.
echo Report saved in the "decisions" folder.
pause
