@echo off
rem Tricho installer for Windows.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY="
for %%P in (py python python3) do (
  if not defined PY (
    %%P -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "PY=%%P"
  )
)

if not defined PY (
  echo Error: Python 3.10 or newer is required but was not found.
  echo Install it from https://www.python.org/downloads/ and try again.
  exit /b 1
)

echo Using %PY%
%PY% install.py %*
exit /b %errorlevel%
