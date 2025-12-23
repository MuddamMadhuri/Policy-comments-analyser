@echo off
REM Set PYTHONPATH to the current directory so python can find 'policy_analysis'
set PYTHONPATH=%PYTHONPATH%;%~dp0

REM Run the API using the module syntax
python -m uvicorn policy_analysis.api.main:app --host 127.0.0.1 --port 8001 --reload

pause
