@echo off
echo ===============================================
echo AI Memory Auto-Maintenance Task Setup
echo ===============================================
echo.
echo This will create a Windows Scheduled Task to run
echo AI Memory maintenance every 6 hours.
echo.

REM Create the scheduled task
schtasks /create /tn "AI Memory Maintenance" /tr "python.exe \"%~dp0auto_maintenance.py\"" /sc hourly /mo 6 /st 00:00 /ru "%USERNAME%" /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: Task created successfully!
    echo.
    echo The task will run every 6 hours starting at midnight.
    echo You can view/modify it in Task Scheduler under "AI Memory Maintenance"
) else (
    echo.
    echo ERROR: Failed to create task. Try running as Administrator.
)

echo.
pause
