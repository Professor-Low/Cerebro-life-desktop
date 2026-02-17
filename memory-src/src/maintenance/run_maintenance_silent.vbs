' Silent launcher for AI Memory Auto-Maintenance
' This runs the Python script without showing any window
' Used by Task Scheduler for background execution

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "pythonw.exe ""C:\Users\marke\NAS-cerebral-interface\src\maintenance\auto_maintenance.py"" --quiet", 0, False
Set WshShell = Nothing
