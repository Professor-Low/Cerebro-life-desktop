' Silent launcher for AI Memory Auto-Maintenance
' This runs the Python script without showing any window
' Used by Task Scheduler for background execution

Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "pythonw.exe """ & scriptDir & "\auto_maintenance.py"" --quiet", 0, False
Set WshShell = Nothing
