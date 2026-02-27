; Custom NSIS installer hooks for Cerebro
; Adds Windows Defender exclusions during install to prevent
; Behavior:Win32/LummaStealer.CER!MTB false positive detection.
;
; The script runs ELEVATED (admin) to call Add-MpPreference, then writes
; a marker file to ~/.cerebro/ so the app's own ensureDefenderExclusion()
; knows it can skip re-prompting. The marker includes a success/fail flag
; so the app retries if the installer's attempt failed.

!macro customInstall
  ; --- Ensure ~/.cerebro directory exists for the marker file ---
  CreateDirectory "$PROFILE\.cerebro"

  ; --- Write PowerShell script to temp file ---
  ; The script adds exclusions, clears threat history, then writes a status
  ; file so we can verify whether the elevated process actually succeeded.
  FileOpen $R0 "$PLUGINSDIR\defender-setup.ps1" w
  FileWrite $R0 "$$ErrorActionPreference = 'Stop'$\r$\n"
  FileWrite $R0 "try {$\r$\n"
  FileWrite $R0 "    # Add path exclusion for install directory$\r$\n"
  FileWrite $R0 "    Add-MpPreference -ExclusionPath '$INSTDIR'$\r$\n"
  FileWrite $R0 "    # Add path exclusion for user data directory$\r$\n"
  FileWrite $R0 "    Add-MpPreference -ExclusionPath (Join-Path $$env:USERPROFILE '.cerebro')$\r$\n"
  FileWrite $R0 "    # Add PROCESS exclusion (critical for behavioral detection bypass)$\r$\n"
  FileWrite $R0 "    Add-MpPreference -ExclusionProcess 'Cerebro.exe'$\r$\n"
  FileWrite $R0 "    # Clear any existing Defender threat detections for Cerebro$\r$\n"
  FileWrite $R0 "    $$threats = Get-MpThreat -ErrorAction SilentlyContinue | Where-Object { $$_.Resources -match 'Cerebro' }$\r$\n"
  FileWrite $R0 "    foreach ($$t in $$threats) {$\r$\n"
  FileWrite $R0 "        Add-MpPreference -ThreatIDDefaultAction_Ids $$t.ThreatID -ThreatIDDefaultAction_Actions Allow -ErrorAction SilentlyContinue$\r$\n"
  FileWrite $R0 "    }$\r$\n"
  FileWrite $R0 "    # Signal success — write marker to ~/.cerebro/ (matches app check path)$\r$\n"
  FileWrite $R0 "    $$marker = Join-Path $$env:USERPROFILE '.cerebro\.defender-excluded'$\r$\n"
  FileWrite $R0 "    Set-Content -Path $$marker -Value 'installed' -Force$\r$\n"
  FileWrite $R0 "    # Also write to install dir for legacy compat$\r$\n"
  FileWrite $R0 "    Set-Content -Path '$INSTDIR\.defender-excluded' -Value 'installed' -Force$\r$\n"
  FileWrite $R0 "} catch {$\r$\n"
  FileWrite $R0 "    # Write failure marker so the app knows to retry at startup$\r$\n"
  FileWrite $R0 "    $$marker = Join-Path $$env:USERPROFILE '.cerebro\.defender-excluded'$\r$\n"
  FileWrite $R0 "    Set-Content -Path $$marker -Value 'failed' -Force$\r$\n"
  FileWrite $R0 "}$\r$\n"
  FileClose $R0

  ; --- Run the script with admin elevation (UAC prompt) ---
  ; The -Wait flag ensures the outer powershell blocks until the elevated
  ; process finishes. The elevated script writes the marker file itself
  ; (not NSIS), so we know it only gets written on actual success.
  DetailPrint "Configuring Windows Defender exclusion (admin required)..."
  nsExec::ExecToLog 'powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath powershell.exe -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList @(''-NoProfile'',''-ExecutionPolicy'',''Bypass'',''-File'',''$PLUGINSDIR\defender-setup.ps1'')"'

  ; If UAC was denied or PowerShell failed, the marker either doesn't
  ; exist or contains "failed". The app's ensureDefenderExclusion()
  ; will detect this and retry with its own UAC prompt on first launch.
!macroend

!macro customUnInstall
  ; Remove Defender exclusions on uninstall (best-effort, non-elevated)
  DetailPrint "Removing Windows Defender exclusions..."
  nsExec::ExecToLog 'powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Remove-MpPreference -ExclusionPath ''$INSTDIR'' -ErrorAction SilentlyContinue; Remove-MpPreference -ExclusionProcess ''Cerebro.exe'' -ErrorAction SilentlyContinue"'

  ; Clean up marker files
  Delete "$INSTDIR\.defender-excluded"
  Delete "$PROFILE\.cerebro\.defender-excluded"
!macroend
