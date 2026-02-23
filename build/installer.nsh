; Custom NSIS installer hooks for Cerebro
; Adds Windows Defender exclusion during install to prevent
; Behavior:Win32/LummaStealer.CER!MTB false positive detection.
; The installer runs elevated, so Add-MpPreference should succeed.

!macro customInstall
  ; Add Defender exclusion for the install directory and user data directory.
  ; Uses -ErrorAction SilentlyContinue so it won't fail on systems without
  ; Defender or without the required cmdlets (e.g. Windows Home editions).
  DetailPrint "Configuring Windows Defender exclusion..."
  nsExec::ExecToLog 'powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "\
    try { Add-MpPreference -ExclusionPath ''$INSTDIR'' -ErrorAction Stop; Write-Host ''Added exclusion: $INSTDIR'' } catch { Write-Host ''Skipped install dir exclusion: '' + $$_.Exception.Message }; \
    try { Add-MpPreference -ExclusionPath (Join-Path $$env:USERPROFILE ''.cerebro'') -ErrorAction Stop; Write-Host ''Added exclusion: ~/.cerebro'' } catch { Write-Host ''Skipped data dir exclusion: '' + $$_.Exception.Message }; \
    try { Add-MpPreference -ExclusionProcess ''Cerebro.exe'' -ErrorAction Stop; Write-Host ''Added process exclusion: Cerebro.exe'' } catch { Write-Host ''Skipped process exclusion: '' + $$_.Exception.Message }"'
  ; Write marker so the app knows the exclusion was handled at install time
  FileOpen $0 "$INSTDIR\.defender-excluded" w
  FileWrite $0 "installed"
  FileClose $0
!macroend

!macro customUnInstall
  ; Remove Defender exclusions on uninstall
  DetailPrint "Removing Windows Defender exclusions..."
  nsExec::ExecToLog 'powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "\
    try { Remove-MpPreference -ExclusionPath ''$INSTDIR'' -ErrorAction SilentlyContinue } catch {}; \
    try { Remove-MpPreference -ExclusionProcess ''Cerebro.exe'' -ErrorAction SilentlyContinue } catch {}"'
!macroend
