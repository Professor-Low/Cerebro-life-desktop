; Cerebro Desktop — Native Architecture Installer
; No Docker. No Defender exclusions needed.
; Clean install like any normal application.

!macro customInstall
  ; Create data directory for Cerebro
  CreateDirectory "$PROFILE\.cerebro"
  CreateDirectory "$PROFILE\.cerebro\memory"
  CreateDirectory "$PROFILE\.cerebro\logs"
!macroend

!macro customUnInstall
  ; Clean up marker files (leave user data intact)
  Delete "$INSTDIR\.setup-complete"
!macroend
