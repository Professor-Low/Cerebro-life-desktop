const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('cerebroDesktop', {
  // App info
  getVersion: () => ipcRenderer.invoke('get-app-version'),
  getEdition: () => ipcRenderer.invoke('get-edition'),

  // License
  getLicenseStatus: () => ipcRenderer.invoke('get-license-status'),
  activateLicense: (key) => ipcRenderer.invoke('activate-license', key),

  // Docker prerequisites
  checkDocker: () => ipcRenderer.invoke('check-docker'),
  checkClaudeCode: () => ipcRenderer.invoke('check-claude-code'),
  checkWsl: () => ipcRenderer.invoke('check-wsl'),

  // Docker install & daemon
  installDocker: () => ipcRenderer.invoke('install-docker'),
  startDockerDaemon: () => ipcRenderer.invoke('start-docker-daemon'),

  // Docker lifecycle
  setupDocker: () => ipcRenderer.invoke('setup-docker'),
  pullImages: () => ipcRenderer.invoke('pull-images'),
  startStack: () => ipcRenderer.invoke('start-stack'),
  stopStack: () => ipcRenderer.invoke('stop-stack'),
  getDockerStatus: () => ipcRenderer.invoke('get-docker-status'),
  getDockerLogs: () => ipcRenderer.invoke('get-docker-logs'),

  // Updates
  checkForUpdates: () => ipcRenderer.invoke('check-for-updates'),
  applyUpdate: () => ipcRenderer.invoke('apply-update'),

  // MCP configuration
  configureMcp: () => ipcRenderer.invoke('configure-mcp'),

  // Wizard
  wizardComplete: () => ipcRenderer.invoke('wizard-complete'),
  getSetupStatus: () => ipcRenderer.invoke('get-setup-status'),

  // Settings
  toggleAutostart: (enabled) => ipcRenderer.invoke('toggle-autostart', enabled),
  getAutostart: () => ipcRenderer.invoke('get-autostart'),
  enableAutostart: () => ipcRenderer.invoke('enable-autostart'),

  // Restart & setup state
  needsRestart: () => ipcRenderer.invoke('needs-restart'),
  saveSetupState: (state) => ipcRenderer.invoke('save-setup-state', state),
  loadSetupState: () => ipcRenderer.invoke('load-setup-state'),
  clearSetupState: () => ipcRenderer.invoke('clear-setup-state'),
  restartComputer: () => ipcRenderer.invoke('restart-computer'),

  // Event listeners for progress
  onPullProgress: (callback) => {
    ipcRenderer.on('pull-progress', (_event, data) => callback(data));
  },
  onUpdateProgress: (callback) => {
    ipcRenderer.on('update-progress', (_event, data) => callback(data));
  },
  onDockerInstallProgress: (callback) => {
    ipcRenderer.on('docker-install-progress', (_event, data) => callback(data));
  },
  onDockerStartProgress: (callback) => {
    ipcRenderer.on('docker-start-progress', (_event, data) => callback(data));
  },

  // Resume after restart signal
  onResumeAfterRestart: (callback) => {
    ipcRenderer.on('resume-after-restart', (_event, state) => callback(state));
  },

  isElectron: true,
});
