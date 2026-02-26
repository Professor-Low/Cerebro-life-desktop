const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('cerebroDesktop', {
  // App info
  getVersion: () => ipcRenderer.invoke('get-app-version'),
  getEdition: () => ipcRenderer.invoke('get-edition'),

  // License
  getLicenseStatus: () => ipcRenderer.invoke('get-license-status'),
  activateLicense: (key) => ipcRenderer.invoke('activate-license', key),
  refreshLicense: () => ipcRenderer.invoke('refresh-license'),

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

  // File access settings
  getFileAccessConfig: () => ipcRenderer.invoke('get-file-access-config'),
  saveFileAccessConfig: (config) => ipcRenderer.invoke('save-file-access-config', config),
  getFileAccessPresets: () => ipcRenderer.invoke('get-file-access-presets'),
  browseFolder: () => ipcRenderer.invoke('browse-folder'),
  restartDockerStack: () => ipcRenderer.invoke('restart-docker-stack'),

  // Memory storage
  getMemoryConfig: () => ipcRenderer.invoke('get-memory-config'),
  browseStorageFolder: () => ipcRenderer.invoke('browse-storage-folder'),
  setStoragePath: (path) => ipcRenderer.invoke('set-storage-path', path),
  getStorageStats: (path) => ipcRenderer.invoke('get-storage-stats', path),
  migrateStorage: (destPath) => ipcRenderer.invoke('migrate-storage', destPath),

  // Restart & setup state
  needsRestart: () => ipcRenderer.invoke('needs-restart'),
  saveSetupState: (state) => ipcRenderer.invoke('save-setup-state', state),
  loadSetupState: () => ipcRenderer.invoke('load-setup-state'),
  clearSetupState: () => ipcRenderer.invoke('clear-setup-state'),
  restartComputer: () => ipcRenderer.invoke('restart-computer'),

  // Chrome CDP
  launchChromeCDP: () => ipcRenderer.invoke('launch-chrome-cdp'),
  stopChromeCDP: () => ipcRenderer.invoke('stop-chrome-cdp'),

  // Claude credentials
  checkClaudeCredentials: () => ipcRenderer.invoke('check-claude-credentials'),
  refreshClaudeCredentials: () => ipcRenderer.invoke('refresh-claude-credentials'),
  launchClaudeLogin: () => ipcRenderer.invoke('launch-claude-login'),
  silentRefreshOAuth: () => ipcRenderer.invoke('silent-refresh-oauth'),

  // Event listeners for progress
  onPullProgress: (callback) => {
    ipcRenderer.removeAllListeners('pull-progress');
    ipcRenderer.on('pull-progress', (_event, data) => callback(data));
  },
  onUpdateProgress: (callback) => {
    ipcRenderer.removeAllListeners('update-progress');
    ipcRenderer.on('update-progress', (_event, data) => callback(data));
  },
  onDockerInstallProgress: (callback) => {
    ipcRenderer.removeAllListeners('docker-install-progress');
    ipcRenderer.on('docker-install-progress', (_event, data) => callback(data));
  },
  onDockerStartProgress: (callback) => {
    ipcRenderer.removeAllListeners('docker-start-progress');
    ipcRenderer.on('docker-start-progress', (_event, data) => callback(data));
  },

  // Resume after restart signal
  onResumeAfterRestart: (callback) => {
    ipcRenderer.removeAllListeners('resume-after-restart');
    ipcRenderer.on('resume-after-restart', (_event, state) => callback(state));
  },

  // License expiry event (subscription cancelled mid-session)
  onLicenseExpired: (callback) => {
    ipcRenderer.removeAllListeners('license-expired');
    ipcRenderer.on('license-expired', (_event, data) => callback(data));
  },

  // License failure reason (sent to activation page on startup)
  onLicenseFailure: (callback) => {
    ipcRenderer.removeAllListeners('license-failure');
    ipcRenderer.on('license-failure', (_event, data) => callback(data));
  },

  // Port conflict error (sent to activation page when port 61000 is blocked)
  onPortConflict: (callback) => {
    ipcRenderer.removeAllListeners('port-conflict');
    ipcRenderer.on('port-conflict', (_event, data) => callback(data));
  },

  // Credential events
  onCredentialsExpired: (callback) => {
    ipcRenderer.removeAllListeners('credentials-expired');
    ipcRenderer.on('credentials-expired', (_event, data) => callback(data));
  },
  onCredentialsRefreshed: (callback) => {
    ipcRenderer.removeAllListeners('credentials-refreshed');
    ipcRenderer.on('credentials-refreshed', (_event, data) => callback(data));
  },

  isElectron: true,
});
