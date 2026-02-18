const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('cerebroDesktop', {
  // App info
  getVersion: () => ipcRenderer.invoke('get-app-version'),
  getEdition: () => ipcRenderer.invoke('get-edition'),

  // Service status
  getBackendStatus: () => ipcRenderer.invoke('get-backend-status'),
  getRedisStatus: () => ipcRenderer.invoke('get-redis-status'),
  getMcpStatus: () => ipcRenderer.invoke('get-mcp-status'),

  // Settings
  toggleAutostart: (enabled) => ipcRenderer.invoke('toggle-autostart', enabled),
  getAutostart: () => ipcRenderer.invoke('get-autostart'),
  restartBackend: () => ipcRenderer.invoke('restart-backend'),

  // MCP configuration
  configureMcp: () => ipcRenderer.invoke('configure-mcp'),

  // License
  getLicenseStatus: () => ipcRenderer.invoke('get-license-status'),
  activateLicense: (key) => ipcRenderer.invoke('activate-license', key),

  // Wizard
  startServices: () => ipcRenderer.invoke('start-services'),
  wizardComplete: () => ipcRenderer.invoke('wizard-complete'),
  getSetupStatus: () => ipcRenderer.invoke('get-setup-status'),

  isElectron: true,
});
