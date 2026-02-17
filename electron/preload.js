const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('cerebroDesktop', {
  getVersion: () => ipcRenderer.invoke('get-app-version'),
  getEdition: () => ipcRenderer.invoke('get-edition'),
  getBackendStatus: () => ipcRenderer.invoke('get-backend-status'),
  getRedisStatus: () => ipcRenderer.invoke('get-redis-status'),
  getMcpStatus: () => ipcRenderer.invoke('get-mcp-status'),
  toggleAutostart: (enabled) => ipcRenderer.invoke('toggle-autostart', enabled),
  getAutostart: () => ipcRenderer.invoke('get-autostart'),
  restartBackend: () => ipcRenderer.invoke('restart-backend'),
  configureMcp: () => ipcRenderer.invoke('configure-mcp'),
  getLicenseStatus: () => ipcRenderer.invoke('get-license-status'),
  activateLicense: (key) => ipcRenderer.invoke('activate-license', key),
  isElectron: true,
});
