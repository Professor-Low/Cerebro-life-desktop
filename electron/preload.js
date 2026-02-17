const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('cerebroDesktop', {
  getVersion: () => ipcRenderer.invoke('get-app-version'),
  getBackendStatus: () => ipcRenderer.invoke('get-backend-status'),
  getRedisStatus: () => ipcRenderer.invoke('get-redis-status'),
  toggleAutostart: (enabled) => ipcRenderer.invoke('toggle-autostart', enabled),
  getAutostart: () => ipcRenderer.invoke('get-autostart'),
  restartBackend: () => ipcRenderer.invoke('restart-backend'),
  isElectron: true,
});
