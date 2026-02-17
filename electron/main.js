const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { BackendManager } = require('./backend-manager');
const { RedisManager } = require('./redis-manager');
const { createTray, updateTrayStatus } = require('./tray');

const isDev = process.env.CEREBRO_DEV === '1';
// Remote mode: connect to an existing Cerebro server instead of starting local backend/redis
// Priority: env var > config file > empty (local mode)
let REMOTE_URL = process.env.CEREBRO_REMOTE_URL || '';
if (!REMOTE_URL) {
  try {
    const configPath = isDev
      ? path.join(__dirname, '..', 'cerebro-remote.json')
      : path.join(app.getPath('userData'), 'cerebro-remote.json');
    if (require('fs').existsSync(configPath)) {
      const cfg = JSON.parse(require('fs').readFileSync(configPath, 'utf-8'));
      REMOTE_URL = cfg.serverUrl || '';
    }
  } catch (_) { /* no config file, local mode */ }
}
const isRemote = !!REMOTE_URL;

let mainWindow = null;
let splashWindow = null;
let tray = null;
let backendManager = null;
let redisManager = null;
let isQuitting = false;

function getResourcePath(...parts) {
  if (isDev) {
    return path.join(__dirname, '..', ...parts);
  }
  return path.join(process.resourcesPath, ...parts);
}

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 400,
    height: 300,
    frame: false,
    transparent: true,
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  splashWindow.loadFile(path.join(__dirname, 'splash.html'));
  splashWindow.center();
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    show: false,
    icon: path.join(__dirname, '..', 'assets', 'icon.png'),
    title: 'Cerebro',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
    },
  });

  mainWindow.on('close', (e) => {
    if (!isQuitting) {
      e.preventDefault();
      mainWindow.hide();
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  return mainWindow;
}

async function startServices() {
  const serverUrl = isRemote ? REMOTE_URL : 'http://localhost:59000';

  if (isRemote) {
    // Remote mode — skip local backend/redis, connect to existing server
    console.log(`[Main] Remote mode: connecting to ${serverUrl}`);
    updateSplashStatus(`Connecting to ${serverUrl}...`);
  } else {
    // Local mode — start Redis + Backend
    updateSplashStatus('Starting Redis...');
    redisManager = new RedisManager({
      port: 16379,
      binaryPath: getResourcePath('redis'),
    });

    try {
      await redisManager.start();
      console.log('[Main] Redis started on port 16379');
    } catch (err) {
      console.error('[Main] Redis failed to start:', err.message);
    }

    updateSplashStatus('Starting Cerebro backend...');
    backendManager = new BackendManager({
      binaryPath: getResourcePath('backend'),
      port: 59000,
      isDev,
      backendSrc: isDev ? path.join(__dirname, '..', '..', 'backend') : null,
    });

    try {
      await backendManager.start();
      console.log('[Main] Backend started on port 59000');
    } catch (err) {
      console.error('[Main] Backend failed to start:', err.message);
      if (splashWindow) splashWindow.close();
      dialog.showErrorBox(
        'Cerebro - Startup Error',
        `Failed to start the backend server.\n\n${err.message}\n\nPlease check the logs and try again.`
      );
      app.quit();
      return;
    }
  }

  // Load frontend
  updateSplashStatus('Loading Cerebro...');
  mainWindow.loadURL(serverUrl);

  mainWindow.webContents.on('did-finish-load', () => {
    if (splashWindow) {
      splashWindow.close();
      splashWindow = null;
    }
    mainWindow.show();
    mainWindow.focus();
  });

  mainWindow.webContents.on('did-fail-load', (_event, errorCode, errorDescription) => {
    console.error(`[Main] Frontend failed to load: ${errorCode} ${errorDescription}`);
    setTimeout(() => {
      if (mainWindow) mainWindow.loadURL(serverUrl);
    }, 2000);
  });
}

function updateSplashStatus(message) {
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.webContents.executeJavaScript(
      `document.getElementById('status').textContent = ${JSON.stringify(message)}`
    ).catch(() => {});
  }
}

async function shutdown() {
  console.log('[Main] Shutting down...');

  if (backendManager) {
    try {
      await backendManager.stop();
    } catch (err) {
      console.error('[Main] Error stopping backend:', err.message);
    }
  }

  if (redisManager) {
    try {
      await redisManager.stop();
    } catch (err) {
      console.error('[Main] Error stopping Redis:', err.message);
    }
  }
}

// App lifecycle
app.whenReady().then(async () => {
  createSplashWindow();
  createMainWindow();

  // Create system tray
  const trayIconPath = path.join(__dirname, '..', 'assets', 'tray-icon.png');
  tray = createTray(mainWindow, trayIconPath);

  await startServices();
  updateTrayStatus(tray, 'running');
});

app.on('window-all-closed', () => {
  // Don't quit - we stay in system tray
});

app.on('activate', () => {
  if (mainWindow) {
    mainWindow.show();
    mainWindow.focus();
  }
});

app.on('before-quit', async (e) => {
  if (!isQuitting) {
    isQuitting = true;
    e.preventDefault();
    updateTrayStatus(tray, 'stopping');
    await shutdown();
    app.quit();
  }
});

// IPC handlers
ipcMain.handle('get-app-version', () => app.getVersion());
ipcMain.handle('get-backend-status', () => backendManager?.isRunning() ?? false);
ipcMain.handle('get-redis-status', () => redisManager?.isRunning() ?? false);

ipcMain.handle('toggle-autostart', (_event, enabled) => {
  app.setLoginItemSettings({
    openAtLogin: enabled,
    path: app.getPath('exe'),
  });
  return app.getLoginItemSettings().openAtLogin;
});

ipcMain.handle('get-autostart', () => {
  return app.getLoginItemSettings().openAtLogin;
});

ipcMain.handle('restart-backend', async () => {
  if (backendManager) {
    await backendManager.stop();
    await backendManager.start();
    return true;
  }
  return false;
});
