const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { BackendManager } = require('./backend-manager');
const { RedisManager } = require('./redis-manager');
const { LicenseManager } = require('./license-manager');
const { createTray, updateTrayStatus } = require('./tray');

const isDev = process.env.CEREBRO_DEV === '1';

// --- Edition detection ---
// Full Stack: bundled backend/ directory exists in resources
// Client: no backend/ → must connect to remote server
function detectEdition() {
  if (isDev) {
    return fs.existsSync(path.join(__dirname, '..', 'backend')) ? 'fullstack' : 'client';
  }
  return fs.existsSync(path.join(process.resourcesPath, 'backend')) ? 'fullstack' : 'client';
}

const edition = detectEdition();
const isFullStack = edition === 'fullstack';
const licenseManager = new LicenseManager();

// Remote URL: env var > config file > prompt user (client edition)
let REMOTE_URL = process.env.CEREBRO_REMOTE_URL || '';
if (!REMOTE_URL && !isFullStack) {
  try {
    const configPath = isDev
      ? path.join(__dirname, '..', 'cerebro-remote.json')
      : path.join(app.getPath('userData'), 'cerebro-remote.json');
    if (fs.existsSync(configPath)) {
      const cfg = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
      REMOTE_URL = cfg.serverUrl || '';
    }
  } catch (_) { /* no config file */ }
}

let mainWindow = null;
let splashWindow = null;
let tray = null;
let backendManager = null;
let redisManager = null;
let mcpManager = null;
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

// --- License check ---
async function checkLicense() {
  // Skip license check in dev mode
  if (isDev) return true;

  updateSplashStatus('Checking license...');

  const result = await licenseManager.revalidate();
  if (result.valid) {
    console.log(`[Main] License valid (plan: ${result.plan}${result.offline ? ', offline mode' : ''})`);
    return true;
  }

  console.log(`[Main] License invalid: ${result.reason}`);
  return false;
}

function showActivationScreen() {
  // Close splash, show activation in main window
  if (splashWindow) {
    splashWindow.close();
    splashWindow = null;
  }

  mainWindow.loadFile(path.join(__dirname, 'activation.html'));
  mainWindow.show();
  mainWindow.focus();
}

async function promptForServerUrl() {
  const { response } = await dialog.showMessageBox({
    type: 'question',
    title: 'Cerebro Client - Server Setup',
    message: 'No server URL configured.\n\nPlease set the CEREBRO_REMOTE_URL environment variable or create a cerebro-remote.json config file.\n\nExample config:\n{"serverUrl": "http://your-server:59000"}',
    buttons: ['OK', 'Quit'],
    defaultId: 0,
  });

  if (response === 1) {
    app.quit();
    return null;
  }
  return null;
}

async function startServices() {
  if (isFullStack) {
    // Full Stack mode — start Redis + Backend + MCP
    const serverUrl = 'http://localhost:59000';

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
      backendSrc: isDev ? path.join(__dirname, '..', 'backend-src') : null,
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

    // Start MCP server (Full Stack only)
    if (fs.existsSync(getResourcePath('mcp-server')) || isDev) {
      try {
        const { McpManager } = require('./mcp-manager');
        updateSplashStatus('Starting MCP memory server...');
        mcpManager = new McpManager({
          binaryPath: getResourcePath('mcp-server'),
          isDev,
          mcpSrc: isDev ? path.join(__dirname, '..', 'memory-src') : null,
        });
        await mcpManager.start();
        console.log('[Main] MCP server started');
      } catch (err) {
        // MCP is optional — don't block startup
        console.error('[Main] MCP server failed to start:', err.message);
      }
    }

    // Load frontend
    updateSplashStatus('Loading Cerebro...');
    mainWindow.loadURL(serverUrl);
  } else {
    // Client mode — connect to remote server
    if (!REMOTE_URL) {
      await promptForServerUrl();
      if (!REMOTE_URL) return;
    }

    console.log(`[Main] Client mode: connecting to ${REMOTE_URL}`);
    updateSplashStatus(`Connecting to ${REMOTE_URL}...`);
    mainWindow.loadURL(REMOTE_URL);
  }

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
      const url = isFullStack ? 'http://localhost:59000' : REMOTE_URL;
      if (mainWindow) mainWindow.loadURL(url);
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

  if (mcpManager) {
    try {
      await mcpManager.stop();
    } catch (err) {
      console.error('[Main] Error stopping MCP server:', err.message);
    }
  }

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
  console.log(`[Main] Edition: ${edition}`);
  createSplashWindow();
  createMainWindow();

  // Create system tray
  const trayIconPath = path.join(__dirname, '..', 'assets', 'tray-icon.png');
  tray = createTray(mainWindow, trayIconPath);

  // License gate
  const licensed = await checkLicense();
  if (!licensed) {
    showActivationScreen();
    return; // Don't start services until license is activated
  }

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
ipcMain.handle('get-edition', () => edition);
ipcMain.handle('get-backend-status', () => backendManager?.isRunning() ?? false);
ipcMain.handle('get-redis-status', () => redisManager?.isRunning() ?? false);
ipcMain.handle('get-mcp-status', () => mcpManager?.isRunning() ?? false);

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

// License IPC handlers
ipcMain.handle('get-license-status', () => {
  return licenseManager.getStatus();
});

ipcMain.handle('activate-license', async (_event, key) => {
  const result = await licenseManager.activate(key);
  if (result.success) {
    // License activated — start services after a brief delay
    console.log(`[Main] License activated (plan: ${result.plan}), starting services...`);
    setTimeout(async () => {
      await startServices();
      updateTrayStatus(tray, 'running');
    }, 500);
  }
  return result;
});

// Configure MCP for Claude Code (Full Stack only)
ipcMain.handle('configure-mcp', async () => {
  if (!isFullStack) return { success: false, error: 'Client edition does not bundle MCP' };

  const os = require('os');
  const mcpConfigPath = path.join(os.homedir(), '.claude', 'mcp.json');
  const mcpConfigDir = path.dirname(mcpConfigPath);

  const mcpBinaryDir = getResourcePath('mcp-server');
  const isWin = process.platform === 'win32';
  const binaryName = isWin ? 'cerebro-mcp-server.exe' : 'cerebro-mcp-server';
  const binaryPath = path.join(mcpBinaryDir, 'cerebro-mcp-server', binaryName);

  try {
    if (!fs.existsSync(mcpConfigDir)) {
      fs.mkdirSync(mcpConfigDir, { recursive: true });
    }

    let config = {};
    if (fs.existsSync(mcpConfigPath)) {
      config = JSON.parse(fs.readFileSync(mcpConfigPath, 'utf-8'));
    }

    if (!config.mcpServers) config.mcpServers = {};

    config.mcpServers['cerebro'] = {
      command: binaryPath,
      args: [],
      env: {
        CEREBRO_DATA_DIR: path.join(os.homedir(), '.cerebro', 'data'),
      },
    };

    fs.writeFileSync(mcpConfigPath, JSON.stringify(config, null, 2));
    return { success: true, configPath: mcpConfigPath };
  } catch (err) {
    return { success: false, error: err.message };
  }
});
