const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const { DockerManager } = require('./docker-manager');
const { LicenseManager } = require('./license-manager');
const { createTray, updateTrayStatus } = require('./tray');

// Linux GPU handling: use real GPU if available, avoid SwiftShader fallback
if (process.platform === 'linux') {
  // Force Chromium to use the actual GPU (NVIDIA/Intel) instead of SwiftShader blocklist fallback
  app.commandLine.appendSwitch('ignore-gpu-blocklist');
  app.commandLine.appendSwitch('enable-gpu-rasterization');
  // Disable SwiftShader specifically — if no real GPU, prefer basic software rendering over SwiftShader
  app.commandLine.appendSwitch('disable-software-rasterizer');
}

const isDev = process.env.CEREBRO_DEV === '1';
const dockerManager = new DockerManager();
const licenseManager = new LicenseManager();

let mainWindow = null;
let splashWindow = null;
let tray = null;
let isQuitting = false;

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

function updateSplashStatus(message) {
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.webContents.executeJavaScript(
      `document.getElementById('status').textContent = ${JSON.stringify(message)}`
    ).catch(() => {});
  }
}

async function checkLicense() {
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

function showSetupWizard() {
  if (splashWindow) {
    splashWindow.close();
    splashWindow = null;
  }

  mainWindow.loadFile(path.join(__dirname, 'activation.html'));
  mainWindow.show();
  mainWindow.focus();
}

async function startDockerStack() {
  updateSplashStatus('Starting Docker containers...');

  // 1. Check if Docker is installed — if not, wizard handles install
  const installed = await dockerManager.isDockerInstalled();
  if (!installed) {
    console.log('[Main] Docker not installed, deferring to wizard');
    return { ok: false, error: 'Docker is not installed' };
  }

  // 2. Check if Docker daemon is running — auto-start if not
  const running = await dockerManager.isDockerRunning();
  if (!running) {
    updateSplashStatus('Starting Docker...');
    const startResult = await dockerManager.startDockerDaemon((p) => {
      updateSplashStatus(p.message || 'Starting Docker...');
    });
    if (!startResult.success) {
      console.error('[Main] Failed to start Docker daemon:', startResult.error);
      return { ok: false, error: `Failed to start Docker daemon: ${startResult.error}` };
    }
  }

  // 3. Write/refresh config (always refresh compose to fix volume mounts)
  updateSplashStatus('Writing configuration...');
  await dockerManager.writeComposeFile();
  dockerManager.writeEnvFile();
  await dockerManager.setClaudeCliPath();

  // 4. Start the compose stack
  try {
    updateSplashStatus('Starting containers...');
    await dockerManager.startStack();
    return { ok: true };
  } catch (err) {
    console.error('[Main] Docker stack failed to start:', err.message);
    return { ok: false, error: err.message };
  }
}

async function loadFrontend() {
  updateSplashStatus('Loading Cerebro...');

  let retries = 0;
  const maxRetries = 15;

  mainWindow.loadURL('http://localhost:61000');

  mainWindow.webContents.on('did-finish-load', () => {
    if (splashWindow) {
      splashWindow.close();
      splashWindow = null;
    }
    mainWindow.show();
    mainWindow.focus();

    // Check for updates in the background
    checkForUpdatesQuietly();
  });

  mainWindow.webContents.on('did-fail-load', (_event, errorCode, errorDescription) => {
    retries++;
    console.error(`[Main] Frontend failed to load (attempt ${retries}/${maxRetries}): ${errorCode} ${errorDescription}`);
    if (retries < maxRetries) {
      setTimeout(() => {
        if (mainWindow) mainWindow.loadURL('http://localhost:61000');
      }, 2000);
    } else {
      console.error('[Main] Frontend failed to load after max retries, showing wizard');
      if (mainWindow) showSetupWizard();
    }
  });
}

async function checkForUpdatesQuietly() {
  try {
    const result = await dockerManager.checkForUpdates();
    if (result.updateAvailable && mainWindow) {
      mainWindow.webContents.executeJavaScript(`
        if (typeof window.__cerebroShowUpdateBanner === 'function') {
          window.__cerebroShowUpdateBanner();
        }
      `).catch(() => {});
    }
  } catch {
    // Silent failure for update checks
  }
}

// --- Linux desktop integration (first launch) ---
function installDesktopIntegration() {
  if (process.platform !== 'linux') return;

  const os = require('os');
  const homeDir = os.homedir();
  const markerPath = path.join(app.getPath('userData'), '.desktop-integrated');

  if (fs.existsSync(markerPath)) return;

  try {
    const iconSrc = path.join(__dirname, '..', 'assets', 'icon-256.png');
    const iconFile = fs.existsSync(iconSrc) ? iconSrc : path.join(__dirname, '..', 'assets', 'icon.png');

    if (!fs.existsSync(iconFile)) {
      console.log('[Main] No icon file found, skipping desktop integration');
      return;
    }

    const iconDestDir = path.join(homeDir, '.local', 'share', 'icons', 'hicolor', '256x256', 'apps');
    fs.mkdirSync(iconDestDir, { recursive: true });
    fs.copyFileSync(iconFile, path.join(iconDestDir, 'cerebro.png'));

    const desktopDir = path.join(homeDir, '.local', 'share', 'applications');
    fs.mkdirSync(desktopDir, { recursive: true });

    const appImagePath = process.env.APPIMAGE || process.execPath;
    const desktopEntry = [
      '[Desktop Entry]',
      'Name=Cerebro',
      'Comment=Your AI, Everywhere',
      `Exec="${appImagePath}" %U`,
      'Icon=cerebro',
      'Type=Application',
      'Categories=Utility;ArtificialIntelligence;Science;',
      'StartupWMClass=cerebro-desktop',
      'Terminal=false',
    ].join('\n') + '\n';

    fs.writeFileSync(path.join(desktopDir, 'cerebro.desktop'), desktopEntry);

    const { execFileSync } = require('child_process');
    try {
      execFileSync('update-desktop-database', [desktopDir], { timeout: 5000 });
    } catch (_) { /* not critical */ }
    try {
      execFileSync('gtk-update-icon-cache', ['-f', '-t', path.join(homeDir, '.local', 'share', 'icons', 'hicolor')], { timeout: 5000 });
    } catch (_) { /* not critical */ }

    fs.writeFileSync(markerPath, new Date().toISOString());
    console.log('[Main] Desktop integration installed');
  } catch (err) {
    console.error('[Main] Desktop integration failed:', err.message);
  }
}

async function shutdown() {
  console.log('[Main] Shutting down...');
  await dockerManager.stopStack();
}

// App lifecycle
app.whenReady().then(async () => {
  console.log('[Main] Cerebro Desktop starting (Docker architecture)');

  installDesktopIntegration();

  createSplashWindow();
  createMainWindow();

  // Create system tray
  const trayIconPath = path.join(__dirname, '..', 'assets', 'tray-icon.png');
  tray = createTray(mainWindow, trayIconPath);

  // Post-restart resume: check for saved setup state before license gate
  const savedState = dockerManager.loadSetupState();
  if (savedState && savedState.step === 'needs-setup') {
    console.log('[Main] Resuming setup after restart');
    showSetupWizard();
    // Signal the activation page to jump to the setup step once loaded
    mainWindow.webContents.on('did-finish-load', () => {
      mainWindow.webContents.send('resume-after-restart', savedState);
    });
    return;
  }

  // License gate
  const licensed = await checkLicense();
  if (!licensed) {
    showSetupWizard();
    return;
  }

  // Forward credential events to renderer
  dockerManager.on('credentials-expired', (data) => {
    if (mainWindow) mainWindow.webContents.send('credentials-expired', data);
  });
  dockerManager.on('credentials-refreshed', (data) => {
    if (mainWindow) mainWindow.webContents.send('credentials-refreshed', data);
  });

  // Returning user: start Docker stack and load frontend
  const dockerResult = await startDockerStack();
  if (!dockerResult.ok) {
    // Docker not running or not setup — show wizard
    console.error('[Main] Docker start failed:', dockerResult.error);
    showSetupWizard();
    return;
  }

  await loadFrontend();
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

// --- IPC handlers ---

// App info
ipcMain.handle('get-app-version', () => app.getVersion());
ipcMain.handle('get-edition', () => 'docker');

// License
ipcMain.handle('get-license-status', () => licenseManager.getStatus());
ipcMain.handle('activate-license', async (_event, key) => {
  const result = await licenseManager.activate(key);
  if (result.success) {
    console.log(`[Main] License activated (plan: ${result.plan})`);
  }
  return result;
});

// Docker status
ipcMain.handle('check-docker', async () => {
  const installed = await dockerManager.isDockerInstalled();
  if (!installed) return { installed: false, running: false };
  const running = await dockerManager.isDockerRunning();
  return { installed, running };
});

ipcMain.handle('check-claude-code', async () => {
  return dockerManager.isClaudeInstalled();
});

ipcMain.handle('get-docker-status', async () => {
  return dockerManager.getStatus();
});

ipcMain.handle('get-docker-logs', async () => {
  return dockerManager.getLogs();
});

// Docker install & daemon management
ipcMain.handle('install-docker', async () => {
  try {
    const result = await dockerManager.installDocker((progress) => {
      if (mainWindow) {
        mainWindow.webContents.send('docker-install-progress', progress);
      }
    });
    return result;
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('check-wsl', async () => {
  return dockerManager.checkWslAvailable();
});

ipcMain.handle('start-docker-daemon', async () => {
  try {
    const result = await dockerManager.startDockerDaemon((progress) => {
      if (mainWindow) {
        mainWindow.webContents.send('docker-start-progress', progress);
      }
    });
    return result;
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Setup & lifecycle
ipcMain.handle('setup-docker', async () => {
  try {
    await dockerManager.writeComposeFile();
    dockerManager.writeEnvFile();
    await dockerManager.setClaudeCliPath();
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('pull-images', async () => {
  try {
    await dockerManager.pullImages((progress) => {
      if (mainWindow) {
        mainWindow.webContents.send('pull-progress', progress);
      }
    });
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('start-stack', async () => {
  try {
    await dockerManager.startStack();
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('stop-stack', async () => {
  try {
    await dockerManager.stopStack();
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Updates
ipcMain.handle('check-for-updates', async () => {
  return dockerManager.checkForUpdates();
});

ipcMain.handle('apply-update', async () => {
  try {
    await dockerManager.applyUpdate((progress) => {
      if (mainWindow) {
        mainWindow.webContents.send('update-progress', progress);
      }
    });
    // Reload frontend after update
    mainWindow.loadURL('http://localhost:61000');
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// MCP configuration (for Claude Code integration)
ipcMain.handle('configure-mcp', async () => {
  const os = require('os');
  const mcpConfigPath = path.join(os.homedir(), '.claude', 'mcp.json');
  const mcpConfigDir = path.dirname(mcpConfigPath);

  try {
    if (!fs.existsSync(mcpConfigDir)) {
      fs.mkdirSync(mcpConfigDir, { recursive: true });
    }

    let config = {};
    if (fs.existsSync(mcpConfigPath)) {
      config = JSON.parse(fs.readFileSync(mcpConfigPath, 'utf-8'));
    }

    if (!config.mcpServers) config.mcpServers = {};

    // Point MCP at the Docker container's memory server
    // Use 'docker run' instead of 'docker exec' because the memory container
    // exits after init — it doesn't stay running as a persistent service.
    config.mcpServers['cerebro'] = {
      command: 'docker',
      args: [
        'run', '--rm', '-i',
        '-v', 'cerebro_cerebro-data:/data/memory',
        '-e', 'AI_MEMORY_PATH=/data/memory',
        '-e', 'CEREBRO_STANDALONE=1',
        'ghcr.io/professor-low/cerebro-memory:latest',
        'cerebro', 'serve',
      ],
      env: {},
    };

    fs.writeFileSync(mcpConfigPath, JSON.stringify(config, null, 2));
    return { success: true, configPath: mcpConfigPath };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// Wizard completion: start stack and load frontend
ipcMain.handle('wizard-complete', async () => {
  try {
    const result = await startDockerStack();
    if (!result.ok) {
      return { success: false, error: result.error || 'Failed to start Docker stack' };
    }
    await loadFrontend();
    updateTrayStatus(tray, 'running');
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('get-setup-status', async () => {
  const dockerStatus = await dockerManager.isDockerRunning().catch(() => false);
  const claudeStatus = await dockerManager.isClaudeInstalled().catch(() => ({ installed: false }));

  return {
    licensed: licenseManager.getStatus().valid,
    dockerInstalled: await dockerManager.isDockerInstalled().catch(() => false),
    dockerRunning: dockerStatus,
    claudeInstalled: claudeStatus.installed,
    setupComplete: dockerManager.isSetupComplete(),
    stackRunning: dockerManager.isRunning(),
  };
});

// Settings
ipcMain.handle('toggle-autostart', (_event, enabled) => {
  app.setLoginItemSettings({
    openAtLogin: enabled,
    path: process.env.APPIMAGE || app.getPath('exe'),
  });
  return app.getLoginItemSettings().openAtLogin;
});

ipcMain.handle('get-autostart', () => {
  return app.getLoginItemSettings().openAtLogin;
});

ipcMain.handle('enable-autostart', () => {
  app.setLoginItemSettings({
    openAtLogin: true,
    path: process.env.APPIMAGE || app.getPath('exe'),
  });
  return true;
});

// Claude credentials
ipcMain.handle('check-claude-credentials', () => {
  return dockerManager.checkClaudeCredentials();
});

ipcMain.handle('refresh-claude-credentials', () => {
  return dockerManager.refreshClaudeCredentials();
});

ipcMain.handle('silent-refresh-oauth', async () => {
  return dockerManager.silentRefreshOAuthToken();
});

// Launch Claude CLI login — captures OAuth URL and opens in-app browser window
let _authWindow = null;

ipcMain.handle('launch-claude-login', async () => {
  try {
    // Close any existing auth window
    if (_authWindow && !_authWindow.isDestroyed()) _authWindow.close();

    // Spawn `claude auth login` with BROWSER env trick to capture the URL
    // On Unix: BROWSER=echo prints the URL to stdout instead of opening a browser
    // On Windows: we parse stdout/stderr for the URL pattern as fallback
    const env = { ...process.env };
    if (process.platform !== 'win32') {
      env.BROWSER = 'echo';
    }

    return new Promise((resolve) => {
      const proc = spawn('claude', ['auth', 'login'], {
        env,
        shell: true,
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      let output = '';
      let urlFound = false;
      const urlRegex = /https:\/\/[^\s"'<>]+/g;

      function handleOutput(chunk) {
        const text = chunk.toString();
        output += text;
        console.log('[Claude Auth]', text.trim());

        // Look for OAuth URL in output
        if (!urlFound) {
          const matches = text.match(urlRegex);
          if (matches) {
            // Find the auth/OAuth URL (prefer claude.ai or anthropic URLs)
            const authUrl = matches.find(u =>
              u.includes('claude.ai') || u.includes('anthropic.com') || u.includes('oauth')
            ) || matches[0];

            if (authUrl) {
              urlFound = true;
              openAuthWindow(authUrl);
              resolve({ success: true, message: 'Login window opened inside Cerebro.' });
            }
          }
        }
      }

      proc.stdout.on('data', handleOutput);
      proc.stderr.on('data', handleOutput);

      // If CLI opens browser directly (Windows), we still poll for credentials
      setTimeout(() => {
        if (!urlFound) {
          // Fallback: CLI probably opened browser directly, still poll for completion
          console.log('[Claude Auth] No URL captured, CLI may have opened browser directly');
          resolve({ success: true, message: 'Login opened — complete authentication in the browser.' });
        }
      }, 8000);

      proc.on('error', (err) => {
        console.error('[Claude Auth] Process error:', err.message);
        if (!urlFound) {
          resolve({ success: false, error: 'Could not start claude auth login: ' + err.message });
        }
      });

      proc.on('close', (code) => {
        console.log('[Claude Auth] Process exited with code', code);
        // Check if credentials were written successfully
        setTimeout(() => {
          const status = dockerManager.checkClaudeCredentials();
          if (status.valid) {
            const refreshResult = dockerManager.refreshClaudeCredentials();
            if (mainWindow) {
              mainWindow.webContents.send('credentials-refreshed', {
                expiresIn: refreshResult.expiresIn || status.expiresIn,
                message: 'Login successful — credentials refreshed',
              });
            }
            if (_authWindow && !_authWindow.isDestroyed()) _authWindow.close();
          }
        }, 1000);
      });

      // Poll for fresh credentials while login is in progress (5s intervals, 5 min max)
      let attempts = 0;
      const maxAttempts = 60;
      const watcher = setInterval(() => {
        attempts++;
        const status = dockerManager.checkClaudeCredentials();
        if (status.valid && status.expiresIn > 30) {
          clearInterval(watcher);
          const refreshResult = dockerManager.refreshClaudeCredentials();
          if (mainWindow) {
            mainWindow.webContents.send('credentials-refreshed', {
              expiresIn: refreshResult.expiresIn || status.expiresIn,
              message: 'Login successful — credentials refreshed',
            });
          }
          // Close auth window and kill CLI process
          if (_authWindow && !_authWindow.isDestroyed()) _authWindow.close();
          try { proc.kill(); } catch (_) {}
        }
        if (attempts >= maxAttempts) {
          clearInterval(watcher);
          try { proc.kill(); } catch (_) {}
        }
      }, 5000);
    });
  } catch (err) {
    return { success: false, error: err.message };
  }
});

function openAuthWindow(url) {
  _authWindow = new BrowserWindow({
    width: 500,
    height: 700,
    title: 'Cerebro — Claude Authentication',
    parent: mainWindow,
    modal: true,
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  _authWindow.loadURL(url);

  _authWindow.on('closed', () => {
    _authWindow = null;
  });

  // Watch for successful redirects back to localhost (OAuth callback)
  _authWindow.webContents.on('will-redirect', (_event, redirectUrl) => {
    if (redirectUrl.includes('localhost') || redirectUrl.includes('127.0.0.1')) {
      console.log('[Claude Auth] OAuth callback detected, login completing...');
    }
  });
}

// Restart & setup state
ipcMain.handle('needs-restart', async () => {
  return dockerManager.checkNeedsRestart();
});

ipcMain.handle('save-setup-state', (_event, state) => {
  dockerManager.saveSetupState(state);
  return true;
});

ipcMain.handle('load-setup-state', () => {
  return dockerManager.loadSetupState();
});

ipcMain.handle('clear-setup-state', () => {
  dockerManager.clearSetupState();
  return true;
});

ipcMain.handle('restart-computer', () => {
  const { execFile } = require('child_process');
  if (process.platform === 'win32') {
    execFile('shutdown', ['/r', '/t', '5'], (err) => {
      if (err) console.error('[Main] Restart failed:', err.message);
    });
  } else {
    execFile('loginctl', ['reboot'], (err) => {
      if (err) console.error('[Main] Restart failed:', err.message);
    });
  }
  return true;
});
