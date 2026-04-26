/**
 * native-manager.js — Native Runtime Manager for Cerebro v6
 *
 * Replaces docker-manager.js. Spawns and manages a PyInstaller-bundled
 * FastAPI backend as a child process — no containers, no Docker, no Redis
 * reservation tax.
 *
 * Architecture:
 *   - Backend: PyInstaller binary at process.resourcesPath/backend/cerebro-backend(.exe)
 *   - Frontend: static files at process.resourcesPath/frontend/
 *   - Data: ~/.cerebro/data/  (memory, FAISS index, logs)
 *   - Config: ~/.cerebro/.env, ~/.cerebro/.setup-state.json
 *   - Redis: optional (in-memory fallback baked into backend since v5.1)
 *
 * In dev mode (CEREBRO_DEV=1): runs `python -m uvicorn` against backend-src/.
 *
 * IPC API (mirrors docker-manager surface so main.js patches stay minimal):
 *   isRuntimeReady() / isBackendRunning() / startBackend() / stopBackend()
 *   getStatus() / getLogs() / writeEnvFile() / setClaudeCliPath()
 *   loadSetupState() / saveSetupState() / loadFileAccessConfig() / saveFileAccessConfig()
 *   isClaudeInstalled() / getClaudeCliPath()
 *   isDefenderExcluded() / addDefenderExclusion()
 *   startCredentialWatch() / stopCredentialWatch()
 *
 * Drops (Docker-specific, no longer needed):
 *   isDockerInstalled, isDockerRunning, installDocker, startDockerDaemon,
 *   pullImages, applyUpdate (electron-updater handles full app), checkWslAvailable,
 *   downloadDockerInstaller, writeComposeFile, _injectStorageMount, _injectFileMounts,
 *   _injectPorts, installKokoroTts (voice stripped per Professor's v6 spec),
 *   verifyStorageMount (no Docker volumes anymore)
 */

const { spawn, execFile } = require('child_process');
const { EventEmitter } = require('events');
const path = require('path');
const fs = require('fs');
const os = require('os');
const http = require('http');
const net = require('net');

const CEREBRO_DIR = path.join(os.homedir(), '.cerebro');
const ENV_FILE = path.join(CEREBRO_DIR, '.env');
const SETUP_STATE_FILE = path.join(CEREBRO_DIR, '.setup-state.json');
const FILE_ACCESS_CONFIG = path.join(CEREBRO_DIR, 'file-access.json');
const DATA_DIR = path.join(CEREBRO_DIR, 'data');
const LOG_DIR = path.join(CEREBRO_DIR, 'logs');
const BACKEND_LOG = path.join(LOG_DIR, 'backend.log');

class NativeManager extends EventEmitter {
  constructor() {
    super();
    this._proc = null;
    this._running = false;
    this._needsRestart = false;
    this._portConfig = null;
    this._logBuffer = [];
    this._logBufferMax = 1000;
    this._credWatchInterval = null;
    this._ensureDirs();
  }

  // ---------- Setup ----------

  _ensureDirs() {
    fs.mkdirSync(CEREBRO_DIR, { recursive: true });
    fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.mkdirSync(LOG_DIR, { recursive: true });
  }

  setPortConfig(cfg) {
    this._portConfig = cfg;
  }

  get backendPort() {
    return (this._portConfig && this._portConfig.backendPort) || 61000;
  }

  // ---------- Bundle paths ----------

  /**
   * Resolve the path to the bundled backend binary (or python script in dev).
   */
  _getBackendCommand() {
    const isDev = process.env.CEREBRO_DEV === '1';
    if (isDev) {
      // Dev: invoke python directly against backend-src/main.py
      const pythonExe = process.platform === 'win32' ? 'python.exe' : 'python3';
      const backendSrc = path.join(__dirname, '..', 'backend-src');
      const mainPy = path.join(backendSrc, 'main.py');
      return {
        cmd: pythonExe,
        args: ['-u', mainPy],
        cwd: backendSrc,
      };
    }

    // Production: invoke bundled binary from resourcesPath
    const binName = process.platform === 'win32' ? 'cerebro-backend.exe' : 'cerebro-backend';
    const resourcesPath = process.resourcesPath || path.join(__dirname, '..');
    const binDir = path.join(resourcesPath, 'backend');
    const binPath = path.join(binDir, binName);
    return {
      cmd: binPath,
      args: [],
      cwd: binDir,
    };
  }

  /**
   * Resolve frontend static directory for the backend to serve.
   */
  _getFrontendDir() {
    const isDev = process.env.CEREBRO_DEV === '1';
    if (isDev) {
      return path.join(__dirname, '..', 'frontend');
    }
    const resourcesPath = process.resourcesPath || path.join(__dirname, '..');
    return path.join(resourcesPath, 'frontend');
  }

  /**
   * Check whether the backend binary exists on disk (production check).
   * In dev mode this checks for python availability.
   */
  async isRuntimeReady() {
    const { cmd } = this._getBackendCommand();
    const isDev = process.env.CEREBRO_DEV === '1';
    if (isDev) {
      // Dev mode — just verify python exists
      return new Promise((resolve) => {
        execFile(cmd, ['--version'], { timeout: 5000 }, (err) => resolve(!err));
      });
    }
    return fs.existsSync(cmd);
  }

  // ---------- Environment file ----------

  writeEnvFile() {
    this._ensureDirs();
    const lines = [
      '# Cerebro v6 — auto-generated environment',
      `CEREBRO_DATA_DIR=${DATA_DIR}`,
      `CEREBRO_LOG_DIR=${LOG_DIR}`,
      `CEREBRO_FRONTEND_DIR=${this._getFrontendDir()}`,
      `CEREBRO_HOST=127.0.0.1`,
      `CEREBRO_PORT=${this.backendPort}`,
      `CEREBRO_CORS_ORIGINS=http://localhost:${this.backendPort},http://127.0.0.1:${this.backendPort}`,
      `CEREBRO_EXTERNAL_PORT=${this.backendPort}`,
      // Redis is optional — backend falls back to in-memory if unset
      // Users can set CEREBRO_REDIS_URL=redis://localhost:6379 manually
      '',
    ];
    fs.writeFileSync(ENV_FILE, lines.join('\n'));
    return ENV_FILE;
  }

  // ---------- Process lifecycle ----------

  /**
   * Spawn the backend process. Resolves once health check passes.
   */
  async startBackend(options = {}) {
    if (this._proc && this._running) {
      console.log('[Native] Backend already running');
      return { success: true, alreadyRunning: true };
    }

    this.emit('status', 'starting');

    const ready = await this.isRuntimeReady();
    if (!ready) {
      this.emit('status', 'error');
      throw new Error('Backend binary not found. Reinstall Cerebro to repair the bundle.');
    }

    // Verify port available
    const portFree = await this._isPortFree(this.backendPort);
    if (!portFree) {
      // Could be us already running, or another app — probe health
      const healthy = await this._probeHealth(this.backendPort);
      if (healthy) {
        this._running = true;
        this.emit('status', 'running');
        return { success: true, alreadyRunning: true };
      }
      this.emit('status', 'error');
      throw new Error(`Port ${this.backendPort} is in use by another app. Change port in Settings → Connection.`);
    }

    this.writeEnvFile();

    const { cmd, args, cwd } = this._getBackendCommand();
    const env = {
      ...process.env,
      CEREBRO_DATA_DIR: DATA_DIR,
      CEREBRO_LOG_DIR: LOG_DIR,
      CEREBRO_FRONTEND_DIR: this._getFrontendDir(),
      CEREBRO_HOST: '127.0.0.1',
      CEREBRO_PORT: String(this.backendPort),
      CEREBRO_CORS_ORIGINS: `http://localhost:${this.backendPort},http://127.0.0.1:${this.backendPort}`,
      CEREBRO_EXTERNAL_PORT: String(this.backendPort),
      // Force unbuffered Python output so we can stream logs
      PYTHONUNBUFFERED: '1',
    };

    console.log(`[Native] Spawning backend: ${cmd} ${args.join(' ')}`);

    try {
      this._proc = spawn(cmd, args, {
        cwd,
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      });
    } catch (err) {
      this.emit('status', 'error');
      throw new Error(`Failed to spawn backend: ${err.message}`);
    }

    // Stream logs
    const onData = (chunk) => {
      const line = chunk.toString();
      this._appendLog(line);
    };
    this._proc.stdout.on('data', onData);
    this._proc.stderr.on('data', onData);

    this._proc.on('exit', (code, signal) => {
      console.log(`[Native] Backend exited: code=${code} signal=${signal}`);
      this._running = false;
      this._proc = null;
      this.emit('status', 'stopped');
      if (code !== 0 && code !== null) {
        this.emit('crashed', { code, signal });
      }
    });

    this._proc.on('error', (err) => {
      console.error('[Native] Backend process error:', err);
      this._running = false;
      this.emit('status', 'error');
    });

    // Wait for health check
    try {
      await this._waitForBackend(options.maxAttempts || 240);
      this._running = true;
      this.emit('status', 'running');
      return { success: true };
    } catch (err) {
      // Health check failed — try to capture last log lines for diagnostics
      const tail = this._logBuffer.slice(-20).join('');
      this.emit('status', 'error');
      try { this._proc && this._proc.kill(); } catch {}
      throw new Error(`Backend failed to become healthy: ${err.message}\n--- Last log lines ---\n${tail}`);
    }
  }

  /**
   * Gracefully stop the backend process.
   */
  async stopBackend() {
    if (!this._proc) {
      this._running = false;
      this.emit('status', 'stopped');
      return { success: true, alreadyStopped: true };
    }

    this.emit('status', 'stopping');

    return new Promise((resolve) => {
      const proc = this._proc;
      const timeout = setTimeout(() => {
        try { proc.kill('SIGKILL'); } catch {}
      }, 10000);

      proc.once('exit', () => {
        clearTimeout(timeout);
        this._running = false;
        this._proc = null;
        this.emit('status', 'stopped');
        resolve({ success: true });
      });

      try {
        if (process.platform === 'win32') {
          proc.kill();
        } else {
          proc.kill('SIGTERM');
        }
      } catch (err) {
        clearTimeout(timeout);
        this._running = false;
        this._proc = null;
        this.emit('status', 'stopped');
        resolve({ success: true, killError: err.message });
      }
    });
  }

  // ---------- Health & status ----------

  isRunning() {
    return this._running;
  }

  isBackendRunning() {
    return this._running;
  }

  isSetupComplete() {
    return fs.existsSync(ENV_FILE);
  }

  needsRestart() {
    return this._needsRestart;
  }

  async checkNeedsRestart() {
    return this._needsRestart;
  }

  wasDockerInstalledThisSession() {
    // Compatibility shim for main.js — always false in native build
    return false;
  }

  async getStatus() {
    const healthy = this._running ? await this._probeHealth(this.backendPort) : false;
    return {
      running: this._running,
      healthy,
      port: this.backendPort,
      pid: this._proc ? this._proc.pid : null,
      backendBinary: this._getBackendCommand().cmd,
    };
  }

  _probeHealth(port) {
    return new Promise((resolve) => {
      const req = http.get(`http://127.0.0.1:${port}/health`, { timeout: 2000 }, (res) => {
        res.resume();
        resolve(res.statusCode === 200);
      });
      req.on('error', () => resolve(false));
      req.on('timeout', () => { req.destroy(); resolve(false); });
    });
  }

  _isPortFree(port) {
    return new Promise((resolve) => {
      const server = net.createServer();
      server.once('error', () => resolve(false));
      server.once('listening', () => {
        server.close(() => resolve(true));
      });
      server.listen(port, '127.0.0.1');
    });
  }

  _waitForBackend(maxAttempts = 240) {
    return new Promise((resolve, reject) => {
      let attempts = 0;
      const check = async () => {
        attempts += 1;
        if (attempts > maxAttempts) {
          return reject(new Error(`Health check timed out after ${Math.round(maxAttempts * 0.5)}s`));
        }
        const ok = await this._probeHealth(this.backendPort);
        if (ok) return resolve();
        setTimeout(check, 500);
      };
      // Give the process a moment to bind the port
      setTimeout(check, 1500);
    });
  }

  // ---------- Logs ----------

  _appendLog(line) {
    this._logBuffer.push(line);
    if (this._logBuffer.length > this._logBufferMax) {
      this._logBuffer.shift();
    }
    // Also persist to log file (best effort, non-blocking)
    try {
      fs.appendFile(BACKEND_LOG, line, () => {});
    } catch {}
  }

  async getLogs(lines = 200) {
    // Prefer file (persisted across restarts) but fall back to in-memory buffer
    try {
      if (fs.existsSync(BACKEND_LOG)) {
        const data = fs.readFileSync(BACKEND_LOG, 'utf-8');
        const all = data.split('\n');
        return all.slice(-lines).join('\n');
      }
    } catch {}
    return this._logBuffer.slice(-lines).join('');
  }

  // ---------- Setup state ----------

  saveSetupState(state) {
    this._ensureDirs();
    fs.writeFileSync(SETUP_STATE_FILE, JSON.stringify({ ...state, savedAt: Date.now() }));
  }

  loadSetupState() {
    try {
      if (fs.existsSync(SETUP_STATE_FILE)) {
        return JSON.parse(fs.readFileSync(SETUP_STATE_FILE, 'utf-8'));
      }
    } catch {}
    return null;
  }

  clearSetupState() {
    try { fs.unlinkSync(SETUP_STATE_FILE); } catch {}
  }

  // ---------- File access (native semantics) ----------
  // In Docker mode this configured bind mounts. In native mode the backend
  // reads files directly from the OS — this config becomes an allow-list of
  // paths the backend is permitted to read/write on the user's behalf.

  loadFileAccessConfig() {
    try {
      if (fs.existsSync(FILE_ACCESS_CONFIG)) {
        return JSON.parse(fs.readFileSync(FILE_ACCESS_CONFIG, 'utf-8'));
      }
    } catch {}
    return { fileMounts: [] };
  }

  saveFileAccessConfig(config) {
    this._ensureDirs();
    fs.writeFileSync(FILE_ACCESS_CONFIG, JSON.stringify(config, null, 2));
  }

  getPresetMounts() {
    const homeDir = os.homedir();
    const presets = [
      { id: 'desktop', label: 'Desktop', folder: 'Desktop' },
      { id: 'documents', label: 'Documents', folder: 'Documents' },
      { id: 'downloads', label: 'Downloads', folder: 'Downloads' },
    ];
    const sshDir = path.join(homeDir, '.ssh');
    if (fs.existsSync(sshDir)) {
      presets.push({ id: 'devices', label: 'Devices (SSH)', folder: '.ssh' });
    }
    return presets.map(p => ({
      id: p.id,
      label: p.label,
      hostPath: path.join(homeDir, p.folder),
      // In native mode there's no container — hostPath IS the path
      containerPath: path.join(homeDir, p.folder),
      readOnly: true,
      preset: true,
    }));
  }

  // ---------- Claude CLI integration ----------

  async isClaudeInstalled() {
    const cliPath = await this.getClaudeCliPath();
    return !!cliPath;
  }

  async getClaudeCliPath() {
    // Check common install locations in priority order
    const candidates = [];
    const homeDir = os.homedir();

    if (process.platform === 'win32') {
      candidates.push(
        path.join(process.env.APPDATA || '', 'npm', 'claude.cmd'),
        path.join(process.env.APPDATA || '', 'npm', 'claude.exe'),
        path.join(homeDir, 'AppData', 'Roaming', 'npm', 'claude.cmd'),
      );
    } else {
      candidates.push(
        '/usr/local/bin/claude',
        '/usr/bin/claude',
        path.join(homeDir, '.local', 'bin', 'claude'),
        path.join(homeDir, '.nvm', 'versions', 'node', 'lts', 'bin', 'claude'),
      );
    }

    for (const c of candidates) {
      if (c && fs.existsSync(c)) return c;
    }

    // Fallback: search PATH
    return new Promise((resolve) => {
      const cmd = process.platform === 'win32' ? 'where' : 'which';
      execFile(cmd, ['claude'], { timeout: 5000 }, (err, stdout) => {
        if (err || !stdout) return resolve(null);
        const first = stdout.toString().split('\n')[0].trim();
        resolve(first || null);
      });
    });
  }

  async setClaudeCliPath() {
    const cliPath = await this.getClaudeCliPath();
    if (!cliPath) return null;
    // Write into env file for backend pickup
    try {
      let env = '';
      if (fs.existsSync(ENV_FILE)) env = fs.readFileSync(ENV_FILE, 'utf-8');
      if (!env.includes('CLAUDE_CLI_PATH=')) {
        env += `\nCLAUDE_CLI_PATH=${cliPath}\n`;
        fs.writeFileSync(ENV_FILE, env);
      }
    } catch (err) {
      console.warn('[Native] Failed to write CLAUDE_CLI_PATH:', err.message);
    }
    return cliPath;
  }

  // ---------- Credential watch ----------
  // OAuth credentials for Claude CLI live in ~/.claude/.credentials.json on host.
  // In native mode this is the ONLY copy — no container to sync to. We still
  // watch and emit events so the UI can prompt the user when refresh is needed.

  checkClaudeCredentials() {
    try {
      const credsPath = path.join(os.homedir(), '.claude', '.credentials.json');
      if (!fs.existsSync(credsPath)) {
        return { valid: false, error: 'No Claude credentials found' };
      }
      const data = JSON.parse(fs.readFileSync(credsPath, 'utf-8'));
      const oauth = data.claudeAiOauth || {};
      const expiresAt = oauth.expiresAt || 0;
      const now = Date.now();
      const expiresInMin = Math.floor((expiresAt - now) / 60000);
      return {
        valid: expiresAt > now,
        expiresIn: expiresInMin,
        expiresAt,
      };
    } catch (err) {
      return { valid: false, error: err.message };
    }
  }

  startCredentialWatch() {
    if (this._credWatchInterval) return;
    this._credWatchInterval = setInterval(() => {
      const status = this.checkClaudeCredentials();
      if (!status.valid) {
        this.emit('credentials-expired', {
          message: status.error || 'Claude credentials expired',
          needsLogin: true,
        });
      } else if (status.expiresIn !== undefined && status.expiresIn < 30) {
        this.emit('credentials-refreshed', {
          expiresIn: status.expiresIn,
          message: 'Credentials expiring soon',
          warning: true,
        });
      }
    }, 15 * 60 * 1000);
    console.log('[Native] Credential watch started (15min interval)');
  }

  stopCredentialWatch() {
    if (this._credWatchInterval) {
      clearInterval(this._credWatchInterval);
      this._credWatchInterval = null;
    }
  }

  // ---------- Windows Defender (kept from docker-manager) ----------
  // Cerebro spawns Chrome with --remote-debugging-port for browser automation,
  // which still trips Defender on Windows. Same exclusion logic applies.

  async isDefenderExcluded() {
    if (process.platform !== 'win32') return true;
    try {
      const appDir = path.dirname(process.execPath);
      const exeName = path.basename(process.execPath);
      const tmpdir = os.tmpdir();
      const ts = Date.now();
      const resultFile = path.join(tmpdir, `cerebro-defcheck-${ts}.txt`);
      const scriptPath = path.join(tmpdir, `cerebro-defcheck-${ts}.ps1`);
      const scriptContent = [
        '$ErrorActionPreference = "SilentlyContinue"',
        '$p = Get-MpPreference',
        `$hasPath = $p.ExclusionPath -contains "${appDir}"`,
        `$hasProc = $p.ExclusionProcess -contains "${exeName}"`,
        `Set-Content -Path "${resultFile}" -Value "$hasPath|$hasProc"`,
      ].join('\r\n');
      fs.writeFileSync(scriptPath, scriptContent, 'utf-8');
      try {
        await this._runElevated('powershell.exe', [
          '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', scriptPath,
        ], { timeout: 15000 });
        const result = fs.readFileSync(resultFile, 'utf-8').trim();
        const [hasPath, hasProc] = result.split('|');
        return hasPath === 'True' && hasProc === 'True';
      } finally {
        try { fs.unlinkSync(scriptPath); } catch {}
        try { fs.unlinkSync(resultFile); } catch {}
      }
    } catch {
      return false;
    }
  }

  async addDefenderExclusion() {
    if (process.platform !== 'win32') return { success: true, skipped: true };
    try {
      const appDir = path.dirname(process.execPath);
      const exeName = path.basename(process.execPath);
      const dataDir = CEREBRO_DIR;
      const scriptPath = path.join(os.tmpdir(), `cerebro-defadd-${Date.now()}.ps1`);
      const scriptContent = [
        '$ErrorActionPreference = "Stop"',
        `Add-MpPreference -ExclusionPath "${appDir}" -ErrorAction SilentlyContinue`,
        `Add-MpPreference -ExclusionPath "${dataDir}" -ErrorAction SilentlyContinue`,
        `Add-MpPreference -ExclusionProcess "${exeName}" -ErrorAction SilentlyContinue`,
        `Add-MpPreference -ExclusionProcess "cerebro-backend.exe" -ErrorAction SilentlyContinue`,
        `Add-MpPreference -ExclusionProcess "chrome.exe" -ErrorAction SilentlyContinue`,
      ].join('\r\n');
      fs.writeFileSync(scriptPath, scriptContent, 'utf-8');
      try {
        await this._runElevated('powershell.exe', [
          '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', scriptPath,
        ], { timeout: 30000 });
        return { success: true };
      } finally {
        try { fs.unlinkSync(scriptPath); } catch {}
      }
    } catch (err) {
      return { success: false, error: err.message };
    }
  }

  _runElevated(cmd, args, options = {}) {
    // Launch elevated via PowerShell Start-Process -Verb RunAs
    return new Promise((resolve, reject) => {
      const argString = args.map(a => `'${a.replace(/'/g, "''")}'`).join(',');
      const psCmd = `Start-Process -FilePath "${cmd}" -ArgumentList ${argString} -Verb RunAs -Wait -WindowStyle Hidden`;
      execFile('powershell.exe', ['-NoProfile', '-Command', psCmd], {
        timeout: options.timeout || 30000,
      }, (err, stdout, stderr) => {
        if (err) return reject(new Error(stderr || err.message));
        resolve(stdout);
      });
    });
  }

  // ---------- Compatibility shims for main.js ----------
  // These exist so that main.js IPC handlers don't all break at once during
  // the migration. They no-op or return sensible defaults for Docker concepts
  // that don't exist anymore.

  async isDockerInstalled() {
    // In native build there is no Docker requirement
    return true;
  }

  async isDockerRunning() {
    return true;
  }

  async installDocker() {
    return { success: true, skipped: true, message: 'Cerebro v6 is native — no Docker needed' };
  }

  async startDockerDaemon() {
    return { success: true, skipped: true };
  }

  async pullImages(onProgress) {
    if (onProgress) onProgress({ stage: 'native', percent: 100 });
    return { success: true, skipped: true };
  }

  async writeComposeFile() {
    // No-op — native build has no docker-compose
    return null;
  }

  async checkForUpdates() {
    // electron-updater handles full app updates now. Backend ships with the app.
    return { updateAvailable: false, edition: 'native' };
  }

  async applyUpdate() {
    return { success: false, message: 'Use Settings → Check for Updates (electron-updater)' };
  }

  async checkWslAvailable() {
    return { available: true, skipped: true };
  }

  async verifyStorageMount() {
    // Memory lives in ~/.cerebro/data/ — verify it's writable
    try {
      const testFile = path.join(DATA_DIR, '.write-test');
      fs.writeFileSync(testFile, 'ok');
      fs.unlinkSync(testFile);
      return { healthy: true, path: DATA_DIR, warnings: [] };
    } catch (err) {
      return { healthy: false, error: err.message, path: DATA_DIR };
    }
  }

  // Aliases for IPC compatibility
  async startStack(opts) { return this.startBackend(opts); }
  async stopStack(opts) { return this.stopBackend(opts); }
}

module.exports = { NativeManager };
