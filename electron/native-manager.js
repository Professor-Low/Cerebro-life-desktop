/**
 * NativeManager — Manages bundled backend + Redis processes natively.
 * Replaces DockerManager. No Docker, no containers, no Defender issues.
 *
 * Architecture:
 *   Electron → spawns Redis (bundled binary) + Backend (PyInstaller binary)
 *   Both run as child processes, bound to 127.0.0.1 only.
 */

const { EventEmitter } = require('events');
const { spawn, execFile } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');
const net = require('net');
const crypto = require('crypto');
const http = require('http');

const CEREBRO_DIR = path.join(os.homedir(), '.cerebro');
const SETUP_STATE_FILE = path.join(CEREBRO_DIR, '.setup-state.json');
const FILE_ACCESS_CONFIG = path.join(CEREBRO_DIR, 'file-access.json');
const MEMORY_DIR = path.join(CEREBRO_DIR, 'memory');
const REDIS_DATA_DIR = path.join(CEREBRO_DIR, 'redis-data');
const LOG_DIR = path.join(CEREBRO_DIR, 'logs');

class NativeManager extends EventEmitter {
  constructor() {
    super();
    this._backendProcess = null;
    this._redisProcess = null;
    this._running = false;
    this._portConfig = null;
    this._secret = null;
    this._token = null;
    this._resourcesPath = null;
    this._ensureDirs();
  }

  _ensureDirs() {
    for (const dir of [CEREBRO_DIR, MEMORY_DIR, REDIS_DATA_DIR, LOG_DIR,
      path.join(MEMORY_DIR, 'agents'),
      path.join(MEMORY_DIR, 'cerebro', 'cognitive_loop'),
      path.join(MEMORY_DIR, 'cerebro', 'skills'),
      path.join(MEMORY_DIR, 'cerebro', 'chrome_profile'),
      path.join(MEMORY_DIR, 'cerebro', 'recordings'),
      path.join(MEMORY_DIR, 'embeddings', 'chunks'),
      path.join(MEMORY_DIR, 'learnings'),
      path.join(MEMORY_DIR, 'projects'),
      path.join(MEMORY_DIR, 'agent_contexts'),
      path.join(MEMORY_DIR, 'mood'),
      path.join(MEMORY_DIR, 'conversations'),
      path.join(MEMORY_DIR, 'schedules'),
    ]) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  setPortConfig(cfg) {
    this._portConfig = cfg;
  }

  get backendPort() {
    return (this._portConfig && this._portConfig.backendPort) || 59000;
  }

  get redisPort() {
    return (this._portConfig && this._portConfig.redisPort) || 16379;
  }

  /**
   * Resolve path to bundled resources (extraResources in electron-builder).
   * In dev mode, falls back to project root.
   */
  _getResourcesPath() {
    if (this._resourcesPath) return this._resourcesPath;

    const { app } = require('electron');
    // In packaged app: process.resourcesPath points to <install>/resources/
    // extraResources are copied there by electron-builder
    const candidates = [
      path.join(process.resourcesPath || '', 'bin'),
      path.join(app.getAppPath(), '..', 'bin'),
      path.join(app.getAppPath(), 'bin'),
      // Dev mode fallback
      path.join(__dirname, '..', 'bin'),
    ];

    for (const p of candidates) {
      if (fs.existsSync(p)) {
        this._resourcesPath = p;
        console.log(`[Native] Resources path: ${p}`);
        return p;
      }
    }

    // Last resort
    this._resourcesPath = candidates[0];
    return this._resourcesPath;
  }

  _getBackendBinaryPath() {
    const binDir = this._getResourcesPath();
    if (process.platform === 'win32') {
      return path.join(binDir, 'cerebro-server', 'cerebro-server.exe');
    }
    return path.join(binDir, 'cerebro-server', 'cerebro-server');
  }

  _getRedisBinaryPath() {
    const binDir = this._getResourcesPath();
    if (process.platform === 'win32') {
      return path.join(binDir, 'redis', 'redis-server.exe');
    }
    return path.join(binDir, 'redis', 'redis-server');
  }

  _getFrontendPath() {
    const binDir = this._getResourcesPath();
    // Frontend is bundled alongside the backend
    const candidates = [
      path.join(binDir, '..', 'frontend'),
      path.join(binDir, 'frontend'),
      path.join(__dirname, '..', 'frontend'),
    ];
    for (const p of candidates) {
      if (fs.existsSync(p)) return p;
    }
    return candidates[0];
  }

  /**
   * Generate or load the JWT secret for backend auth.
   */
  _ensureSecret() {
    const secretFile = path.join(CEREBRO_DIR, '.cerebro_secret');
    if (fs.existsSync(secretFile)) {
      this._secret = fs.readFileSync(secretFile, 'utf-8').trim();
    } else {
      this._secret = crypto.randomBytes(32).toString('hex');
      fs.writeFileSync(secretFile, this._secret, { mode: 0o600 });
    }
    return this._secret;
  }

  /**
   * Generate a long-lived JWT token for internal API auth.
   */
  _generateToken() {
    // Simple HMAC-based token (not full JWT, but sufficient for localhost auth)
    const payload = JSON.stringify({
      sub: 'cerebro-desktop',
      iat: Math.floor(Date.now() / 1000),
      exp: Math.floor(Date.now() / 1000) + 365 * 86400,
    });
    const header = Buffer.from(JSON.stringify({ alg: 'HS256', typ: 'JWT' })).toString('base64url');
    const body = Buffer.from(payload).toString('base64url');
    const signature = crypto.createHmac('sha256', this._secret)
      .update(`${header}.${body}`).digest('base64url');
    this._token = `${header}.${body}.${signature}`;

    // Persist token
    const tokenFile = path.join(CEREBRO_DIR, '.cerebro_token');
    fs.writeFileSync(tokenFile, this._token, { mode: 0o600 });
    return this._token;
  }

  /**
   * Check if a port is available.
   */
  _isPortAvailable(port) {
    return new Promise((resolve) => {
      const server = net.createServer();
      server.once('error', () => resolve(false));
      server.once('listening', () => {
        server.close(() => resolve(true));
      });
      server.listen(port, '127.0.0.1');
    });
  }

  /**
   * Wait for a port to become available (process started).
   */
  _waitForPort(port, timeoutMs = 30000) {
    const start = Date.now();
    return new Promise((resolve, reject) => {
      const check = () => {
        if (Date.now() - start > timeoutMs) {
          reject(new Error(`Port ${port} did not become available within ${timeoutMs}ms`));
          return;
        }
        const socket = new net.Socket();
        socket.setTimeout(1000);
        socket.once('connect', () => {
          socket.destroy();
          resolve(true);
        });
        socket.once('error', () => {
          socket.destroy();
          setTimeout(check, 500);
        });
        socket.once('timeout', () => {
          socket.destroy();
          setTimeout(check, 500);
        });
        socket.connect(port, '127.0.0.1');
      };
      check();
    });
  }

  /**
   * Wait for backend health endpoint to respond.
   */
  _waitForHealth(timeoutMs = 60000) {
    const start = Date.now();
    return new Promise((resolve, reject) => {
      const check = () => {
        if (Date.now() - start > timeoutMs) {
          reject(new Error('Backend health check timed out'));
          return;
        }
        const req = http.get(`http://127.0.0.1:${this.backendPort}/health`, { timeout: 3000 }, (res) => {
          if (res.statusCode === 200) {
            resolve(true);
          } else {
            setTimeout(check, 1000);
          }
          res.resume();
        });
        req.on('error', () => setTimeout(check, 1000));
        req.on('timeout', () => { req.destroy(); setTimeout(check, 1000); });
      };
      check();
    });
  }

  /**
   * Start Redis server.
   */
  async startRedis(onProgress) {
    const redisPath = this._getRedisBinaryPath();

    if (!fs.existsSync(redisPath)) {
      throw new Error(`Redis binary not found at ${redisPath}`);
    }

    // Check if port is already in use (maybe Redis is already running)
    const portFree = await this._isPortAvailable(this.redisPort);
    if (!portFree) {
      console.log(`[Native] Redis port ${this.redisPort} already in use — assuming Redis is running`);
      return;
    }

    if (onProgress) onProgress({ stage: 'redis', message: 'Starting Redis...' });

    const args = [
      '--port', String(this.redisPort),
      '--bind', '127.0.0.1',
      '--dir', REDIS_DATA_DIR,
      '--save', '60', '1',        // Save every 60s if at least 1 key changed
      '--appendonly', 'no',
      '--loglevel', 'warning',
      '--daemonize', 'no',
      '--maxmemory', '256mb',
      '--maxmemory-policy', 'allkeys-lru',
    ];

    const logStream = fs.createWriteStream(path.join(LOG_DIR, 'redis.log'), { flags: 'a' });

    // Capture early stderr for diagnostics before piping to log
    let earlyStderr = '';

    const spawnOpts = {
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
      detached: false,
    };

    // On Windows, try shell mode as fallback if first attempt fails
    const attempts = process.platform === 'win32' ? 2 : 1;

    for (let attempt = 1; attempt <= attempts; attempt++) {
      if (attempt === 2) {
        console.log('[Native] Redis: retrying with shell: true');
        spawnOpts.shell = true;
      }

      this._redisProcess = spawn(redisPath, args, spawnOpts);

      // Capture stderr for early diagnostics
      earlyStderr = '';
      this._redisProcess.stderr.on('data', (chunk) => {
        earlyStderr += chunk.toString();
      });

      this._redisProcess.stdout.pipe(logStream);
      // Pipe stderr to log after a brief capture window
      setTimeout(() => {
        if (this._redisProcess && this._redisProcess.stderr) {
          this._redisProcess.stderr.pipe(logStream);
        }
      }, 2000);

      this._redisProcess.on('error', (err) => {
        console.error('[Native] Redis failed to start:', err.message);
      });

      this._redisProcess.on('exit', (code, signal) => {
        console.log(`[Native] Redis exited (code: ${code}, signal: ${signal})`);
        if (earlyStderr) console.error('[Native] Redis stderr:', earlyStderr.trim());
        this._redisProcess = null;
      });

      try {
        // Increased timeout: 30s (was 15s) — Windows Electron can be slow to spawn
        await this._waitForPort(this.redisPort, 30000);
        console.log(`[Native] Redis running on port ${this.redisPort}`);
        return; // Success — exit the retry loop
      } catch (err) {
        console.error(`[Native] Redis attempt ${attempt}/${attempts} failed: ${err.message}`);
        if (earlyStderr) console.error('[Native] Redis stderr:', earlyStderr.trim());
        // Kill the failed process before retry
        if (this._redisProcess) {
          try { this._redisProcess.kill(); } catch (_) {}
          this._redisProcess = null;
        }
        if (attempt === attempts) throw err; // Final attempt failed
        // Brief pause before retry
        await new Promise(r => setTimeout(r, 1000));
      }
    }
  }

  /**
   * Start the backend server.
   */
  async startBackend(onProgress) {
    const backendPath = this._getBackendBinaryPath();

    if (!fs.existsSync(backendPath)) {
      throw new Error(`Backend binary not found at ${backendPath}`);
    }

    // Ensure secret and token exist
    this._ensureSecret();
    this._generateToken();

    if (onProgress) onProgress({ stage: 'backend', message: 'Starting Cerebro backend...' });

    const frontendPath = this._getFrontendPath();

    // Ensure the backend can find the frontend — create symlink/junction if needed
    // The PyInstaller binary looks for frontend/ relative to its own directory
    const backendBinDir = path.dirname(backendPath);
    const expectedFrontend = path.join(backendBinDir, 'frontend');
    if (frontendPath && fs.existsSync(frontendPath) && !fs.existsSync(expectedFrontend)) {
      try {
        // On Windows use 'junction', on Unix use 'dir' symlink
        const linkType = process.platform === 'win32' ? 'junction' : 'dir';
        fs.symlinkSync(frontendPath, expectedFrontend, linkType);
        console.log(`[Native] Created ${linkType} link: ${expectedFrontend} -> ${frontendPath}`);
      } catch (err) {
        console.warn(`[Native] Could not create frontend link: ${err.message}`);
      }
    }

    const env = {
      ...process.env,
      CEREBRO_STANDALONE: '1',
      CEREBRO_HOST: '127.0.0.1',
      CEREBRO_PORT: String(this.backendPort),
      REDIS_URL: `redis://127.0.0.1:${this.redisPort}/0`,
      AI_MEMORY_PATH: MEMORY_DIR,
      CEREBRO_DATA_DIR: CEREBRO_DIR,
      CEREBRO_SECRET: this._secret,
      CEREBRO_TOKEN: this._token,
      CEREBRO_DEVICE: 'desktop',
      ENABLE_EMBEDDINGS: '1',
      CEREBRO_CORS_ORIGINS: `http://127.0.0.1:${this.backendPort},http://localhost:${this.backendPort}`,
      CEREBRO_FRONTEND_DIR: frontendPath,
      HOME: os.homedir(),
      // Tell the backend where MCP modules are (bundled alongside)
      CEREBRO_MCP_SRC: path.join(this._getResourcesPath(), 'mcp_modules'),
    };

    const logStream = fs.createWriteStream(path.join(LOG_DIR, 'backend.log'), { flags: 'a' });

    // PyInstaller dist is a directory — run the executable from within it
    const backendDir = path.dirname(backendPath);

    this._backendProcess = spawn(backendPath, [], {
      stdio: ['ignore', 'pipe', 'pipe'],
      env,
      cwd: backendDir,
      windowsHide: true,
      detached: false,
    });

    this._backendProcess.stdout.pipe(logStream);
    this._backendProcess.stderr.pipe(logStream);

    // Also capture stderr for error diagnosis
    this._backendProcess.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      if (text.includes('ERROR') || text.includes('Traceback')) {
        console.error('[Native Backend]', text.trim().slice(0, 200));
      }
    });

    this._backendProcess.on('error', (err) => {
      console.error('[Native] Backend failed to start:', err.message);
    });

    this._backendProcess.on('exit', (code, signal) => {
      console.log(`[Native] Backend exited (code: ${code}, signal: ${signal})`);
      this._backendProcess = null;
      this._running = false;
    });

    // Wait for health endpoint
    if (onProgress) onProgress({ stage: 'backend', message: 'Waiting for backend to initialize...' });
    await this._waitForHealth(90000);  // Backend can take a while on first start (model loading)
    this._running = true;
    console.log(`[Native] Backend running on port ${this.backendPort}`);
  }

  /**
   * Start the full stack (Redis + Backend).
   */
  async startStack(onProgress) {
    await this.startRedis(onProgress);
    await this.startBackend(onProgress);
  }

  /**
   * Stop everything.
   */
  async stopStack() {
    const kills = [];

    if (this._backendProcess) {
      console.log('[Native] Stopping backend...');
      kills.push(this._killProcess(this._backendProcess, 'backend'));
      this._backendProcess = null;
    }

    if (this._redisProcess) {
      console.log('[Native] Stopping Redis...');
      kills.push(this._killProcess(this._redisProcess, 'redis'));
      this._redisProcess = null;
    }

    await Promise.all(kills);
    this._running = false;
    console.log('[Native] All processes stopped');
  }

  _killProcess(proc, name) {
    return new Promise((resolve) => {
      if (!proc || proc.killed) {
        resolve();
        return;
      }

      const timeout = setTimeout(() => {
        console.warn(`[Native] ${name} didn't exit in time, force killing`);
        try { proc.kill('SIGKILL'); } catch (_) {}
        resolve();
      }, 5000);

      proc.once('exit', () => {
        clearTimeout(timeout);
        resolve();
      });

      try {
        // Graceful shutdown
        if (process.platform === 'win32') {
          proc.kill('SIGTERM');
        } else {
          proc.kill('SIGTERM');
        }
      } catch (_) {
        clearTimeout(timeout);
        resolve();
      }
    });
  }

  isRunning() {
    return this._running && this._backendProcess !== null;
  }

  /**
   * Check for Docker updates — returns false for native mode (updates are via electron-updater).
   */
  async checkForUpdates() {
    return { updateAvailable: false };
  }

  /**
   * Get backend logs.
   */
  getLogs(lines = 100) {
    const logFile = path.join(LOG_DIR, 'backend.log');
    try {
      if (!fs.existsSync(logFile)) return '';
      const content = fs.readFileSync(logFile, 'utf-8');
      const allLines = content.split('\n');
      return allLines.slice(-lines).join('\n');
    } catch {
      return '';
    }
  }


  // --- Docker Data Migration ---

  /**
   * Detect if Docker volumes from a previous Cerebro Docker installation exist.
   * Returns migration info if old data is found, null otherwise.
   */
  detectDockerData() {
    const { execFileSync } = require('child_process');
    const result = { found: false, volumes: [], estimatedSize: 0 };

    // Only check if native data dir is empty (fresh install)
    const memoryFiles = fs.readdirSync(MEMORY_DIR, { recursive: false });
    const hasExistingData = memoryFiles.some(f => !f.startsWith('.'));
    if (hasExistingData) {
      // User already has native data — don't offer migration
      return result;
    }

    try {
      // Check for Docker CLI
      const dockerPath = process.platform === 'win32' ? 'docker.exe' : 'docker';
      const volumeList = execFileSync(dockerPath, ['volume', 'ls', '--format', '{{.Name}}'], {
        timeout: 5000,
        windowsHide: true,
      }).toString().trim();

      const volumeNames = volumeList.split('\n').filter(Boolean);
      const cerebroVolumes = volumeNames.filter(v =>
        v.includes('cerebro') && (v.includes('data') || v.includes('redis'))
      );

      if (cerebroVolumes.length > 0) {
        result.found = true;
        result.volumes = cerebroVolumes;

        // Try to estimate size
        for (const vol of cerebroVolumes) {
          try {
            const inspect = execFileSync(dockerPath, ['volume', 'inspect', vol, '--format', '{{.Mountpoint}}'], {
              timeout: 5000,
              windowsHide: true,
            }).toString().trim();
            result.volumes.push({ name: vol, mountpoint: inspect });
          } catch {}
        }
      }
    } catch {
      // Docker not installed or not running — no migration needed
    }

    return result;
  }

  /**
   * Migrate data from Docker volumes to native ~/.cerebro/ directory.
   * Returns { success, migratedFiles, errors }.
   */
  async migrateFromDocker() {
    const { execFile } = require('child_process');
    const result = { success: false, migratedFiles: 0, errors: [] };

    try {
      const dockerPath = process.platform === 'win32' ? 'docker.exe' : 'docker';

      // Copy memory data from cerebro_cerebro-data volume
      const dataVolumes = ['cerebro_cerebro-data', 'cerebro-cerebro-data', 'cerebro_data'];
      for (const vol of dataVolumes) {
        try {
          // Use a temporary container to copy data out of the volume
          const tempContainer = `cerebro-migrate-${Date.now()}`;
          require('child_process').execFileSync(dockerPath, [
            'run', '--rm', '-d', '--name', tempContainer,
            '-v', `${vol}:/source:ro`,
            '-v', `${MEMORY_DIR}:/dest`,
            'alpine', 'sh', '-c', 'cp -a /source/. /dest/ && sleep 1'
          ], { timeout: 30000, windowsHide: true });

          // Wait for copy to complete
          await new Promise(r => setTimeout(r, 5000));
          try {
            require('child_process').execFileSync(dockerPath, ['rm', '-f', tempContainer], {
              timeout: 5000, windowsHide: true
            });
          } catch {}

          result.migratedFiles++;
          console.log(`[Migration] Copied data from volume: ${vol}`);
          break; // Only need one data volume
        } catch {}
      }

      // Copy Redis data
      const redisVolumes = ['cerebro_cerebro-redis', 'cerebro-cerebro-redis', 'cerebro_redis'];
      for (const vol of redisVolumes) {
        try {
          const tempContainer = `cerebro-migrate-redis-${Date.now()}`;
          require('child_process').execFileSync(dockerPath, [
            'run', '--rm', '-d', '--name', tempContainer,
            '-v', `${vol}:/source:ro`,
            '-v', `${REDIS_DATA_DIR}:/dest`,
            'alpine', 'sh', '-c', 'cp -a /source/. /dest/ && sleep 1'
          ], { timeout: 30000, windowsHide: true });

          await new Promise(r => setTimeout(r, 5000));
          try {
            require('child_process').execFileSync(dockerPath, ['rm', '-f', tempContainer], {
              timeout: 5000, windowsHide: true
            });
          } catch {}

          result.migratedFiles++;
          console.log(`[Migration] Copied Redis data from volume: ${vol}`);
          break;
        } catch {}
      }

      result.success = result.migratedFiles > 0;
      if (result.success) {
        // Mark migration as complete so we don't offer again
        fs.writeFileSync(path.join(CEREBRO_DIR, '.docker-migrated'), new Date().toISOString());
      }
    } catch (err) {
      result.errors.push(err.message);
    }

    return result;
  }

  /**
   * Check if Docker migration was already performed.
   */
  isDockerMigrationDone() {
    return fs.existsSync(path.join(CEREBRO_DIR, '.docker-migrated'));
  }


  // --- Setup state management ---
  isSetupComplete() {
    return fs.existsSync(path.join(CEREBRO_DIR, '.setup-complete'));
  }

  markSetupComplete() {
    fs.writeFileSync(path.join(CEREBRO_DIR, '.setup-complete'), new Date().toISOString());
  }

  saveSetupState(state) {
    fs.writeFileSync(SETUP_STATE_FILE, JSON.stringify(state, null, 2));
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

  // --- File access config (for Claude Code integration) ---
  loadFileAccessConfig() {
    try {
      if (fs.existsSync(FILE_ACCESS_CONFIG)) {
        return JSON.parse(fs.readFileSync(FILE_ACCESS_CONFIG, 'utf-8'));
      }
    } catch {}
    return { allowedDirs: [], readOnlyDirs: [] };
  }

  saveFileAccessConfig(config) {
    fs.writeFileSync(FILE_ACCESS_CONFIG, JSON.stringify(config, null, 2));
  }

  getPresetMounts() {
    const home = os.homedir();
    return [
      { name: 'Documents', path: path.join(home, 'Documents'), description: 'Your Documents folder' },
      { name: 'Desktop', path: path.join(home, 'Desktop'), description: 'Your Desktop folder' },
      { name: 'Downloads', path: path.join(home, 'Downloads'), description: 'Your Downloads folder' },
      { name: 'Projects', path: path.join(home, 'Projects'), description: 'Your Projects folder' },
    ].filter(p => fs.existsSync(p.path));
  }

  // --- Claude Code credentials ---
  checkClaudeCredentials() {
    const claudeDir = path.join(os.homedir(), '.claude');
    const credFiles = [
      path.join(claudeDir, '.credentials.json'),
      path.join(claudeDir, 'credentials.json'),
    ];

    for (const credFile of credFiles) {
      try {
        if (fs.existsSync(credFile)) {
          const creds = JSON.parse(fs.readFileSync(credFile, 'utf-8'));
          if (creds.claudeAiOauth) {
            const token = creds.claudeAiOauth;
            const expiresAt = token.expiresAt || token.expires_at;
            if (expiresAt) {
              const expiresIn = Math.floor((new Date(expiresAt).getTime() - Date.now()) / 1000);
              return { valid: expiresIn > 0, expiresIn, expiresAt, source: credFile };
            }
            return { valid: true, expiresIn: null, source: credFile };
          }
        }
      } catch {}
    }

    return { valid: false, expiresIn: 0, reason: 'no_credentials' };
  }

  refreshClaudeCredentials() {
    return this.checkClaudeCredentials();
  }

  async silentRefreshOAuthToken() {
    // In native mode, Claude CLI handles its own token refresh
    return this.checkClaudeCredentials();
  }

  /**
   * Set Claude CLI path — not needed in native mode (Claude CLI runs directly from PATH).
   */
  async setClaudeCliPath() {
    return;
  }

  /**
   * Check if Claude Code is installed.
   */
  async isClaudeInstalled() {
    return new Promise((resolve) => {
      execFile(process.platform === 'win32' ? 'where' : 'which', ['claude'], { timeout: 5000 },
        (err) => resolve({ installed: !err }));
    });
  }

  /**
   * Verify storage mount health.
   */
  async verifyStorageMount() {
    const warnings = [];
    let totalSize = 0;
    let availableSize = 0;

    try {
      const stats = fs.statfsSync ? fs.statfsSync(MEMORY_DIR) : null;
      if (stats) {
        totalSize = stats.bsize * stats.blocks;
        availableSize = stats.bsize * stats.bavail;
        if (availableSize < 100 * 1024 * 1024) {
          warnings.push('Less than 100MB free disk space');
        }
      }
    } catch {}

    if (!fs.existsSync(MEMORY_DIR)) {
      warnings.push('Memory directory does not exist');
    }

    return { healthy: warnings.length === 0, totalSize, availableSize, warnings };
  }

  // --- Defender check (stub — not needed in native mode) ---
  async isDefenderExcluded() { return true; }
  async addDefenderExclusion() { return { success: true }; }

  checkNeedsRestart() { return false; }
}

module.exports = { NativeManager, CEREBRO_DIR, MEMORY_DIR, LOG_DIR };
