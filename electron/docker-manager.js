const { spawn, execFile } = require('child_process');
const { EventEmitter } = require('events');
const path = require('path');
const fs = require('fs');
const os = require('os');
const crypto = require('crypto');
const http = require('http');
const https = require('https');

const CEREBRO_DIR = path.join(os.homedir(), '.cerebro');
const COMPOSE_FILE = path.join(CEREBRO_DIR, 'docker-compose.yml');
const ENV_FILE = path.join(CEREBRO_DIR, '.env');
const BACKEND_IMAGE = 'ghcr.io/professor-low/cerebro-backend';
const MEMORY_IMAGE = 'ghcr.io/professor-low/cerebro-memory';

class DockerManager extends EventEmitter {
  constructor() {
    super();
    this._running = false;
    this._ensureCerebroDir();
  }

  _ensureCerebroDir() {
    fs.mkdirSync(CEREBRO_DIR, { recursive: true });
  }

  /**
   * Run a command safely using execFile (no shell injection risk) and return stdout.
   */
  _run(cmd, args, options = {}) {
    return new Promise((resolve, reject) => {
      execFile(cmd, args, {
        timeout: options.timeout || 30000,
        maxBuffer: 1024 * 1024,
        ...options,
      }, (err, stdout, stderr) => {
        if (err) {
          reject(new Error(`${cmd} failed: ${stderr || err.message}`));
          return;
        }
        resolve(stdout.trim());
      });
    });
  }

  /**
   * Run a command with live output via callback.
   */
  _spawnWithOutput(cmd, args, onData) {
    return new Promise((resolve, reject) => {
      const proc = spawn(cmd, args, {
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (data) => {
        const line = data.toString();
        stdout += line;
        if (onData) onData(line);
      });

      proc.stderr.on('data', (data) => {
        const line = data.toString();
        stderr += line;
        if (onData) onData(line);
      });

      proc.on('close', (code) => {
        if (code === 0) {
          resolve(stdout.trim());
        } else {
          reject(new Error(`Exit code ${code}: ${stderr.slice(0, 500)}`));
        }
      });

      proc.on('error', reject);
    });
  }

  /**
   * Check if Docker CLI is installed.
   */
  async isDockerInstalled() {
    try {
      await this._run('docker', ['--version']);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Check if Docker daemon is running.
   */
  async isDockerRunning() {
    try {
      await this._run('docker', ['info'], { timeout: 10000 });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Check if Claude Code CLI is installed.
   */
  async isClaudeInstalled() {
    try {
      const result = await this._run('claude', ['--version'], { timeout: 5000 });
      return { installed: true, version: result };
    } catch {
      return { installed: false };
    }
  }

  /**
   * Get the path to the Claude CLI binary.
   */
  async getClaudeCliPath() {
    const isWin = process.platform === 'win32';
    try {
      const result = await this._run(isWin ? 'where' : 'which', ['claude'], { timeout: 5000 });
      return result.split('\n')[0].trim();
    } catch {
      const fallbacks = isWin
        ? [path.join(os.homedir(), '.claude', 'local', 'claude.exe'), 'C:\\Program Files\\Claude\\claude.exe']
        : ['/usr/local/bin/claude', path.join(os.homedir(), '.claude', 'local', 'claude')];
      for (const p of fallbacks) {
        if (fs.existsSync(p)) return p;
      }
      return null;
    }
  }

  /**
   * Write docker-compose.yml to ~/.cerebro/
   */
  async writeComposeFile() {
    const composeSrc = path.join(__dirname, '..', 'docker', 'docker-compose.yml');

    let content;
    if (fs.existsSync(composeSrc)) {
      content = fs.readFileSync(composeSrc, 'utf-8');
    } else {
      content = this._getDefaultComposeContent();
    }

    fs.writeFileSync(COMPOSE_FILE, content);
    console.log(`[Docker] Wrote docker-compose.yml to ${COMPOSE_FILE}`);
  }

  /**
   * Generate and store CEREBRO_SECRET + env vars.
   */
  writeEnvFile() {
    let existingEnv = {};
    if (fs.existsSync(ENV_FILE)) {
      const lines = fs.readFileSync(ENV_FILE, 'utf-8').split('\n');
      for (const line of lines) {
        const match = line.match(/^([^=]+)=(.*)$/);
        if (match) existingEnv[match[1]] = match[2];
      }
    }

    if (!existingEnv.CEREBRO_SECRET) {
      existingEnv.CEREBRO_SECRET = crypto.randomBytes(32).toString('hex');
    }

    const claudeConfigPath = path.join(os.homedir(), '.claude');
    existingEnv.CLAUDE_CONFIG_PATH = claudeConfigPath;

    const envContent = Object.entries(existingEnv)
      .map(([k, v]) => `${k}=${v}`)
      .join('\n') + '\n';

    fs.writeFileSync(ENV_FILE, envContent);
    console.log(`[Docker] Wrote .env to ${ENV_FILE}`);
  }

  /**
   * Set the Claude CLI path in the env file.
   */
  async setClaudeCliPath() {
    const cliPath = await this.getClaudeCliPath();
    if (cliPath) {
      let existingEnv = {};
      if (fs.existsSync(ENV_FILE)) {
        const lines = fs.readFileSync(ENV_FILE, 'utf-8').split('\n');
        for (const line of lines) {
          const match = line.match(/^([^=]+)=(.*)$/);
          if (match) existingEnv[match[1]] = match[2];
        }
      }
      existingEnv.CLAUDE_CLI_PATH = cliPath;
      const envContent = Object.entries(existingEnv)
        .map(([k, v]) => `${k}=${v}`)
        .join('\n') + '\n';
      fs.writeFileSync(ENV_FILE, envContent);
    }
  }

  /**
   * Pull Docker images (for initial setup or updates).
   */
  async pullImages(onProgress) {
    if (onProgress) onProgress({ stage: 'pulling', message: 'Pulling Docker images...' });

    try {
      await this._spawnWithOutput(
        'docker',
        ['compose', '-f', COMPOSE_FILE, '--env-file', ENV_FILE, 'pull'],
        (line) => {
          if (onProgress) onProgress({ stage: 'pulling', message: line.trim() });
        }
      );
      if (onProgress) onProgress({ stage: 'done', message: 'Images pulled successfully' });
      return true;
    } catch (err) {
      if (onProgress) onProgress({ stage: 'error', message: err.message });
      throw err;
    }
  }

  /**
   * Start the Docker Compose stack.
   */
  async startStack() {
    this.emit('status', 'starting');

    try {
      await this._run('docker', [
        'compose', '-f', COMPOSE_FILE, '--env-file', ENV_FILE,
        'up', '-d', '--remove-orphans',
      ], { timeout: 60000 });

      await this._waitForBackend();

      this._running = true;
      this.emit('status', 'running');
      console.log('[Docker] Stack started');
      return true;
    } catch (err) {
      this.emit('status', 'error');
      throw err;
    }
  }

  /**
   * Stop the Docker Compose stack.
   */
  async stopStack() {
    this.emit('status', 'stopping');

    try {
      await this._run('docker', [
        'compose', '-f', COMPOSE_FILE, '--env-file', ENV_FILE,
        'down',
      ], { timeout: 30000 });

      this._running = false;
      this.emit('status', 'stopped');
      console.log('[Docker] Stack stopped');
    } catch (err) {
      console.error('[Docker] Error stopping stack:', err.message);
      this._running = false;
    }
  }

  /**
   * Get status of all containers.
   */
  async getStatus() {
    try {
      const output = await this._run('docker', [
        'compose', '-f', COMPOSE_FILE, '--env-file', ENV_FILE,
        'ps', '--format', 'json',
      ], { timeout: 10000 });

      if (!output) return [];

      return output.split('\n')
        .filter(line => line.trim())
        .map(line => {
          try { return JSON.parse(line); }
          catch { return null; }
        })
        .filter(Boolean);
    } catch {
      return [];
    }
  }

  /**
   * Check for newer images on ghcr.io.
   */
  async checkForUpdates() {
    try {
      const localBackend = await this._run('docker', [
        'inspect', '--format', '{{index .RepoDigests 0}}',
        `${BACKEND_IMAGE}:latest`,
      ]).catch(() => null);

      const remoteResult = await this._run('docker', [
        'manifest', 'inspect', `${BACKEND_IMAGE}:latest`,
      ], { timeout: 15000 }).catch(() => null);

      if (!localBackend || !remoteResult) {
        return { updateAvailable: false };
      }

      const localDigest = localBackend.split('@')[1] || '';
      const remoteDigest = remoteResult.match(/"digest":\s*"([^"]+)"/)?.[1] || '';

      return {
        updateAvailable: localDigest !== remoteDigest && remoteDigest !== '',
        currentDigest: localDigest.slice(0, 16),
        remoteDigest: remoteDigest.slice(0, 16),
      };
    } catch {
      return { updateAvailable: false };
    }
  }

  /**
   * Apply update: pull new images and restart stack.
   */
  async applyUpdate(onProgress) {
    await this.pullImages(onProgress);
    await this.stopStack();
    await this.startStack();
  }

  /**
   * Get recent logs from the stack.
   */
  async getLogs(lines = 50) {
    try {
      return await this._run('docker', [
        'compose', '-f', COMPOSE_FILE, '--env-file', ENV_FILE,
        'logs', '--tail', String(lines),
      ], { timeout: 10000 });
    } catch (err) {
      return `Error getting logs: ${err.message}`;
    }
  }

  /**
   * Wait for the backend to respond on port 59000.
   */
  _waitForBackend(maxAttempts = 60) {
    return new Promise((resolve, reject) => {
      let attempts = 0;

      const check = () => {
        if (attempts >= maxAttempts) {
          reject(new Error('Backend health check timed out after 30s'));
          return;
        }
        attempts++;

        const req = http.get('http://127.0.0.1:59000/health', (res) => {
          let data = '';
          res.on('data', (chunk) => { data += chunk; });
          res.on('end', () => {
            if (res.statusCode === 200) {
              resolve();
            } else {
              setTimeout(check, 500);
            }
          });
        });

        req.on('error', () => setTimeout(check, 500));
        req.setTimeout(2000, () => {
          req.destroy();
          setTimeout(check, 500);
        });
      };

      setTimeout(check, 2000);
    });
  }

  /**
   * Check if WSL2 is available (Windows only).
   * If present, Docker install won't require a restart.
   */
  async checkWslAvailable() {
    if (process.platform !== 'win32') return { available: true, platform: 'linux' };
    try {
      const output = await this._run('wsl', ['--version'], { timeout: 10000 });
      return { available: true, version: output.split('\n')[0] };
    } catch {
      return { available: false };
    }
  }

  /**
   * Download Docker Desktop installer with progress reporting.
   * Caches in temp dir if recent and valid (>100MB).
   */
  async downloadDockerInstaller(onProgress) {
    const isWin = process.platform === 'win32';
    if (!isWin) return { path: null, platform: 'linux' };

    const installerPath = path.join(os.tmpdir(), 'DockerDesktopInstaller.exe');

    // Cache: skip if file exists, recent (<24h), and valid size (>100MB)
    if (fs.existsSync(installerPath)) {
      const stat = fs.statSync(installerPath);
      const ageHours = (Date.now() - stat.mtimeMs) / (1000 * 60 * 60);
      if (ageHours < 24 && stat.size > 100 * 1024 * 1024) {
        if (onProgress) onProgress({ percent: 100, transferredMB: Math.round(stat.size / 1024 / 1024), totalMB: Math.round(stat.size / 1024 / 1024) });
        return { path: installerPath, cached: true };
      }
    }

    const url = 'https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe';
    return new Promise((resolve, reject) => {
      const file = fs.createWriteStream(installerPath);
      const request = https.get(url, (response) => {
        // Follow redirects
        if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
          file.close();
          fs.unlinkSync(installerPath);
          https.get(response.headers.location, (res) => {
            this._pipeDownload(res, installerPath, onProgress, resolve, reject);
          }).on('error', reject);
          return;
        }
        this._pipeDownload(response, installerPath, onProgress, resolve, reject);
      });
      request.on('error', (err) => {
        fs.unlink(installerPath, () => {});
        reject(err);
      });
      request.setTimeout(300000, () => {
        request.destroy();
        reject(new Error('Download timed out'));
      });
    });
  }

  _pipeDownload(response, filePath, onProgress, resolve, reject) {
    const totalBytes = parseInt(response.headers['content-length'], 10) || 0;
    let transferred = 0;
    const file = fs.createWriteStream(filePath);

    response.on('data', (chunk) => {
      transferred += chunk.length;
      file.write(chunk);
      if (onProgress) {
        onProgress({
          percent: totalBytes > 0 ? Math.round((transferred / totalBytes) * 100) : 0,
          transferredMB: Math.round(transferred / 1024 / 1024),
          totalMB: Math.round(totalBytes / 1024 / 1024),
        });
      }
    });

    response.on('end', () => {
      file.end(() => {
        resolve({ path: filePath, cached: false });
      });
    });

    response.on('error', (err) => {
      file.close();
      fs.unlink(filePath, () => {});
      reject(err);
    });
  }

  /**
   * Install Docker Desktop silently.
   * Windows: uses PowerShell Start-Process -Verb RunAs for elevation.
   * Linux: uses pkexec with get-docker.sh script.
   * Returns { success, needsRestart, error }
   */
  async installDocker(onProgress) {
    const isWin = process.platform === 'win32';

    if (isWin) {
      // 1. Check WSL2 availability
      if (onProgress) onProgress({ stage: 'checking', message: 'Checking WSL2...' });
      const wsl = await this.checkWslAvailable();

      // 2. Download installer
      if (onProgress) onProgress({ stage: 'downloading', message: 'Downloading Docker Desktop...' });
      const download = await this.downloadDockerInstaller((p) => {
        if (onProgress) onProgress({ stage: 'downloading', message: `Downloading... ${p.percent}%`, ...p });
      });

      // 3. Run installer with elevation via PowerShell
      if (onProgress) onProgress({ stage: 'installing', message: 'Installing Docker Desktop (UAC prompt)...' });
      try {
        await this._runElevated(
          download.path,
          ['install', '--quiet', '--accept-license', '--backend=wsl-2'],
          { timeout: 300000 }
        );
      } catch (err) {
        return { success: false, needsRestart: false, error: `Installation failed: ${err.message}` };
      }

      // 4. Return based on WSL2 availability
      const needsRestart = !wsl.available;
      if (onProgress) {
        onProgress({
          stage: needsRestart ? 'restart-required' : 'done',
          message: needsRestart ? 'Docker installed. Please restart your PC for WSL2, then reopen Cerebro.' : 'Docker Desktop installed successfully!',
        });
      }
      return { success: true, needsRestart };

    } else {
      // Linux: use get-docker.sh
      if (onProgress) onProgress({ stage: 'downloading', message: 'Downloading Docker install script...' });
      const scriptPath = path.join(os.tmpdir(), 'get-docker.sh');

      try {
        await new Promise((resolve, reject) => {
          const file = fs.createWriteStream(scriptPath);
          https.get('https://get.docker.com', (res) => {
            res.pipe(file);
            file.on('finish', () => file.close(resolve));
          }).on('error', reject);
        });

        if (onProgress) onProgress({ stage: 'installing', message: 'Installing Docker (sudo prompt)...' });
        const user = os.userInfo().username;
        await this._run('pkexec', ['sh', '-c', `sh ${scriptPath} && usermod -aG docker ${user}`], { timeout: 300000 });

        if (onProgress) onProgress({ stage: 'done', message: 'Docker installed successfully!' });
        return { success: true, needsRestart: false };
      } catch (err) {
        return { success: false, needsRestart: false, error: `Installation failed: ${err.message}` };
      }
    }
  }

  /**
   * Run an executable with elevation via PowerShell Start-Process -Verb RunAs (Windows).
   */
  _runElevated(exePath, args, options = {}) {
    return new Promise((resolve, reject) => {
      const argsStr = args.map(a => `'${a}'`).join(',');
      const psArgs = [
        '-NoProfile', '-Command',
        `Start-Process -FilePath '${exePath}' -ArgumentList ${argsStr} -Verb RunAs -Wait -PassThru | ForEach-Object { exit $_.ExitCode }`,
      ];

      execFile('powershell.exe', psArgs, {
        timeout: options.timeout || 300000,
        maxBuffer: 1024 * 1024,
      }, (err, stdout, stderr) => {
        if (err) {
          reject(new Error(stderr || err.message));
        } else {
          resolve(stdout.trim());
        }
      });
    });
  }

  /**
   * Start Docker daemon and poll until ready.
   * Windows: launches Docker Desktop.exe detached, polls docker info.
   * Linux: uses pkexec systemctl start docker, polls docker info.
   * Timeout: 120s Windows, 30s Linux.
   */
  async startDockerDaemon(onProgress) {
    // Check if already running
    const alreadyRunning = await this.isDockerRunning();
    if (alreadyRunning) {
      if (onProgress) onProgress({ message: 'Docker is already running' });
      return { success: true };
    }

    const isWin = process.platform === 'win32';
    const timeoutSec = isWin ? 120 : 30;

    if (isWin) {
      // Launch Docker Desktop detached
      const dockerDesktopPaths = [
        'C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe',
        path.join(os.homedir(), 'AppData', 'Local', 'Docker', 'Docker Desktop.exe'),
      ];
      let launched = false;
      for (const p of dockerDesktopPaths) {
        if (fs.existsSync(p)) {
          const proc = spawn(p, [], { detached: true, stdio: 'ignore' });
          proc.unref();
          launched = true;
          break;
        }
      }
      if (!launched) {
        return { success: false, error: 'Docker Desktop executable not found' };
      }
    } else {
      // Linux: start via systemctl
      try {
        await this._run('pkexec', ['systemctl', 'start', 'docker'], { timeout: 15000 });
      } catch (err) {
        return { success: false, error: `Failed to start Docker: ${err.message}` };
      }
    }

    // Poll docker info until ready
    const startTime = Date.now();
    while ((Date.now() - startTime) / 1000 < timeoutSec) {
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      if (onProgress) onProgress({ message: `Waiting for Docker daemon... (${elapsed}s)` });

      const running = await this.isDockerRunning();
      if (running) {
        if (onProgress) onProgress({ message: 'Docker is ready!' });
        return { success: true };
      }

      await new Promise(r => setTimeout(r, 2000));
    }

    return { success: false, error: `Docker daemon did not start within ${timeoutSec}s` };
  }

  isRunning() {
    return this._running;
  }

  isSetupComplete() {
    return fs.existsSync(COMPOSE_FILE) && fs.existsSync(ENV_FILE);
  }

  _getDefaultComposeContent() {
    return `services:
  redis:
    image: redis:7-alpine
    ports:
      - "16379:6379"
    volumes:
      - cerebro-redis:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  backend:
    image: ghcr.io/professor-low/cerebro-backend:latest
    ports:
      - "59000:59000"
    environment:
      REDIS_URL: redis://redis:6379/0
      AI_MEMORY_PATH: /data/memory
      CEREBRO_CORS_ORIGINS: "http://localhost:59000"
      CEREBRO_HOST: "0.0.0.0"
      CEREBRO_PORT: "59000"
      CEREBRO_STANDALONE: "1"
      CEREBRO_SECRET: "\${CEREBRO_SECRET}"
    volumes:
      - cerebro-data:/data/memory
      - "\${CLAUDE_CLI_PATH:-/usr/local/bin/claude}:/usr/local/bin/claude:ro"
      - "\${CLAUDE_CONFIG_PATH:-~/.claude}:/root/.claude:ro"
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:59000/health"]
      interval: 10s
      timeout: 5s
      start_period: 15s
      retries: 3

  memory:
    image: ghcr.io/professor-low/cerebro-memory:latest
    volumes:
      - cerebro-data:/data/memory
    environment:
      AI_MEMORY_PATH: /data/memory
    restart: unless-stopped

volumes:
  cerebro-redis:
  cerebro-data:
`;
  }
}

module.exports = { DockerManager };
