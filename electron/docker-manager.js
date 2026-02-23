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
const SETUP_STATE_FILE = path.join(CEREBRO_DIR, '.setup-state.json');
const FILE_ACCESS_CONFIG = path.join(CEREBRO_DIR, 'file-access.json');
const BACKEND_IMAGE = 'ghcr.io/professor-low/cerebro-backend';

class DockerManager extends EventEmitter {
  constructor() {
    super();
    this._running = false;
    this._dockerInstalledThisSession = false;
    this._needsRestart = false;
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
      // PATH lookup failed — check common install locations
    }
    const dockerPath = this._findDockerPath();
    if (dockerPath) {
      try {
        await this._run(dockerPath, ['--version']);
        return true;
      } catch {
        return true; // Binary exists even if --version fails
      }
    }
    return false;
  }

  /**
   * Check if Docker daemon is running.
   * Tries docker info first (needs socket access / docker group).
   * Falls back to systemctl on Linux (works without docker group).
   */
  async isDockerRunning() {
    const dockerCmd = this._dockerCmd();
    try {
      await this._run(dockerCmd, ['info'], { timeout: 10000 });
      return true;
    } catch {
      // docker info may fail if user not in docker group yet (needs re-login)
    }
    // Linux fallback: check systemd service status (no group needed)
    if (process.platform !== 'win32') {
      try {
        const result = await this._run('systemctl', ['is-active', 'docker'], { timeout: 5000 });
        return result.trim() === 'active';
      } catch {
        return false;
      }
    }
    return false;
  }

  /**
   * Find the Docker binary path (fallback when PATH doesn't include it).
   * Caches the result for subsequent calls.
   */
  _findDockerPath() {
    if (this._dockerPath !== undefined) return this._dockerPath;
    const candidates = process.platform === 'win32'
      ? ['C:\\Program Files\\Docker\\Docker\\resources\\bin\\docker.exe']
      : ['/usr/bin/docker', '/usr/local/bin/docker'];
    for (const p of candidates) {
      if (fs.existsSync(p)) {
        this._dockerPath = p;
        return p;
      }
    }
    this._dockerPath = null;
    return null;
  }

  /**
   * Get the docker command — resolved path or just 'docker' for PATH lookup.
   */
  _dockerCmd() {
    return this._findDockerPath() || 'docker';
  }

  /**
   * Check if Claude Code CLI is installed.
   */
  async isClaudeInstalled() {
    // Try PATH first
    try {
      const result = await this._run('claude', ['--version'], { timeout: 5000 });
      return { installed: true, version: result };
    } catch {
      // PATH lookup failed — check common install locations
    }

    const claudePath = await this.getClaudeCliPath();
    if (claudePath) {
      try {
        const result = await this._run(claudePath, ['--version'], { timeout: 5000 });
        return { installed: true, version: result, path: claudePath };
      } catch {
        // Binary exists but --version failed, still count as installed
        return { installed: true, path: claudePath };
      }
    }

    return { installed: false };
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
        ? [
            path.join(os.homedir(), '.claude', 'local', 'claude.exe'),
            'C:\\Program Files\\Claude\\claude.exe',
          ]
        : [
            path.join(os.homedir(), '.local', 'bin', 'claude'),
            path.join(os.homedir(), '.claude', 'local', 'claude'),
            '/usr/local/bin/claude',
          ];
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

    content = this._injectFileMounts(content);
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
    existingEnv.CEREBRO_DIR = CEREBRO_DIR;

    // Create a clean Claude config for the container (no hooks from host)
    this._createClaudeConfig(claudeConfigPath);

    // Remove legacy env vars that are no longer used (CLI is baked into Docker image)
    delete existingEnv.CLAUDE_CLI_PATH;
    delete existingEnv.CLAUDE_CONFIG_PATH;

    // Point CLAUDE_CONFIG_DIR to the clean copy so compose mounts it
    const cleanConfigDir = path.join(CEREBRO_DIR, 'claude-config');
    existingEnv.CLAUDE_CONFIG_DIR = cleanConfigDir;

    // Copy frontend from the app bundle to a real directory Docker can mount
    const frontendDest = path.join(CEREBRO_DIR, 'frontend');
    this._syncFrontend(frontendDest);
    existingEnv.CEREBRO_FRONTEND_DIR = frontendDest;

    const envContent = Object.entries(existingEnv)
      .map(([k, v]) => `${k}=${v}`)
      .join('\n') + '\n';

    fs.writeFileSync(ENV_FILE, envContent);
    console.log(`[Docker] Wrote .env to ${ENV_FILE}`);
  }

  /**
   * Copy frontend files from the app bundle (possibly inside asar) to a real
   * directory that Docker can mount as a volume.
   */
  _syncFrontend(destDir) {
    const srcDir = path.join(__dirname, '..', 'frontend');
    try {
      if (!fs.existsSync(srcDir)) {
        console.warn('[Docker] Frontend source not found, skipping sync');
        return;
      }
      this._copyDirRecursive(srcDir, destDir);
      console.log(`[Docker] Synced frontend to ${destDir}`);
    } catch (err) {
      console.warn(`[Docker] Frontend sync failed: ${err.message}`);
    }
  }

  /**
   * Recursively copy a directory and all its contents.
   */
  _copyDirRecursive(src, dest) {
    fs.mkdirSync(dest, { recursive: true });
    const entries = fs.readdirSync(src);
    for (const entry of entries) {
      const srcPath = path.join(src, entry);
      const destPath = path.join(dest, entry);
      const stat = fs.statSync(srcPath);
      if (stat.isDirectory()) {
        this._copyDirRecursive(srcPath, destPath);
      } else {
        fs.copyFileSync(srcPath, destPath);
      }
    }
  }

  /**
   * Create a clean copy of Claude config for the Docker container.
   * Strips hooks (they reference host paths) and copies credentials.
   */
  _createClaudeConfig(sourceConfigDir) {
    const destDir = path.join(CEREBRO_DIR, 'claude-config');
    try {
      fs.mkdirSync(destDir, { recursive: true });

      // Copy credentials if they exist
      const credSrc = path.join(sourceConfigDir, '.credentials.json');
      if (fs.existsSync(credSrc)) {
        fs.copyFileSync(credSrc, path.join(destDir, '.credentials.json'));
      }

      // Copy settings without hooks or host-specific config
      const settingsSrc = path.join(sourceConfigDir, 'settings.json');
      if (fs.existsSync(settingsSrc)) {
        const settings = JSON.parse(fs.readFileSync(settingsSrc, 'utf-8'));
        delete settings.hooks;
        delete settings.mcpServers;  // MCP server configs reference host paths
        fs.writeFileSync(
          path.join(destDir, 'settings.json'),
          JSON.stringify(settings, null, 2)
        );
      }

      // Create empty subdirs needed by Claude CLI
      for (const sub of ['cache', 'debug', 'plugins', 'projects', 'todos', 'downloads']) {
        fs.mkdirSync(path.join(destDir, sub), { recursive: true });
      }

      // Remove host-specific files that should never leak into standalone containers
      const junkFiles = [
        'CLAUDE.md', 'mcp.json', 'statusline.sh', 'history.jsonl',
      ];
      for (const f of junkFiles) {
        const fp = path.join(destDir, f);
        if (fs.existsSync(fp)) {
          fs.unlinkSync(fp);
          console.log(`[Docker] Removed leaked host file: ${f}`);
        }
      }
      const junkDirs = ['hooks', 'session-env'];
      for (const d of junkDirs) {
        const dp = path.join(destDir, d);
        if (fs.existsSync(dp)) {
          fs.rmSync(dp, { recursive: true, force: true });
          console.log(`[Docker] Removed leaked host dir: ${d}`);
        }
      }

      // Write standalone CLAUDE.md for agent instructions inside container
      const standaloneMdSrc = path.join(__dirname, '..', 'docker', 'standalone-claude.md');
      if (fs.existsSync(standaloneMdSrc)) {
        fs.copyFileSync(standaloneMdSrc, path.join(destDir, 'CLAUDE.md'));
        console.log(`[Docker] Injected standalone CLAUDE.md for agents`);
      }

      console.log(`[Docker] Created clean Claude config at ${destDir}`);
    } catch (err) {
      console.warn(`[Docker] Failed to create Claude config: ${err.message}`);
    }
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
        this._dockerCmd(),
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

    // Always refresh Claude credentials from host before starting
    const credResult = this.refreshClaudeCredentials();
    if (!credResult.valid) {
      console.warn(`[Docker] Claude credentials issue: ${credResult.error}`);
      this.emit('credentials-expired', { message: credResult.error, needsLogin: true });
    }

    try {
      await this._run(this._dockerCmd(), [
        'compose', '-f', COMPOSE_FILE, '--env-file', ENV_FILE,
        'up', '-d', '--remove-orphans',
      ], { timeout: 120000 });

      await this._waitForBackend(240); // 120s timeout

      this._running = true;
      this.emit('status', 'running');
      this.startCredentialWatch();
      console.log('[Docker] Stack started');
      return true;
    } catch (err) {
      this.emit('status', 'error');
      // Append container logs to help diagnose
      try {
        const logs = await this.getLogs(30);
        if (logs && !logs.startsWith('Error')) {
          err.message += '\n\nContainer logs:\n' + logs;
        }
      } catch { /* ignore log fetch errors */ }
      throw err;
    }
  }

  /**
   * Stop the Docker Compose stack.
   */
  async stopStack() {
    this.emit('status', 'stopping');
    this.stopCredentialWatch();

    try {
      await this._run(this._dockerCmd(), [
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
      const output = await this._run(this._dockerCmd(), [
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
      const localBackend = await this._run(this._dockerCmd(), [
        'inspect', '--format', '{{index .RepoDigests 0}}',
        `${BACKEND_IMAGE}:latest`,
      ]).catch(() => null);

      const remoteResult = await this._run(this._dockerCmd(), [
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
      return await this._run(this._dockerCmd(), [
        'compose', '-f', COMPOSE_FILE, '--env-file', ENV_FILE,
        'logs', '--tail', String(lines),
      ], { timeout: 10000 });
    } catch (err) {
      return `Error getting logs: ${err.message}`;
    }
  }

  /**
   * Wait for the backend to respond on port 61000.
   */
  _waitForBackend(maxAttempts = 240) {
    const timeoutSec = Math.round(maxAttempts * 0.5);
    return new Promise((resolve, reject) => {
      let attempts = 0;

      const check = () => {
        if (attempts >= maxAttempts) {
          reject(new Error(`Backend health check timed out after ${timeoutSec}s`));
          return;
        }
        attempts++;

        const req = http.get('http://127.0.0.1:61000/health', (res) => {
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
      const request = https.get(url, (response) => {
        // Follow redirects
        if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
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

    // Wait for 'close' event — this fires after the OS file handle is fully released,
    // preventing "file in use" errors when Start-Process tries to execute the installer.
    file.on('close', () => {
      resolve({ path: filePath, cached: false });
    });

    response.on('end', () => {
      file.end();
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
   * Linux: uses _runAsRoot (pkexec → terminal fallback) with get-docker.sh.
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
      this._dockerInstalledThisSession = true;
      this._needsRestart = needsRestart;
      if (onProgress) {
        onProgress({
          stage: needsRestart ? 'restart-required' : 'done',
          message: needsRestart ? 'Docker installed. Please restart your PC for WSL2, then reopen Cerebro.' : 'Docker Desktop installed successfully!',
        });
      }
      return { success: true, needsRestart };

    } else {
      // Linux: detect distro and install Docker appropriately
      const user = os.userInfo().username;
      const wrapperPath = path.join(os.tmpdir(), 'cerebro-install-docker.sh');
      // Use ~/.cerebro/ for markers — /tmp has sticky bit so only root can unlink root-owned files
      const markerPath = path.join(CEREBRO_DIR, '.docker-install-done');
      try { fs.unlinkSync(markerPath); } catch {}

      try {
        const distro = this._detectLinuxDistro();

        if (distro === 'arch') {
          // Arch-based (Arch, Manjaro, EndeavourOS): use pacman
          if (onProgress) onProgress({ stage: 'installing', message: 'Installing Docker via pacman (password required)...' });

          fs.writeFileSync(wrapperPath, [
            '#!/bin/sh',
            // Sync package database first — stale DB causes 404s on mirrors
            // docker-compose provides the 'docker compose' V2 plugin
            'pacman -Sy --noconfirm docker docker-compose',
            'systemctl enable docker',
            'systemctl start docker',
            `usermod -aG docker "${user}"`,
            'exit $?',
          ].join('\n'));
          fs.chmodSync(wrapperPath, '755');
        } else {
          // Debian/Ubuntu/Fedora/RHEL/etc: use get-docker.sh
          if (onProgress) onProgress({ stage: 'downloading', message: 'Downloading Docker install script...' });
          const getDockerScript = path.join(os.tmpdir(), 'get-docker.sh');

          await new Promise((resolve, reject) => {
            const file = fs.createWriteStream(getDockerScript);
            https.get('https://get.docker.com', (res) => {
              if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
                file.close();
                https.get(res.headers.location, (r) => {
                  r.pipe(file);
                  file.on('finish', () => file.close(resolve));
                }).on('error', reject);
                return;
              }
              res.pipe(file);
              file.on('finish', () => file.close(resolve));
            }).on('error', reject);
          });

          if (onProgress) onProgress({ stage: 'installing', message: 'Installing Docker (password required)...' });

          fs.writeFileSync(wrapperPath, [
            '#!/bin/sh',
            `sh "${getDockerScript}"`,
            `usermod -aG docker "${user}"`,
            'exit $?',
          ].join('\n'));
          fs.chmodSync(wrapperPath, '755');
        }

        await this._runAsRoot(wrapperPath, [], { timeout: 300000, markerPath });

        this._dockerInstalledThisSession = true;
        this._needsRestart = true; // docker group needs re-login
        if (onProgress) onProgress({ stage: 'done', message: 'Docker installed! A restart is needed for group changes.' });
        return { success: true, needsRestart: true };
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
      // Linux: start via systemctl with root elevation
      try {
        const startScript = path.join(os.tmpdir(), 'cerebro-start-docker.sh');
        const markerPath = path.join(CEREBRO_DIR, '.docker-start-done');
        try { fs.unlinkSync(markerPath); } catch {}
        fs.writeFileSync(startScript, [
          '#!/bin/sh',
          'systemctl start docker',
          'exit $?',
        ].join('\n'));
        fs.chmodSync(startScript, '755');
        await this._runAsRoot(startScript, [], { timeout: 30000, markerPath });
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

  /**
   * Run a script as root on Linux.
   * Tries pkexec first (works if graphical polkit agent is running).
   * Falls back to spawning a terminal emulator with sudo for password entry.
   */
  async _runAsRoot(scriptPath, args, options = {}) {
    // Try pkexec first
    try {
      await this._run('pkexec', [scriptPath, ...args], {
        timeout: options.timeout || 300000,
      });
      return;
    } catch (err) {
      // If pkexec fails due to no agent/tty, try terminal fallback
      const msg = err.message || '';
      if (msg.includes('textual authentication agent') || msg.includes('/dev/tty') || msg.includes('No such device')) {
        console.log('[Docker] pkexec failed (no polkit agent), falling back to terminal');
      } else {
        throw err;
      }
    }

    // Fallback: spawn a visible terminal with sudo
    await this._runInTerminal(scriptPath, options);
  }

  /**
   * Run a script in a visible terminal window with sudo.
   * Polls for a marker file to detect completion.
   */
  _runInTerminal(scriptPath, options = {}) {
    return new Promise(async (resolve, reject) => {
      const markerPath = options.markerPath || path.join(CEREBRO_DIR, '.root-done');
      try { fs.unlinkSync(markerPath); } catch {}

      // Wrap in sudo with marker file — marker is in ~/.cerebro/ (user-writable, no sticky bit)
      const wrapperPath = path.join(os.tmpdir(), 'cerebro-terminal-wrapper.sh');
      fs.writeFileSync(wrapperPath, [
        '#!/bin/sh',
        `sudo "${scriptPath}"`,
        'RESULT=$?',
        `echo $RESULT > "${markerPath}"`,
        'echo ""',
        'if [ $RESULT -eq 0 ]; then echo "Done! This window will close in 3 seconds..."; else echo "Failed (exit $RESULT). This window will close in 5 seconds..."; fi',
        'sleep 3',
        'exit $RESULT',
      ].join('\n'));
      fs.chmodSync(wrapperPath, '755');

      // Find an available terminal emulator
      const terminal = await this._findTerminal();
      if (!terminal) {
        reject(new Error('No terminal emulator found. Please install Docker manually: curl -fsSL https://get.docker.com | sudo sh'));
        return;
      }

      console.log(`[Docker] Using terminal: ${terminal.cmd}`);
      const proc = spawn(terminal.cmd, [...terminal.args, wrapperPath], {
        detached: true,
        stdio: 'ignore',
      });
      proc.unref();

      // Poll for the marker file
      const timeout = options.timeout || 300000;
      const startTime = Date.now();
      const poll = setInterval(() => {
        if (Date.now() - startTime > timeout) {
          clearInterval(poll);
          reject(new Error('Timed out waiting for installation to complete'));
          return;
        }
        if (fs.existsSync(markerPath)) {
          clearInterval(poll);
          try {
            const result = fs.readFileSync(markerPath, 'utf-8').trim();
            fs.unlinkSync(markerPath);
            if (result === '0') {
              resolve();
            } else {
              reject(new Error(`Script exited with code ${result}`));
            }
          } catch (readErr) {
            reject(readErr);
          }
        }
      }, 1000);

      proc.on('error', (err) => {
        clearInterval(poll);
        reject(err);
      });
    });
  }

  /**
   * Detect the Linux distribution family.
   * Returns 'arch', 'debian', 'fedora', or 'unknown'.
   */
  _detectLinuxDistro() {
    try {
      const osRelease = fs.readFileSync('/etc/os-release', 'utf-8');
      if (/^ID=arch$/m.test(osRelease) || /^ID_LIKE=.*arch/m.test(osRelease)) return 'arch';
      if (/^ID_LIKE=.*debian/m.test(osRelease) || /^ID=ubuntu$/m.test(osRelease) || /^ID=debian$/m.test(osRelease)) return 'debian';
      if (/^ID_LIKE=.*fedora/m.test(osRelease) || /^ID=fedora$/m.test(osRelease)) return 'fedora';
    } catch {}
    // Fallback: check for pacman
    try {
      fs.accessSync('/usr/bin/pacman', fs.constants.X_OK);
      return 'arch';
    } catch {}
    return 'unknown';
  }

  /**
   * Find an available terminal emulator on Linux.
   * Returns { cmd, args } where args come before the script path.
   */
  async _findTerminal() {
    const terminals = [
      { cmd: 'xterm', args: ['-T', 'Cerebro - Docker Setup', '-e'] },
      { cmd: 'gnome-terminal', args: ['--title=Cerebro - Docker Setup', '--wait', '--'] },
      { cmd: 'konsole', args: ['--noclose', '-e'] },
      { cmd: 'xfce4-terminal', args: ['--title=Cerebro - Docker Setup', '-e'] },
      { cmd: 'mate-terminal', args: ['--title=Cerebro - Docker Setup', '-e'] },
      { cmd: 'lxterminal', args: ['--title=Cerebro - Docker Setup', '-e'] },
      { cmd: 'alacritty', args: ['--title', 'Cerebro - Docker Setup', '-e'] },
      { cmd: 'kitty', args: ['--title', 'Cerebro - Docker Setup'] },
      { cmd: 'x-terminal-emulator', args: ['-e'] },
    ];

    for (const t of terminals) {
      try {
        await this._run('which', [t.cmd], { timeout: 2000 });
        return t;
      } catch {
        continue;
      }
    }
    return null;
  }

  /**
   * Check if Claude CLI credentials exist and are still valid.
   * Returns { valid, expiresIn, error, needsLogin }
   */
  checkClaudeCredentials() {
    const containerCreds = path.join(CEREBRO_DIR, 'claude-config', '.credentials.json');
    const hostCreds = path.join(os.homedir(), '.claude', '.credentials.json');

    // Check container credentials first
    const credsPath = fs.existsSync(containerCreds) ? containerCreds : hostCreds;

    if (!fs.existsSync(credsPath)) {
      return { valid: false, error: 'No Claude credentials found', needsLogin: true };
    }

    try {
      const data = JSON.parse(fs.readFileSync(credsPath, 'utf-8'));
      const oauth = data.claudeAiOauth;
      if (!oauth || !oauth.accessToken) {
        return { valid: false, error: 'No access token in credentials', needsLogin: true };
      }

      const expiresAt = oauth.expiresAt || 0;
      const now = Date.now();
      const expiresInMs = expiresAt - now;
      const expiresInMinutes = Math.round(expiresInMs / 60000);

      if (expiresInMs <= 0) {
        return { valid: false, expiresIn: expiresInMinutes, error: 'Token expired', needsLogin: true };
      }

      // Warn if expiring within 30 minutes
      if (expiresInMinutes < 30) {
        return { valid: true, expiresIn: expiresInMinutes, warning: 'Token expiring soon' };
      }

      return { valid: true, expiresIn: expiresInMinutes };
    } catch (err) {
      return { valid: false, error: `Failed to read credentials: ${err.message}`, needsLogin: true };
    }
  }

  /**
   * Refresh container Claude credentials from host.
   * Call this before starting agents or on a timer.
   * Returns { refreshed, valid, error }
   */
  refreshClaudeCredentials() {
    const hostCreds = path.join(os.homedir(), '.claude', '.credentials.json');
    const destDir = path.join(CEREBRO_DIR, 'claude-config');
    const destCreds = path.join(destDir, '.credentials.json');

    if (!fs.existsSync(hostCreds)) {
      return { refreshed: false, valid: false, error: 'No host credentials found — run: claude auth login' };
    }

    try {
      // Check if host credentials are valid
      const data = JSON.parse(fs.readFileSync(hostCreds, 'utf-8'));
      const oauth = data.claudeAiOauth;
      const expiresAt = (oauth && oauth.expiresAt) || 0;
      const now = Date.now();

      if (expiresAt <= now) {
        return { refreshed: false, valid: false, error: 'Host credentials also expired — run: claude auth login' };
      }

      // Copy fresh credentials to container config
      fs.mkdirSync(destDir, { recursive: true });
      fs.copyFileSync(hostCreds, destCreds);
      console.log(`[Docker] Refreshed Claude credentials (expires in ${Math.round((expiresAt - now) / 60000)}m)`);

      return { refreshed: true, valid: true, expiresIn: Math.round((expiresAt - now) / 60000) };
    } catch (err) {
      return { refreshed: false, valid: false, error: `Credential refresh failed: ${err.message}` };
    }
  }

  /**
   * Silently refresh OAuth tokens using the refresh token.
   * No user interaction required. Returns a promise.
   */
  silentRefreshOAuthToken() {
    const OAUTH_TOKEN_URL = 'https://console.anthropic.com/v1/oauth/token';
    const CLAUDE_CLIENT_ID = '9d1c250a-e61b-44d9-88ed-5944d1962f5e';

    const hostCreds = path.join(os.homedir(), '.claude', '.credentials.json');
    if (!fs.existsSync(hostCreds)) {
      return Promise.resolve({ success: false, error: 'No credentials file found' });
    }

    let data;
    try {
      data = JSON.parse(fs.readFileSync(hostCreds, 'utf-8'));
    } catch (err) {
      return Promise.resolve({ success: false, error: `Failed to read credentials: ${err.message}` });
    }

    const oauth = data.claudeAiOauth;
    if (!oauth || !oauth.refreshToken) {
      return Promise.resolve({ success: false, error: 'No refresh token available' });
    }

    const body = JSON.stringify({
      grant_type: 'refresh_token',
      refresh_token: oauth.refreshToken,
      client_id: CLAUDE_CLIENT_ID,
    });

    return new Promise((resolve) => {
      const url = new URL(OAUTH_TOKEN_URL);
      const req = https.request({
        hostname: url.hostname,
        path: url.pathname,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
        timeout: 15000,
      }, (res) => {
        let responseData = '';
        res.on('data', (chunk) => { responseData += chunk; });
        res.on('end', () => {
          try {
            if (res.statusCode !== 200) {
              console.error(`[Docker] OAuth refresh failed (${res.statusCode}):`, responseData);
              resolve({ success: false, error: `OAuth refresh returned ${res.statusCode}` });
              return;
            }

            const tokens = JSON.parse(responseData);
            if (!tokens.access_token) {
              resolve({ success: false, error: 'No access_token in response' });
              return;
            }

            // Update credentials file — preserve all other fields (mcpOAuth, etc.)
            const expiresAt = tokens.expires_in
              ? Date.now() + (tokens.expires_in * 1000)
              : Date.now() + (3600 * 1000); // fallback 1 hour

            data.claudeAiOauth.accessToken = tokens.access_token;
            data.claudeAiOauth.expiresAt = expiresAt;
            if (tokens.refresh_token) {
              data.claudeAiOauth.refreshToken = tokens.refresh_token;
            }

            fs.writeFileSync(hostCreds, JSON.stringify(data, null, 2), 'utf-8');
            const expiresInMin = Math.round((expiresAt - Date.now()) / 60000);
            console.log(`[Docker] Silent OAuth refresh successful (expires in ${expiresInMin}m)`);

            // Also copy to container
            this.refreshClaudeCredentials();

            resolve({ success: true, expiresIn: expiresInMin });
          } catch (err) {
            console.error('[Docker] OAuth refresh parse error:', err.message);
            resolve({ success: false, error: `Parse error: ${err.message}` });
          }
        });
      });

      req.on('error', (err) => {
        console.error('[Docker] OAuth refresh network error:', err.message);
        resolve({ success: false, error: `Network error: ${err.message}` });
      });

      req.on('timeout', () => {
        req.destroy();
        resolve({ success: false, error: 'Request timed out' });
      });

      req.write(body);
      req.end();
    });
  }

  /**
   * Check if any agents are currently running via the backend API.
   * Returns a promise that resolves to true if agents are active.
   */
  _hasRunningAgents() {
    return new Promise((resolve) => {
      const req = http.get('http://localhost:61000/agents', { timeout: 3000 }, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try {
            const parsed = JSON.parse(data);
            const agents = parsed.agents || [];
            const running = agents.some(a => a.status === 'running' || a.status === 'queued');
            resolve(running);
          } catch (_) { resolve(false); }
        });
      });
      req.on('error', () => resolve(false));
      req.on('timeout', () => { req.destroy(); resolve(false); });
    });
  }

  /**
   * Start a periodic credential refresh timer.
   * Checks every 15 minutes, attempts silent OAuth refresh before prompting user.
   * Skips silent refresh when agents are running to avoid invalidating their tokens.
   */
  startCredentialWatch() {
    if (this._credWatchInterval) return;

    this._credWatchInterval = setInterval(async () => {
      const status = this.checkClaudeCredentials();
      if (!status.valid || (status.expiresIn && status.expiresIn < 30)) {
        // Check if agents are running — if so, let the CLI handle its own refresh
        const agentsRunning = await this._hasRunningAgents();
        if (agentsRunning && status.valid) {
          // Token is expiring soon but still valid, and agents are running.
          // The CLI processes will refresh their own tokens. Don't compete.
          console.log('[Docker] Credentials expiring soon but agents are running — skipping silent refresh to avoid race condition');
          return;
        }

        if (agentsRunning && !status.valid) {
          // Token is fully expired AND agents are running — they're likely already failing.
          // Still skip silent refresh to not make it worse. Emit expired event so user knows.
          console.log('[Docker] Credentials expired with agents running — notifying user');
          this.emit('credentials-expired', {
            message: 'Token expired while agents were running. Agents may need to be restarted after re-authentication.',
            needsLogin: true,
          });
          return;
        }

        // No agents running — safe to do silent refresh
        console.log('[Docker] Credentials ' + (status.valid ? 'expiring soon' : 'expired') + ', attempting silent refresh...');

        const silentResult = await this.silentRefreshOAuthToken();
        if (silentResult.success) {
          this.emit('credentials-refreshed', {
            expiresIn: silentResult.expiresIn,
            message: 'Tokens refreshed automatically',
            silent: true,
          });
          return;
        }

        console.log(`[Docker] Silent refresh failed: ${silentResult.error}, trying host copy...`);

        // Fallback: try copying from host
        const result = this.refreshClaudeCredentials();
        if (!result.valid) {
          this.emit('credentials-expired', {
            message: result.error,
            needsLogin: true,
          });
        } else {
          this.emit('credentials-refreshed', { expiresIn: result.expiresIn });
        }
      }
    }, 15 * 60 * 1000); // every 15 minutes

    console.log('[Docker] Credential watch started (15min interval)');
  }

  stopCredentialWatch() {
    if (this._credWatchInterval) {
      clearInterval(this._credWatchInterval);
      this._credWatchInterval = null;
    }
  }

  isRunning() {
    return this._running;
  }

  isSetupComplete() {
    return fs.existsSync(COMPOSE_FILE) && fs.existsSync(ENV_FILE);
  }

  wasDockerInstalledThisSession() {
    return this._dockerInstalledThisSession;
  }

  needsRestart() {
    return this._needsRestart;
  }

  /**
   * Async check: does the user need a restart?
   * True if Docker was installed this session, OR if Docker service is running
   * but the user can't access the socket (docker group not active yet).
   */
  async checkNeedsRestart() {
    if (this._needsRestart) return true;

    // On Linux, check if Docker service runs but user lacks socket access
    if (process.platform !== 'win32') {
      const dockerCmd = this._dockerCmd();
      let socketWorks = false;
      let serviceRunning = false;

      try {
        await this._run(dockerCmd, ['info'], { timeout: 5000 });
        socketWorks = true;
      } catch {}

      if (!socketWorks) {
        try {
          const result = await this._run('systemctl', ['is-active', 'docker'], { timeout: 5000 });
          serviceRunning = result.trim() === 'active';
        } catch {}
      }

      // Service runs but socket denied → needs re-login
      if (serviceRunning && !socketWorks) return true;
    }

    return false;
  }

  saveSetupState(state) {
    this._ensureCerebroDir();
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
    try {
      if (fs.existsSync(SETUP_STATE_FILE)) {
        fs.unlinkSync(SETUP_STATE_FILE);
      }
    } catch {}
  }

  loadFileAccessConfig() {
    try {
      if (fs.existsSync(FILE_ACCESS_CONFIG)) {
        return JSON.parse(fs.readFileSync(FILE_ACCESS_CONFIG, 'utf-8'));
      }
    } catch {}
    return { fileMounts: [] };
  }

  saveFileAccessConfig(config) {
    this._ensureCerebroDir();
    fs.writeFileSync(FILE_ACCESS_CONFIG, JSON.stringify(config, null, 2));
  }

  getPresetMounts() {
    const homeDir = os.homedir();
    const isWin = process.platform === 'win32';
    const presets = [
      { id: 'desktop', label: 'Desktop', folder: 'Desktop' },
      { id: 'documents', label: 'Documents', folder: 'Documents' },
      { id: 'downloads', label: 'Downloads', folder: 'Downloads' },
    ];
    return presets.map(p => ({
      id: p.id,
      label: p.label,
      hostPath: isWin ? path.join(homeDir, p.folder) : path.join(homeDir, p.folder),
      containerPath: `/mounts/${p.id}`,
      readOnly: true,
      preset: true,
    }));
  }

  _injectFileMounts(composeContent) {
    const config = this.loadFileAccessConfig();
    if (!config.fileMounts || config.fileMounts.length === 0) return composeContent;

    const anchor = '${CLAUDE_CONFIG_DIR:-~/.claude}:/home/cerebro/.claude';
    const anchorIdx = composeContent.indexOf(anchor);
    if (anchorIdx === -1) {
      console.warn('[Docker] File mounts anchor line not found in compose content');
      return composeContent;
    }

    // Find end of the anchor line
    const endOfLine = composeContent.indexOf('\n', anchorIdx);
    if (endOfLine === -1) return composeContent;

    const mountLines = config.fileMounts.map(m => {
      const suffix = m.readOnly ? ':ro' : '';
      // Convert Windows backslashes to forward slashes — YAML double-quoted
      // strings interpret \U, \t etc. as escape sequences, breaking the parse.
      const hostPath = m.hostPath.replace(/\\/g, '/');
      return `      - "${hostPath}:${m.containerPath}${suffix}"`;
    }).join('\n');

    return composeContent.slice(0, endOfLine + 1) + mountLines + '\n' + composeContent.slice(endOfLine + 1);
  }

  _getDefaultComposeContent() {
    return `services:
  redis:
    image: redis:7-alpine
    ports:
      - "127.0.0.1:16379:6379"
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
      - "127.0.0.1:61000:59000"
    environment:
      REDIS_URL: redis://redis:6379/0
      AI_MEMORY_PATH: /data/memory
      CEREBRO_CORS_ORIGINS: "http://localhost:61000"
      CEREBRO_HOST: "0.0.0.0"
      CEREBRO_PORT: "59000"
      CEREBRO_STANDALONE: "1"
      CEREBRO_DEVICE: "\${CEREBRO_DEVICE:-standalone}"
      CEREBRO_NAS_IP: ""
      CEREBRO_MCP_SRC: "/app/mcp_modules"
      CEREBRO_SECRET: "\${CEREBRO_SECRET}"
      HOME: /home/cerebro
    tmpfs:
      - /home/cerebro:uid=1000,gid=1000
    volumes:
      - cerebro-data:/data/memory
      - "\${CLAUDE_CONFIG_DIR:-~/.claude}:/home/cerebro/.claude"
      - "\${CEREBRO_FRONTEND_DIR:-./frontend}:/app/frontend"
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

volumes:
  cerebro-redis:
  cerebro-data:
`;
  }
}

module.exports = { DockerManager };
