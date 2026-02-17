const { spawn } = require('child_process');
const { EventEmitter } = require('events');
const path = require('path');
const http = require('http');
const fs = require('fs');

class BackendManager extends EventEmitter {
  constructor(options = {}) {
    super();
    this.port = options.port || 59000;
    this.binaryPath = options.binaryPath;
    this.isDev = options.isDev || false;
    this.backendSrc = options.backendSrc;
    this.process = null;
    this._running = false;
    this._healthCheckInterval = null;
  }

  async start() {
    if (this._running) return;

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Backend failed to start within 30 seconds'));
      }, 30000);

      try {
        if (this.isDev && this.backendSrc) {
          this._startDev();
        } else {
          this._startBinary();
        }
      } catch (err) {
        clearTimeout(timeout);
        reject(err);
        return;
      }

      this.process.on('error', (err) => {
        clearTimeout(timeout);
        this._running = false;
        this.emit('error', err);
        reject(err);
      });

      this.process.on('exit', (code, signal) => {
        this._running = false;
        this._stopHealthCheck();
        this.emit('exit', code, signal);
        if (!this._stopping) {
          console.error(`[BackendManager] Process exited unexpectedly: code=${code} signal=${signal}`);
        }
      });

      // Pipe output for logging
      if (this.process.stdout) {
        this.process.stdout.on('data', (data) => {
          const msg = data.toString().trim();
          if (msg) console.log(`[Backend] ${msg}`);
        });
      }
      if (this.process.stderr) {
        this.process.stderr.on('data', (data) => {
          const msg = data.toString().trim();
          if (msg) console.error(`[Backend] ${msg}`);
        });
      }

      // Poll health endpoint
      this._pollHealth(resolve, reject, timeout);
    });
  }

  _startDev() {
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    const mainPy = path.join(this.backendSrc, 'main.py');

    console.log(`[BackendManager] Starting dev mode: ${pythonCmd} ${mainPy}`);

    this.process = spawn(pythonCmd, [mainPy], {
      cwd: this.backendSrc,
      env: {
        ...process.env,
        CEREBRO_HOST: '127.0.0.1',
        CEREBRO_PORT: String(this.port),
        REDIS_URL: 'redis://localhost:16379/0',
        CEREBRO_STANDALONE: '1',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  }

  _startBinary() {
    const isWin = process.platform === 'win32';
    const binaryName = isWin ? 'cerebro-server.exe' : 'cerebro-server';
    const binaryDir = this.binaryPath;

    // Look for the binary in several locations
    const candidates = [
      path.join(binaryDir, binaryName),
      path.join(binaryDir, 'cerebro-server', binaryName),
    ];

    let binaryFile = null;
    for (const candidate of candidates) {
      if (fs.existsSync(candidate)) {
        binaryFile = candidate;
        break;
      }
    }

    if (!binaryFile) {
      throw new Error(`Backend binary not found. Checked: ${candidates.join(', ')}`);
    }

    console.log(`[BackendManager] Starting binary: ${binaryFile}`);

    this.process = spawn(binaryFile, [], {
      cwd: path.dirname(binaryFile),
      env: {
        ...process.env,
        CEREBRO_HOST: '127.0.0.1',
        CEREBRO_PORT: String(this.port),
        REDIS_URL: 'redis://localhost:16379/0',
        CEREBRO_STANDALONE: '1',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  }

  _pollHealth(resolve, reject, timeout) {
    let attempts = 0;
    const maxAttempts = 60; // 30 seconds at 500ms intervals

    const check = () => {
      if (attempts >= maxAttempts) {
        clearTimeout(timeout);
        reject(new Error('Backend health check timed out'));
        return;
      }
      attempts++;

      const req = http.get(`http://127.0.0.1:${this.port}/health`, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          if (res.statusCode === 200) {
            clearTimeout(timeout);
            this._running = true;
            this._startHealthCheck();
            this.emit('ready');
            resolve();
          } else {
            setTimeout(check, 500);
          }
        });
      });

      req.on('error', () => {
        setTimeout(check, 500);
      });

      req.setTimeout(2000, () => {
        req.destroy();
        setTimeout(check, 500);
      });
    };

    // Wait a moment before first check
    setTimeout(check, 1000);
  }

  _startHealthCheck() {
    this._healthCheckInterval = setInterval(() => {
      const req = http.get(`http://127.0.0.1:${this.port}/health`, (res) => {
        if (res.statusCode !== 200) {
          this.emit('unhealthy');
        }
        res.resume(); // Consume response data
      });
      req.on('error', () => {
        this.emit('unhealthy');
      });
      req.setTimeout(5000, () => {
        req.destroy();
      });
    }, 10000);
  }

  _stopHealthCheck() {
    if (this._healthCheckInterval) {
      clearInterval(this._healthCheckInterval);
      this._healthCheckInterval = null;
    }
  }

  async stop() {
    this._stopping = true;
    this._stopHealthCheck();

    if (!this.process) {
      this._running = false;
      return;
    }

    return new Promise((resolve) => {
      const forceTimeout = setTimeout(() => {
        if (this.process) {
          console.log('[BackendManager] Force killing backend');
          this.process.kill('SIGKILL');
        }
        this._running = false;
        resolve();
      }, 5000);

      this.process.once('exit', () => {
        clearTimeout(forceTimeout);
        this._running = false;
        this._stopping = false;
        resolve();
      });

      // Graceful shutdown
      if (process.platform === 'win32') {
        this.process.kill();
      } else {
        this.process.kill('SIGTERM');
      }
    });
  }

  isRunning() {
    return this._running;
  }
}

module.exports = { BackendManager };
