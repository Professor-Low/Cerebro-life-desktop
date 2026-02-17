const { spawn } = require('child_process');
const { EventEmitter } = require('events');
const path = require('path');
const fs = require('fs');
const os = require('os');
const net = require('net');

class RedisManager extends EventEmitter {
  constructor(options = {}) {
    super();
    this.port = options.port || 16379;
    this.binaryPath = options.binaryPath;
    this.process = null;
    this._running = false;
    this._configPath = null;
  }

  async start() {
    if (this._running) return;

    // Create minimal config
    this._configPath = path.join(os.tmpdir(), `cerebro-redis-${this.port}.conf`);
    const config = [
      `bind 127.0.0.1`,
      `port ${this.port}`,
      `maxmemory 256mb`,
      `maxmemory-policy allkeys-lru`,
      `save ""`,
      `appendonly no`,
      `daemonize no`,
      `loglevel warning`,
    ].join('\n');
    fs.writeFileSync(this._configPath, config);

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Redis failed to start within 10 seconds'));
      }, 10000);

      const binaryFile = this._findBinary();
      if (!binaryFile) {
        clearTimeout(timeout);
        reject(new Error('Redis binary not found. Install Redis or place redis-server in the redis/ directory.'));
        return;
      }

      console.log(`[RedisManager] Starting: ${binaryFile} ${this._configPath}`);

      this.process = spawn(binaryFile, [this._configPath], {
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      this.process.on('error', (err) => {
        clearTimeout(timeout);
        this._running = false;
        this.emit('error', err);
        reject(err);
      });

      this.process.on('exit', (code, signal) => {
        this._running = false;
        this.emit('exit', code, signal);
        if (!this._stopping) {
          console.error(`[RedisManager] Process exited unexpectedly: code=${code} signal=${signal}`);
        }
      });

      if (this.process.stdout) {
        this.process.stdout.on('data', (data) => {
          const msg = data.toString().trim();
          if (msg) console.log(`[Redis] ${msg}`);
        });
      }
      if (this.process.stderr) {
        this.process.stderr.on('data', (data) => {
          const msg = data.toString().trim();
          if (msg) console.error(`[Redis] ${msg}`);
        });
      }

      // Poll for Redis readiness via TCP connection
      this._pollReady(resolve, reject, timeout);
    });
  }

  _findBinary() {
    const isWin = process.platform === 'win32';
    const binaryName = isWin ? 'redis-server.exe' : 'redis-server';

    // Check bundled location first
    const bundled = path.join(this.binaryPath, binaryName);
    if (fs.existsSync(bundled)) {
      return bundled;
    }

    // Check common system paths
    const systemPaths = isWin
      ? ['C:\\Program Files\\Redis\\redis-server.exe', 'C:\\Redis\\redis-server.exe']
      : ['/usr/bin/redis-server', '/usr/local/bin/redis-server', '/opt/homebrew/bin/redis-server'];

    for (const p of systemPaths) {
      if (fs.existsSync(p)) {
        return p;
      }
    }

    // Search PATH directories
    const pathDirs = (process.env.PATH || '').split(path.delimiter);
    for (const dir of pathDirs) {
      const candidate = path.join(dir, binaryName);
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    }

    return null;
  }

  _pollReady(resolve, reject, timeout) {
    let attempts = 0;
    const maxAttempts = 20;

    const check = () => {
      if (attempts >= maxAttempts) {
        clearTimeout(timeout);
        reject(new Error('Redis health check timed out'));
        return;
      }
      attempts++;

      const client = new net.Socket();
      client.setTimeout(1000);

      client.connect(this.port, '127.0.0.1', () => {
        client.write('PING\r\n');
      });

      client.on('data', (data) => {
        const response = data.toString().trim();
        client.destroy();
        if (response === '+PONG') {
          clearTimeout(timeout);
          this._running = true;
          this.emit('ready');
          resolve();
        } else {
          setTimeout(check, 500);
        }
      });

      client.on('error', () => {
        client.destroy();
        setTimeout(check, 500);
      });

      client.on('timeout', () => {
        client.destroy();
        setTimeout(check, 500);
      });
    };

    setTimeout(check, 300);
  }

  async stop() {
    this._stopping = true;

    if (!this.process) {
      this._running = false;
      return;
    }

    return new Promise((resolve) => {
      const forceTimeout = setTimeout(() => {
        if (this.process) {
          console.log('[RedisManager] Force killing Redis');
          this.process.kill('SIGKILL');
        }
        this._running = false;
        this._cleanup();
        resolve();
      }, 3000);

      this.process.once('exit', () => {
        clearTimeout(forceTimeout);
        this._running = false;
        this._stopping = false;
        this._cleanup();
        resolve();
      });

      // Try graceful shutdown via SHUTDOWN command
      const client = new net.Socket();
      client.connect(this.port, '127.0.0.1', () => {
        client.write('SHUTDOWN NOSAVE\r\n');
        client.destroy();
      });
      client.on('error', () => {
        // Fall back to SIGTERM
        if (this.process) {
          if (process.platform === 'win32') {
            this.process.kill();
          } else {
            this.process.kill('SIGTERM');
          }
        }
        client.destroy();
      });
    });
  }

  _cleanup() {
    if (this._configPath && fs.existsSync(this._configPath)) {
      try {
        fs.unlinkSync(this._configPath);
      } catch {
        // Ignore cleanup errors
      }
    }
  }

  isRunning() {
    return this._running;
  }
}

module.exports = { RedisManager };
