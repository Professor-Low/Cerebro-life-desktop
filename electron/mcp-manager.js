const { spawn } = require('child_process');
const { EventEmitter } = require('events');
const path = require('path');
const fs = require('fs');
const os = require('os');

class McpManager extends EventEmitter {
  constructor(options = {}) {
    super();
    this.binaryPath = options.binaryPath;
    this.isDev = options.isDev || false;
    this.mcpSrc = options.mcpSrc;
    this.process = null;
    this._running = false;
    this._dataDir = path.join(os.homedir(), '.cerebro', 'data');
  }

  async start() {
    if (this._running) return;

    // Ensure data directory exists
    fs.mkdirSync(this._dataDir, { recursive: true });

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        // MCP server is stdio-based, so "started" means process is alive
        // Give it a generous timeout since it initializes on first tool call
        reject(new Error('MCP server failed to start within 15 seconds'));
      }, 15000);

      try {
        if (this.isDev && this.mcpSrc) {
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
        this.emit('exit', code, signal);
        if (!this._stopping) {
          console.error(`[McpManager] Process exited unexpectedly: code=${code} signal=${signal}`);
        }
      });

      if (this.process.stderr) {
        this.process.stderr.on('data', (data) => {
          const msg = data.toString().trim();
          if (msg) console.log(`[MCP] ${msg}`);
        });
      }

      // MCP server communicates via stdio (JSON-RPC).
      // It's "ready" once the process is alive and hasn't crashed.
      // Wait a short moment to confirm it stays up.
      setTimeout(() => {
        if (this.process && !this.process.killed) {
          clearTimeout(timeout);
          this._running = true;
          this.emit('ready');
          resolve();
        }
      }, 2000);
    });
  }

  _startDev() {
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    const cliPath = path.join(this.mcpSrc, 'src', 'cli.py');

    console.log(`[McpManager] Starting dev mode: ${pythonCmd} -m cerebro_memory.cli serve`);

    // In dev mode, run the installed cerebro command
    this.process = spawn('cerebro', ['serve'], {
      env: {
        ...process.env,
        CEREBRO_DATA_DIR: this._dataDir,
      },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
  }

  _startBinary() {
    const isWin = process.platform === 'win32';
    const binaryName = isWin ? 'cerebro-mcp-server.exe' : 'cerebro-mcp-server';
    const binaryDir = this.binaryPath;

    const candidates = [
      path.join(binaryDir, binaryName),
      path.join(binaryDir, 'cerebro-mcp-server', binaryName),
    ];

    let binaryFile = null;
    for (const candidate of candidates) {
      if (fs.existsSync(candidate)) {
        binaryFile = candidate;
        break;
      }
    }

    if (!binaryFile) {
      throw new Error(`MCP server binary not found. Checked: ${candidates.join(', ')}`);
    }

    console.log(`[McpManager] Starting binary: ${binaryFile}`);

    this.process = spawn(binaryFile, [], {
      cwd: path.dirname(binaryFile),
      env: {
        ...process.env,
        CEREBRO_DATA_DIR: this._dataDir,
      },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
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
          console.log('[McpManager] Force killing MCP server');
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

module.exports = { McpManager };
