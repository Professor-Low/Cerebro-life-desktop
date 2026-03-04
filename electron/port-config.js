const path = require('path');
const fs = require('fs');
const os = require('os');

const CEREBRO_DIR = path.join(os.homedir(), '.cerebro');
const PORT_CONFIG_FILE = path.join(CEREBRO_DIR, 'ports.json');

const DEFAULTS = {
  backendPort: 61000,
  redisPort: 16379,
  ttsPort: 8880,
};

function loadPortConfig() {
  try {
    if (fs.existsSync(PORT_CONFIG_FILE)) {
      const raw = JSON.parse(fs.readFileSync(PORT_CONFIG_FILE, 'utf-8'));
      return { ...DEFAULTS, ...raw };
    }
  } catch (e) {
    console.warn('[PortConfig] Failed to read ports.json:', e.message);
  }
  return { ...DEFAULTS };
}

function savePortConfig(cfg) {
  fs.mkdirSync(CEREBRO_DIR, { recursive: true });
  fs.writeFileSync(PORT_CONFIG_FILE, JSON.stringify(cfg, null, 2));
}

function getBackendUrl(cfg) {
  const port = (cfg && cfg.backendPort) || DEFAULTS.backendPort;
  return `http://localhost:${port}`;
}

module.exports = { loadPortConfig, savePortConfig, getBackendUrl, DEFAULTS };
