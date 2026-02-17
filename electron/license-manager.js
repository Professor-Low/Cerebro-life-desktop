const https = require('https');
const http = require('http');
const crypto = require('crypto');
const os = require('os');
const Store = require('electron-store');

const VALIDATE_URL = 'https://cerebro.life/api/license/validate';
const GRACE_PERIOD_DAYS = 7;

class LicenseManager {
  constructor() {
    this.store = new Store({
      name: 'cerebro-license',
      encryptionKey: this._deriveEncryptionKey(),
    });
  }

  /**
   * Derive a machine-specific encryption key for the license store.
   */
  _deriveEncryptionKey() {
    const raw = `${os.hostname()}|${os.cpus()[0]?.model || 'unknown'}|${os.userInfo().username}`;
    return crypto.createHash('sha256').update(raw).digest('hex').slice(0, 32);
  }

  /**
   * Get a stable machine ID for license binding.
   */
  getMachineId() {
    const raw = `${os.hostname()}|${os.cpus()[0]?.model || 'unknown'}|${os.userInfo().username}`;
    return crypto.createHash('sha256').update(raw).digest('hex');
  }

  /**
   * Get stored license info.
   */
  getStatus() {
    const key = this.store.get('licenseKey');
    const plan = this.store.get('plan');
    const lastValidated = this.store.get('lastValidated');
    const expiresAt = this.store.get('expiresAt');

    if (!key) {
      return { valid: false, reason: 'no_license' };
    }

    // Check if we have a cached validation
    if (lastValidated) {
      const lastDate = new Date(lastValidated);
      const now = new Date();
      const daysSince = (now - lastDate) / (1000 * 60 * 60 * 24);

      if (daysSince <= GRACE_PERIOD_DAYS) {
        return {
          valid: true,
          plan,
          expiresAt,
          lastValidated,
          offline: daysSince > 0.5, // More than 12 hours since last check
        };
      }

      // Grace period expired
      return { valid: false, reason: 'grace_expired', plan };
    }

    return { valid: false, reason: 'never_validated' };
  }

  /**
   * Activate a license key by validating with the server.
   */
  async activate(licenseKey) {
    try {
      const result = await this._validateRemote(licenseKey);

      if (result.valid) {
        this.store.set('licenseKey', licenseKey);
        this.store.set('plan', result.plan);
        this.store.set('expiresAt', result.expires_at);
        this.store.set('lastValidated', new Date().toISOString());
        this.store.set('machineId', this.getMachineId());
        return { success: true, plan: result.plan };
      }

      return { success: false, error: result.error || 'License validation failed' };
    } catch (err) {
      return { success: false, error: `Connection error: ${err.message}` };
    }
  }

  /**
   * Re-validate the stored license (called on startup).
   */
  async revalidate() {
    const key = this.store.get('licenseKey');
    if (!key) return { valid: false, reason: 'no_license' };

    try {
      const result = await this._validateRemote(key);

      if (result.valid) {
        this.store.set('plan', result.plan);
        this.store.set('expiresAt', result.expires_at);
        this.store.set('lastValidated', new Date().toISOString());
        return { valid: true, plan: result.plan };
      }

      // Server says invalid — but check grace period
      const lastValidated = this.store.get('lastValidated');
      if (lastValidated) {
        const daysSince = (new Date() - new Date(lastValidated)) / (1000 * 60 * 60 * 24);
        if (daysSince <= GRACE_PERIOD_DAYS) {
          return { valid: true, plan: this.store.get('plan'), offline: true };
        }
      }

      return { valid: false, reason: result.error || 'invalid' };
    } catch (err) {
      // Network error — use grace period
      const lastValidated = this.store.get('lastValidated');
      if (lastValidated) {
        const daysSince = (new Date() - new Date(lastValidated)) / (1000 * 60 * 60 * 24);
        if (daysSince <= GRACE_PERIOD_DAYS) {
          return { valid: true, plan: this.store.get('plan'), offline: true };
        }
      }

      return { valid: false, reason: 'offline_grace_expired' };
    }
  }

  /**
   * Call the remote validation endpoint.
   */
  _validateRemote(licenseKey) {
    return new Promise((resolve, reject) => {
      const body = JSON.stringify({ license_key: licenseKey });
      const url = new URL(VALIDATE_URL);

      const options = {
        hostname: url.hostname,
        port: url.port || 443,
        path: url.pathname,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
        timeout: 10000,
      };

      const transport = url.protocol === 'https:' ? https : http;
      const req = transport.request(options, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try {
            resolve(JSON.parse(data));
          } catch {
            reject(new Error('Invalid response from license server'));
          }
        });
      });

      req.on('error', reject);
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('License server timeout'));
      });

      req.write(body);
      req.end();
    });
  }

  /**
   * Clear stored license data.
   */
  deactivate() {
    this.store.clear();
  }
}

module.exports = { LicenseManager };
