const https = require('https');
const http = require('http');
const crypto = require('crypto');
const os = require('os');
const Store = require('electron-store');

const VALIDATE_URL = 'https://cerebro.life/api/license/validate';
const REDEEM_URL = 'https://cerebro.life/api/activation/redeem';
const GRACE_PERIOD_DAYS = 7;

// License key: CPRO-XXXXXXXX-XXXXXXXX-XXXXXXXX or CPRP-XXXXXXXX-XXXXXXXX-XXXXXXXX
const LICENSE_KEY_REGEX = /^(CPRO|CPRP)-[A-Z0-9]{8}-[A-Z0-9]{8}-[A-Z0-9]{8}$/;
// Activation code: 8 uppercase alphanumeric chars (no dashes)
const ACTIVATION_CODE_REGEX = /^[A-Z0-9]{4,12}$/;

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
   * Detect if input is a license key or activation code.
   */
  _detectFormat(input) {
    const trimmed = input.trim().toUpperCase();
    if (LICENSE_KEY_REGEX.test(trimmed)) return 'license_key';
    if (ACTIVATION_CODE_REGEX.test(trimmed) && !trimmed.includes('-')) return 'activation_code';
    return 'unknown';
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
   * Activate with either a license key or activation code.
   */
  async activate(input) {
    try {
      const trimmed = input.trim().toUpperCase();
      const format = this._detectFormat(trimmed);

      let result;
      if (format === 'activation_code') {
        result = await this._redeemActivationCode(trimmed);
      } else {
        // Try as license key (validate endpoint also accepts both formats now)
        result = await this._validateRemote(trimmed);
      }

      if (result.valid || result.user_id) {
        // Both endpoints return plan — normalize response
        const plan = result.plan;
        const expiresAt = result.expires_at;

        this.store.set('licenseKey', trimmed);
        this.store.set('plan', plan);
        this.store.set('expiresAt', expiresAt);
        this.store.set('lastValidated', new Date().toISOString());
        this.store.set('machineId', this.getMachineId());
        this.store.set('inputFormat', format);
        return { success: true, plan };
      }

      return { success: false, error: result.error || 'Activation failed' };
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
      // Always use validate endpoint for revalidation (supports both formats)
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
   * Call the license validation endpoint (accepts both license keys and activation codes).
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
   * Redeem an activation code via the dedicated endpoint.
   */
  _redeemActivationCode(code) {
    return new Promise((resolve, reject) => {
      const deviceInfo = {
        hostname: os.hostname(),
        platform: os.platform(),
        arch: os.arch(),
        machineId: this.getMachineId(),
      };
      const body = JSON.stringify({ code, device_info: deviceInfo });
      const url = new URL(REDEEM_URL);

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
            reject(new Error('Invalid response from activation server'));
          }
        });
      });

      req.on('error', reject);
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Activation server timeout'));
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
