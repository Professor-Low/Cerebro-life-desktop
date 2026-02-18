const https = require('https');
const http = require('http');
const crypto = require('crypto');
const os = require('os');
const Store = require('electron-store');

const VALIDATE_URL = 'https://www.cerebro.life/api/license/validate';
const REDEEM_URL = 'https://www.cerebro.life/api/activation/redeem';
const GRACE_PERIOD_DAYS = 7;
const MAX_REDIRECTS = 3;

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

  _deriveEncryptionKey() {
    const raw = `${os.hostname()}|${os.cpus()[0]?.model || 'unknown'}|${os.userInfo().username}`;
    return crypto.createHash('sha256').update(raw).digest('hex').slice(0, 32);
  }

  getMachineId() {
    const raw = `${os.hostname()}|${os.cpus()[0]?.model || 'unknown'}|${os.userInfo().username}`;
    return crypto.createHash('sha256').update(raw).digest('hex');
  }

  _detectFormat(input) {
    const trimmed = input.trim().toUpperCase();
    if (LICENSE_KEY_REGEX.test(trimmed)) return 'license_key';
    if (ACTIVATION_CODE_REGEX.test(trimmed) && !trimmed.includes('-')) return 'activation_code';
    return 'unknown';
  }

  getStatus() {
    const key = this.store.get('licenseKey');
    const plan = this.store.get('plan');
    const lastValidated = this.store.get('lastValidated');
    const expiresAt = this.store.get('expiresAt');

    if (!key) {
      return { valid: false, reason: 'no_license' };
    }

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
          offline: daysSince > 0.5,
        };
      }

      return { valid: false, reason: 'grace_expired', plan };
    }

    return { valid: false, reason: 'never_validated' };
  }

  async activate(input) {
    try {
      const trimmed = input.trim().toUpperCase();
      const format = this._detectFormat(trimmed);

      let result;
      if (format === 'activation_code') {
        result = await this._redeemActivationCode(trimmed);
      } else {
        result = await this._validateRemote(trimmed);
      }

      if (result.valid || result.user_id) {
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

      const lastValidated = this.store.get('lastValidated');
      if (lastValidated) {
        const daysSince = (new Date() - new Date(lastValidated)) / (1000 * 60 * 60 * 24);
        if (daysSince <= GRACE_PERIOD_DAYS) {
          return { valid: true, plan: this.store.get('plan'), offline: true };
        }
      }

      return { valid: false, reason: result.error || 'invalid' };
    } catch (err) {
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
   * Make an HTTPS POST request with automatic redirect following.
   */
  _requestWithRedirects(url, body, redirectCount = 0) {
    return new Promise((resolve, reject) => {
      if (redirectCount > MAX_REDIRECTS) {
        reject(new Error('Too many redirects'));
        return;
      }

      const parsed = new URL(url);
      const options = {
        hostname: parsed.hostname,
        port: parsed.port || 443,
        path: parsed.pathname + parsed.search,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
        timeout: 10000,
      };

      const transport = parsed.protocol === 'https:' ? https : http;
      const req = transport.request(options, (res) => {
        // Handle redirects (301, 302, 307, 308)
        if ([301, 302, 307, 308].includes(res.statusCode) && res.headers.location) {
          const redirectUrl = new URL(res.headers.location, url).href;
          console.log(`[License] Redirect ${res.statusCode} -> ${redirectUrl}`);
          res.resume(); // drain response
          this._requestWithRedirects(redirectUrl, body, redirectCount + 1)
            .then(resolve)
            .catch(reject);
          return;
        }

        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try {
            resolve(JSON.parse(data));
          } catch {
            reject(new Error(`Invalid JSON response (HTTP ${res.statusCode}): ${data.slice(0, 200)}`));
          }
        });
      });

      req.on('error', reject);
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Server timeout'));
      });

      req.write(body);
      req.end();
    });
  }

  _validateRemote(licenseKey) {
    const body = JSON.stringify({ license_key: licenseKey });
    return this._requestWithRedirects(VALIDATE_URL, body);
  }

  _redeemActivationCode(code) {
    const deviceInfo = {
      hostname: os.hostname(),
      platform: os.platform(),
      arch: os.arch(),
      machineId: this.getMachineId(),
    };
    const body = JSON.stringify({ code, device_info: deviceInfo });
    return this._requestWithRedirects(REDEEM_URL, body);
  }

  deactivate() {
    this.store.clear();
  }
}

module.exports = { LicenseManager };
