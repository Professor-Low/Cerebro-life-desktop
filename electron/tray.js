const { Tray, Menu, app, nativeImage, BrowserWindow } = require('electron');

let trayInstance = null;

function createTray(mainWindow, iconPath) {
  let icon;
  try {
    icon = nativeImage.createFromPath(iconPath);
    if (!icon.isEmpty()) {
      icon = icon.resize({ width: 16, height: 16 });
    }
  } catch {
    icon = nativeImage.createEmpty();
  }

  trayInstance = new Tray(icon);
  trayInstance.setToolTip('Cerebro - Your AI, Everywhere');

  const contextMenu = buildMenu(mainWindow, 'starting');
  trayInstance.setContextMenu(contextMenu);

  trayInstance.on('click', () => {
    if (mainWindow) {
      if (mainWindow.isVisible()) {
        mainWindow.hide();
      } else {
        mainWindow.show();
        mainWindow.focus();
      }
    }
  });

  return trayInstance;
}

function buildMenu(mainWindow, status) {
  const statusLabels = {
    starting: 'Starting...',
    running: 'Running',
    stopping: 'Stopping...',
    error: 'Error',
  };

  const statusLabel = statusLabels[status] || 'Unknown';

  return Menu.buildFromTemplate([
    {
      label: 'Open Cerebro',
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          mainWindow.focus();
        }
      },
    },
    { type: 'separator' },
    {
      label: `Status: ${statusLabel}`,
      enabled: false,
    },
    { type: 'separator' },
    {
      label: 'Auto-start on Login',
      type: 'checkbox',
      checked: app.getLoginItemSettings().openAtLogin,
      click: (menuItem) => {
        app.setLoginItemSettings({
          openAtLogin: menuItem.checked,
          path: app.getPath('exe'),
        });
      },
    },
    { type: 'separator' },
    {
      label: 'Quit Cerebro',
      click: () => {
        app.quit();
      },
    },
  ]);
}

function updateTrayStatus(tray, status) {
  if (!tray || tray.isDestroyed()) return;

  const mainWindow = BrowserWindow.getAllWindows().find(w => !w.isDestroyed());
  const contextMenu = buildMenu(mainWindow, status);
  tray.setContextMenu(contextMenu);
}

module.exports = { createTray, updateTrayStatus };
