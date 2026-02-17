"""
Cerebro Autonomy Configuration

Feature flags and settings for the autonomous evolution features.
Set flags to False to disable specific features.
"""

import os


class AutonomyConfig:
    """Configuration for Cerebro's autonomous features."""

    # Master switch - set to False to disable all autonomy features
    ENABLE_AUTONOMY = os.environ.get("CEREBRO_AUTONOMY", "true").lower() == "true"

    # Individual feature flags
    ENABLE_MCP_BRIDGE = os.environ.get("CEREBRO_MCP_BRIDGE", "true").lower() == "true"
    ENABLE_PREDICTIONS = os.environ.get("CEREBRO_PREDICTIONS", "true").lower() == "true"
    ENABLE_PROACTIVE = os.environ.get("CEREBRO_PROACTIVE", "true").lower() == "true"
    ENABLE_LEARNING_INJECTION = os.environ.get("CEREBRO_LEARNING", "true").lower() == "true"
    ENABLE_SELF_MODIFICATION = os.environ.get("CEREBRO_SELF_MOD", "true").lower() == "true"
    ENABLE_SIMULATION = os.environ.get("CEREBRO_SIMULATION", "true").lower() == "true"
    ENABLE_TRADING = os.environ.get("CEREBRO_TRADING", "true").lower() == "true"

    # Alpaca Trading settings
    ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
    ALPACA_PAPER = os.environ.get("ALPACA_PAPER", "true").lower() == "true"

    # SimEngine settings
    SIMENGINE_URL = os.environ.get("CEREBRO_SIMENGINE_URL", "http://localhost:8000")
    SIMENGINE_PATH = os.environ.get("CEREBRO_SIMENGINE_PATH", os.path.expanduser("~/SimEngine"))
    SIMENGINE_AUTO_START = os.environ.get("CEREBRO_SIMENGINE_AUTO_START", "true").lower() == "true"

    # Proactive agent settings
    PROACTIVE_CHECK_INTERVAL = int(os.environ.get("CEREBRO_PROACTIVE_INTERVAL", "300"))  # 5 minutes
    PROACTIVE_REQUIRE_APPROVAL = os.environ.get("CEREBRO_PROACTIVE_APPROVAL", "false").lower() == "true"

    # Prediction settings
    PREDICTION_WARNING_THRESHOLD = float(os.environ.get("CEREBRO_PREDICTION_THRESHOLD", "0.6"))

    # Learning injection settings
    LEARNING_MAX_APPLY = int(os.environ.get("CEREBRO_LEARNING_MAX_APPLY", "3"))
    LEARNING_MAX_AVOID = int(os.environ.get("CEREBRO_LEARNING_MAX_AVOID", "3"))

    @classmethod
    def get_status(cls) -> dict:
        """Get current status of all feature flags."""
        return {
            "autonomy_enabled": cls.ENABLE_AUTONOMY,
            "features": {
                "mcp_bridge": cls.ENABLE_MCP_BRIDGE,
                "predictions": cls.ENABLE_PREDICTIONS,
                "proactive_agents": cls.ENABLE_PROACTIVE,
                "learning_injection": cls.ENABLE_LEARNING_INJECTION,
                "self_modification": cls.ENABLE_SELF_MODIFICATION,
                "simulation": cls.ENABLE_SIMULATION,
                "trading": cls.ENABLE_TRADING
            },
            "settings": {
                "proactive_check_interval": cls.PROACTIVE_CHECK_INTERVAL,
                "proactive_require_approval": cls.PROACTIVE_REQUIRE_APPROVAL,
                "prediction_threshold": cls.PREDICTION_WARNING_THRESHOLD,
                "learning_max_apply": cls.LEARNING_MAX_APPLY,
                "learning_max_avoid": cls.LEARNING_MAX_AVOID,
                "simengine_url": cls.SIMENGINE_URL,
                "simengine_path": cls.SIMENGINE_PATH,
                "simengine_auto_start": cls.SIMENGINE_AUTO_START,
                "alpaca_paper": cls.ALPACA_PAPER,
                "alpaca_configured": bool(cls.ALPACA_API_KEY and cls.ALPACA_SECRET_KEY)
            }
        }


# Singleton instance
config = AutonomyConfig()
