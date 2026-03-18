"""
SimEngine Client for Cerebro

Async HTTP client for the SimEngine REST API.
Handles health checks, auto-start, and the full simulation pipeline:
  interpret -> simulate -> analyze
"""

import sys
import asyncio
import subprocess
import logging
from typing import Optional, Dict, Any

import aiohttp

from autonomy_config import AutonomyConfig

logger = logging.getLogger(__name__)


class SimEngineClient:
    """
    Async client for the SimEngine simulation engine.

    Provides methods for each SimEngine API endpoint and a
    convenience run_full_pipeline() that chains them together.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        simengine_path: Optional[str] = None,
        auto_start: Optional[bool] = None,
    ):
        self.base_url = (base_url or AutonomyConfig.SIMENGINE_URL).rstrip("/")
        self.simengine_path = simengine_path or AutonomyConfig.SIMENGINE_PATH
        self.auto_start = auto_start if auto_start is not None else AutonomyConfig.SIMENGINE_AUTO_START
        self._session: Optional[aiohttp.ClientSession] = None
        self._process: Optional[subprocess.Popen] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Health & Lifecycle ──────────────────────────────────────────

    async def check_health(self) -> bool:
        """Check if SimEngine is running and healthy."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def ensure_running(self) -> bool:
        """
        Ensure SimEngine is running. If not and auto_start is enabled,
        start it as a subprocess and wait up to 15 seconds for it to respond.

        Returns True if SimEngine is available after this call.
        """
        if await self.check_health():
            return True

        if not self.auto_start:
            logger.warning("SimEngine is not running and auto_start is disabled")
            return False

        logger.info("SimEngine not running, starting from %s", self.simengine_path)

        try:
            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

            self._process = subprocess.Popen(
                [sys.executable, "-m", "backend.main"],
                cwd=self.simengine_path,
                creationflags=creation_flags,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.error("Failed to start SimEngine: %s", e)
            return False

        # Wait up to 15 seconds for SimEngine to become healthy
        for _ in range(30):
            await asyncio.sleep(0.5)
            if await self.check_health():
                logger.info("SimEngine started successfully")
                return True

        logger.error("SimEngine failed to become healthy within 15 seconds")
        return False

    # ── API Endpoints ───────────────────────────────────────────────

    async def interpret(self, text: str) -> Dict[str, Any]:
        """
        POST /api/interpret - Convert natural language to simulation config.

        Args:
            text: Natural language description of what to simulate.

        Returns:
            Simulation configuration dict.
        """
        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/api/interpret",
            json={"text": text},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def simulate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /api/simulate - Run a simulation with the given config.

        Args:
            config: Simulation configuration (from interpret or manual).

        Returns:
            Dict with result_id, statistics, metadata.
        """
        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/api/simulate",
            json=config,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def analyze(self, result_id: str) -> Dict[str, Any]:
        """
        POST /api/analyze - Analyze simulation results.

        Args:
            result_id: ID from a simulate() call.

        Returns:
            Analysis report dict.
        """
        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/api/analyze",
            json={"result_id": result_id},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_plugins(self) -> Dict[str, Any]:
        """GET /api/plugins - List available simulation plugins."""
        session = await self._get_session()
        async with session.get(f"{self.base_url}/api/plugins") as resp:
            resp.raise_for_status()
            return await resp.json()

    # ── Strategy Endpoints ──────────────────────────────────────────

    async def list_strategies(self) -> list:
        """GET /api/strategies - List saved strategies."""
        session = await self._get_session()
        async with session.get(f"{self.base_url}/api/strategies") as resp:
            resp.raise_for_status()
            return await resp.json()

    async def generate_strategy(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """POST /api/strategies/generate - LLM generates strategy from NL."""
        session = await self._get_session()
        payload: Dict[str, Any] = {"query": query}
        if context:
            payload["context"] = context
        async with session.post(
            f"{self.base_url}/api/strategies/generate",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def run_strategy(self, name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """POST /api/strategies/{name}/run - Execute a strategy."""
        session = await self._get_session()
        payload = {"parameters": parameters or {}}
        async with session.post(
            f"{self.base_url}/api/strategies/{name}/run",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def optimize_strategy(self, name: str) -> Dict[str, Any]:
        """POST /api/strategies/{name}/optimize - Optimize strategy params."""
        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/api/strategies/{name}/optimize",
            json={},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def query_prediction_markets(self, question: str, mode: str = "odds") -> Dict[str, Any]:
        """Run a prediction market query via the predictions plugin."""
        return await self.pipeline(
            f"What are the odds of: {question}" if mode == "odds" else f"Simulate: {question}",
            skip_analysis=(mode == "odds"),
        )

    # ── Pipeline ────────────────────────────────────────────────────

    async def pipeline(
        self, text: str, skip_analysis: bool = False, iterations: int = None
    ) -> Dict[str, Any]:
        """
        POST /api/pipeline - Run interpret+simulate+analyze server-side in one call.

        Args:
            text: Natural language simulation query.
            skip_analysis: Skip the AI analysis step.
            iterations: Override iteration count.

        Returns:
            Combined pipeline result dict.
        """
        session = await self._get_session()
        payload: Dict[str, Any] = {"text": text, "skip_analysis": skip_analysis}
        if iterations is not None:
            payload["iterations"] = iterations
        async with session.post(
            f"{self.base_url}/api/pipeline",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def run_full_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Run the full simulation pipeline.

        Tries the server-side /api/pipeline endpoint first (single HTTP call).
        Falls back to 3-call approach if the endpoint returns 404 (older SimEngine).

        Args:
            query: Natural language description of the simulation.

        Returns:
            Combined result with interpret, simulate, and analyze outputs.
        """
        if not await self.ensure_running():
            return {"error": "SimEngine is not available", "success": False}

        # Try server-side pipeline first
        try:
            result = await self.pipeline(query)
            return result
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                logger.info("Pipeline endpoint not available, falling back to 3-call approach")
            else:
                logger.error("SimEngine pipeline API error: %s %s", e.status, e.message)
                return {"error": f"SimEngine API error: {e.status} {e.message}", "success": False}
        except asyncio.TimeoutError:
            logger.error("SimEngine pipeline timed out (300s)")
            return {"error": "SimEngine pipeline timed out", "success": False}
        except Exception as e:
            logger.warning("Pipeline endpoint failed (%s), trying 3-call fallback", e)

        # Fallback: 3 sequential calls
        try:
            interpretation = await self.interpret(query)
            sim_result = await self.simulate(interpretation)
            result_id = sim_result.get("result_id")

            analysis = None
            if result_id:
                try:
                    analysis = await self.analyze(result_id)
                except Exception as e:
                    logger.warning("Analysis failed (non-fatal): %s", e)

            return {
                "success": True,
                "interpretation": interpretation,
                "simulation": sim_result,
                "analysis": analysis,
            }
        except asyncio.TimeoutError:
            logger.error("SimEngine 3-call pipeline timed out")
            return {"error": "SimEngine timed out", "success": False}
        except aiohttp.ClientResponseError as e:
            logger.error("SimEngine API error: %s %s", e.status, e.message)
            return {"error": f"SimEngine API error: {e.status} {e.message}", "success": False}
        except Exception as e:
            logger.error("SimEngine pipeline error: %s", e)
            return {"error": str(e), "success": False}


# ── Singleton ───────────────────────────────────────────────────────

_client_instance: Optional[SimEngineClient] = None


def get_sim_engine_client(
    base_url: Optional[str] = None,
    simengine_path: Optional[str] = None,
    auto_start: Optional[bool] = None,
) -> SimEngineClient:
    """Get or create the SimEngine client singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = SimEngineClient(base_url, simengine_path, auto_start)
    return _client_instance
