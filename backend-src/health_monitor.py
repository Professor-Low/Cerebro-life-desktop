"""
Health Monitor - Verify System Health After Changes

Monitors Cerebro's health after deployments:
- Check /health, /api/mood, and other endpoints
- Capture baseline latencies
- Detect performance regressions
- Wait-for-healthy with timeout
"""

import asyncio
import httpx
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class EndpointHealth:
    """Health status of a single endpoint."""
    path: str
    healthy: bool
    status_code: Optional[int] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    response_sample: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "healthy": self.healthy,
            "status_code": self.status_code,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "response_sample": self.response_sample
        }


@dataclass
class HealthReport:
    """Complete health report for the system."""
    healthy: bool
    endpoints: List[EndpointHealth] = field(default_factory=list)
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat())
    port: int = 59000
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "healthy": self.healthy,
            "endpoints": [e.to_dict() for e in self.endpoints],
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "checked_at": self.checked_at,
            "port": self.port,
            "error": self.error,
            "healthy_count": sum(1 for e in self.endpoints if e.healthy),
            "total_count": len(self.endpoints)
        }


class HealthMonitor:
    """
    Monitors Cerebro's health and performance.

    Checks critical endpoints and detects performance regressions
    after changes are deployed.
    """

    # Critical endpoints that MUST be healthy
    CRITICAL_ENDPOINTS = [
        {"path": "/health", "method": "GET", "expected_status": 200},
        {"path": "/", "method": "GET", "expected_status": 200},
    ]

    # Important endpoints to check
    IMPORTANT_ENDPOINTS = [
        {"path": "/api/mood", "method": "GET", "expected_status": 200},
        {"path": "/briefing", "method": "GET", "expected_status": [200, 401]},  # May need auth
        {"path": "/agents", "method": "GET", "expected_status": [200, 401]},
        {"path": "/api/autonomy/status", "method": "GET", "expected_status": [200, 401, 503]},
    ]

    # Performance thresholds (in ms)
    LATENCY_WARNING_MS = 500  # Warn if above
    LATENCY_CRITICAL_MS = 2000  # Fail if above

    def __init__(self, base_url: str = None):
        """
        Initialize the health monitor.

        Args:
            base_url: Base URL for health checks (default: http://localhost:59000)
        """
        self.base_url = base_url or "http://localhost:59000"
        self._baseline: Optional[HealthReport] = None

    def set_base_url(self, port: int):
        """Set base URL from port number."""
        self.base_url = f"http://localhost:{port}"

    async def check_endpoint(
        self,
        path: str,
        method: str = "GET",
        expected_status: Any = 200,
        timeout: float = 10.0
    ) -> EndpointHealth:
        """
        Check health of a single endpoint.

        Args:
            path: Endpoint path
            method: HTTP method
            expected_status: Expected status code(s)
            timeout: Request timeout in seconds

        Returns:
            EndpointHealth with check results
        """
        url = f"{self.base_url}{path}"

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                start_time = asyncio.get_event_loop().time()

                if method.upper() == "GET":
                    response = await client.get(url)
                elif method.upper() == "POST":
                    response = await client.post(url)
                else:
                    response = await client.request(method, url)

                latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                # Check status code
                if isinstance(expected_status, list):
                    healthy = response.status_code in expected_status
                else:
                    healthy = response.status_code == expected_status

                # Get response sample
                try:
                    response_sample = response.text[:200] if response.text else None
                except:
                    response_sample = None

                return EndpointHealth(
                    path=path,
                    healthy=healthy,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                    response_sample=response_sample
                )

        except httpx.ConnectError as e:
            return EndpointHealth(
                path=path,
                healthy=False,
                error=f"Connection failed: {str(e)}"
            )
        except httpx.TimeoutException:
            return EndpointHealth(
                path=path,
                healthy=False,
                error=f"Timeout after {timeout}s"
            )
        except Exception as e:
            return EndpointHealth(
                path=path,
                healthy=False,
                error=str(e)
            )

    async def check_health(self, port: int = None, include_important: bool = True) -> HealthReport:
        """
        Run full health check.

        Args:
            port: Port to check (updates base_url if provided)
            include_important: Also check non-critical endpoints

        Returns:
            HealthReport with all endpoint statuses
        """
        if port:
            self.set_base_url(port)

        report = HealthReport(healthy=True, port=port or 59000)

        # Check critical endpoints
        for endpoint in self.CRITICAL_ENDPOINTS:
            result = await self.check_endpoint(
                path=endpoint["path"],
                method=endpoint.get("method", "GET"),
                expected_status=endpoint.get("expected_status", 200)
            )
            report.endpoints.append(result)

            if not result.healthy:
                report.healthy = False

        # Check important endpoints if requested
        if include_important:
            for endpoint in self.IMPORTANT_ENDPOINTS:
                result = await self.check_endpoint(
                    path=endpoint["path"],
                    method=endpoint.get("method", "GET"),
                    expected_status=endpoint.get("expected_status", 200)
                )
                report.endpoints.append(result)
                # Important endpoints don't affect overall health status

        # Calculate latency stats
        latencies = [e.latency_ms for e in report.endpoints if e.latency_ms > 0]
        if latencies:
            report.total_latency_ms = sum(latencies)
            report.avg_latency_ms = report.total_latency_ms / len(latencies)

            # Check for critical latency
            if max(latencies) > self.LATENCY_CRITICAL_MS:
                report.error = f"Critical latency detected: {max(latencies):.0f}ms"
                # Don't fail health just for latency, but flag it

        return report

    async def wait_for_healthy(
        self,
        port: int = None,
        timeout_seconds: int = 60,
        poll_interval: float = 2.0
    ) -> HealthReport:
        """
        Wait for the system to become healthy.

        Polls health endpoint until healthy or timeout.

        Args:
            port: Port to check
            timeout_seconds: Maximum time to wait
            poll_interval: Seconds between checks

        Returns:
            Final HealthReport
        """
        if port:
            self.set_base_url(port)

        start_time = asyncio.get_event_loop().time()
        last_report = None

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_seconds:
                if last_report:
                    last_report.error = f"Timeout after {timeout_seconds}s - system not healthy"
                    return last_report
                return HealthReport(
                    healthy=False,
                    port=port or 59000,
                    error=f"Timeout after {timeout_seconds}s"
                )

            last_report = await self.check_health(port, include_important=False)

            if last_report.healthy:
                return last_report

            await asyncio.sleep(poll_interval)

    def capture_baseline(self, report: HealthReport):
        """
        Capture baseline health report for comparison.

        Args:
            report: Health report to use as baseline
        """
        self._baseline = report

    def detect_performance_regression(self, new_report: HealthReport) -> Dict[str, Any]:
        """
        Compare new health report to baseline for regressions.

        Args:
            new_report: New health report to compare

        Returns:
            Dict with regression analysis
        """
        result = {
            "has_regression": False,
            "latency_increase_ms": 0.0,
            "latency_increase_percent": 0.0,
            "degraded_endpoints": [],
            "message": "No baseline for comparison" if not self._baseline else "No regression detected"
        }

        if not self._baseline:
            return result

        # Compare overall latency
        if self._baseline.avg_latency_ms > 0:
            latency_increase = new_report.avg_latency_ms - self._baseline.avg_latency_ms
            latency_percent = (latency_increase / self._baseline.avg_latency_ms) * 100

            result["latency_increase_ms"] = latency_increase
            result["latency_increase_percent"] = latency_percent

            # Flag significant regression (>50% increase)
            if latency_percent > 50:
                result["has_regression"] = True
                result["message"] = f"Latency increased by {latency_percent:.1f}%"

        # Compare individual endpoints
        baseline_by_path = {e.path: e for e in self._baseline.endpoints}

        for new_endpoint in new_report.endpoints:
            baseline_endpoint = baseline_by_path.get(new_endpoint.path)
            if not baseline_endpoint:
                continue

            # Check if endpoint became unhealthy
            if baseline_endpoint.healthy and not new_endpoint.healthy:
                result["degraded_endpoints"].append({
                    "path": new_endpoint.path,
                    "was": "healthy",
                    "now": "unhealthy",
                    "error": new_endpoint.error
                })
                result["has_regression"] = True

            # Check for significant latency increase on individual endpoint
            if baseline_endpoint.latency_ms > 0 and new_endpoint.latency_ms > 0:
                increase = new_endpoint.latency_ms - baseline_endpoint.latency_ms
                if increase > self.LATENCY_WARNING_MS:
                    result["degraded_endpoints"].append({
                        "path": new_endpoint.path,
                        "was": f"{baseline_endpoint.latency_ms:.0f}ms",
                        "now": f"{new_endpoint.latency_ms:.0f}ms",
                        "increase_ms": increase
                    })

        if result["degraded_endpoints"]:
            result["has_regression"] = True
            result["message"] = f"{len(result['degraded_endpoints'])} endpoint(s) degraded"

        return result

    async def run_continuous_monitoring(
        self,
        port: int = 59000,
        interval_seconds: float = 30.0,
        callback=None
    ):
        """
        Run continuous health monitoring.

        Args:
            port: Port to monitor
            interval_seconds: Seconds between checks
            callback: Async function to call with each report
        """
        self.set_base_url(port)

        while True:
            report = await self.check_health()

            if callback:
                await callback(report)

            if not report.healthy:
                # More frequent checks when unhealthy
                await asyncio.sleep(interval_seconds / 3)
            else:
                await asyncio.sleep(interval_seconds)

    def get_health_summary(self, report: HealthReport) -> str:
        """
        Generate human-readable health summary.

        Args:
            report: Health report

        Returns:
            Formatted summary string
        """
        status = "HEALTHY" if report.healthy else "UNHEALTHY"
        lines = [
            f"Health Status: {status}",
            f"Port: {report.port}",
            f"Checked: {report.checked_at}",
            f"Avg Latency: {report.avg_latency_ms:.0f}ms",
            "",
            "Endpoints:"
        ]

        for endpoint in report.endpoints:
            status_icon = "OK" if endpoint.healthy else "FAIL"
            latency = f"{endpoint.latency_ms:.0f}ms" if endpoint.latency_ms > 0 else "N/A"
            lines.append(f"  [{status_icon}] {endpoint.path} - {latency}")
            if endpoint.error:
                lines.append(f"       Error: {endpoint.error}")

        if report.error:
            lines.append(f"\nError: {report.error}")

        return "\n".join(lines)


# Singleton instance
_monitor_instance: Optional[HealthMonitor] = None


def get_health_monitor(base_url: str = None) -> HealthMonitor:
    """Get or create the health monitor singleton."""
    global _monitor_instance

    if _monitor_instance is None:
        _monitor_instance = HealthMonitor(base_url)

    return _monitor_instance
