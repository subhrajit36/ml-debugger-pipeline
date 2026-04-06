"""
client.py
---------
A reusable HTTP client for interacting with the ML Pipeline Debugger environment.

This module provides a clean Python API for:
  - Checking server health
  - Resetting episodes (loading a task/scenario)
  - Taking actions (stepping through an episode)

Usage:
    from client import MLDebuggerClient

    client = MLDebuggerClient("http://localhost:7860")
    client.health()
    obs = client.reset("task1_easy", scenario_id=1)
    result = client.step("fix_tensor_reshape", {"layer_name": "linear1", "new_shape": 25088})
"""

import requests
from typing import Dict, Any, Optional


class MLDebuggerClient:
    """
    HTTP client for the ML Pipeline Debugger environment server.

    Wraps the 3 endpoints (/health, /reset, /step) into simple Python methods.
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30):
        """
        Args:
            base_url: URL where the environment server is running
            timeout:  HTTP request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ─────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """
        Check if the server is alive.

        Returns:
            {"status": "ok", "version": "1.0.0"}

        Raises:
            requests.RequestException if the server is unreachable
        """
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ─────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────

    def reset(self, task_id: str, scenario_id: int = 1) -> Dict[str, Any]:
        """
        Start a new episode.

        Args:
            task_id:     one of "task1_easy", "task2_medium", "task3_hard"
            scenario_id: which scenario within the task (default 1)

        Returns:
            {
                "observation": {...},
                "task_description": "...",
                "max_steps": int
            }
        """
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "scenario_id": scenario_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ─────────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────────

    def step(self, action_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Take one action in the current episode.

        Args:
            action_type: one of the valid action names (e.g. "fix_tensor_reshape")
            parameters:  dict of parameters for the action

        Returns:
            {
                "observation": {...},
                "score": float,
                "info": {...}
            }
        """
        resp = requests.post(
            f"{self.base_url}/step",
            json={
                "action": {
                    "action_type": action_type,
                    "parameters": parameters or {},
                }
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ─────────────────────────────────────────────
    # Convenience: check if server is online
    # ─────────────────────────────────────────────

    def is_online(self) -> bool:
        """Returns True if the server responds to /health, False otherwise."""
        try:
            data = self.health()
            return data.get("status") == "ok"
        except Exception:
            return False
