"""
app.py
------
The FastAPI web server for the ML Pipeline Debugger OpenEnv environment.

WHAT IS FASTAPI?
  FastAPI is a Python web framework. It lets us define "endpoints" — URLs that
  clients (like the AI agent) can send HTTP requests to.

OUR 4 ENDPOINTS:
  GET  /health  → Check if the server is alive
  POST /reset   → Start a new episode (loads a scenario)
  POST /step    → Take one action in the current episode
  GET  /state   → Return the current environment state

WHY HTTP?
  The OpenEnv standard requires environments to communicate via HTTP so that
  any language / any agent framework can interact with our environment.
"""

import sys
import os

# Add the parent directory to Python's search path
# This lets us import models.py from the root folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from models import (
    ResetRequest, ResetResponse,
    StepRequest, StepResponse,
    HealthResponse,
    StateResponse,
)
from server.environment import MLDebuggerEnvironment


# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Pipeline Debugger",
    description="An OpenEnv environment where an AI agent debugs broken deep learning pipelines.",
    version="1.0.0",
)

# CORS = Cross-Origin Resource Sharing
# This allows the HF Spaces frontend and external tools to call our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Allow any origin (fine for a hackathon)
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create one environment instance per server process
# (In production you'd have one per session, but this is fine for the hackathon)
env = MLDebuggerEnvironment()


# ─────────────────────────────────────────────────────────────
# ENDPOINT 1: Health Check
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """
    Simple liveness check.
    The hackathon validator pings /health to see if the server is running.
    Must return HTTP 200 with {"status": "ok"}.
    """
    return HealthResponse(status="ok", version="1.0.0")


# ─────────────────────────────────────────────────────────────
# ENDPOINT 2: Reset
# ─────────────────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResponse)
def reset(request: Optional[ResetRequest] = None):
    """
    Start a new episode.

    The agent sends which task and scenario to load.
    We return the initial observation + task description.

    Example request body:
      {"task_id": "task1_easy", "scenario_id": 1}

    If no body is sent, defaults to task1_easy, scenario 1.
    """
    if request is None:
        request = ResetRequest()
    try:
        response = env.reset(request)
        return response
    except (ValueError, FileNotFoundError) as e:
        # Return a 400 Bad Request if the task_id or scenario_id is invalid
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────
# ENDPOINT 3: Step
# ─────────────────────────────────────────────────────────────

@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Take one action in the current episode.

    The agent sends an action and gets back an observation + reward.

    Example request body:
      {
        "action": {
          "action_type": "fix_tensor_reshape",
          "parameters": {"layer_name": "linear1", "new_shape": 25088}
        }
      }
    """
    try:
        response = env.step(request.action)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# ENDPOINT 4: State
# ─────────────────────────────────────────────────────────────

@app.get("/state", response_model=StateResponse)
def state():
    """
    Return the current environment state.

    Required by OpenEnv spec: state() → current state.
    Returns episode metadata: task_id, step_count, done, metrics, etc.
    """
    return env.get_state()


# ─────────────────────────────────────────────────────────────
# SERVER ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    """
    Start the server. Called by the [project.scripts] entry point
    and by direct execution.
    """
    import uvicorn
    # Run the server on port 7860
    # 0.0.0.0 means "accept connections from any IP" (required for HF Spaces)
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()