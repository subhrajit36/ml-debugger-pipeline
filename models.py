

"""
models.py
---------
Defines the typed data models for our environment.

WHY WE NEED THIS:
- The agent sends "actions" to the environment
- The environment sends back "observations"
- Using Pydantic models means Python will automatically validate the data
  and give clear error messages if something is wrong.

Pydantic = a Python library that adds type checking at runtime.
"""

from pydantic import BaseModel
from typing import Dict, List, Optional, Any


# ─────────────────────────────────────────────
# ACTION MODELS  (Agent → Environment)
# ─────────────────────────────────────────────

class Action(BaseModel):
    """
    Represents one action the agent wants to take.

    Fields:
      action_type : which action to perform (e.g. "fix_tensor_reshape")
      parameters  : a dictionary of extra info needed for that action
                    e.g. {"layer_name": "linear1", "new_shape": 25088}
    """
    action_type: str
    parameters: Dict[str, Any] = {}


# ─────────────────────────────────────────────
# REWARD MODEL  (typed reward per OpenEnv spec)
# ─────────────────────────────────────────────

class Reward(BaseModel):
    """
    Typed reward model required by OpenEnv spec.

    Fields:
      value     : the numeric reward value
      reason    : human-readable explanation of why this reward was given
      breakdown : optional dict mapping reward components to their values
    """
    value: float = 0.0
    reason: str = ""
    breakdown: Dict[str, float] = {}


# ─────────────────────────────────────────────
# OBSERVATION MODELS  (Environment → Agent)
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """
    Represents what the agent sees after each step.

    Fields:
      error_log            : the current error message (empty string if no error)
      architecture_summary : a text description of the model layers
      data_shape           : shape of the input tensor as a string
      training_logs        : list of log lines from training
      current_metrics      : dict of metric name → value
      done                 : True if the task is finished
      reward               : the reward the agent earned for its last action
      step                 : which step number we are on
      message              : a human-readable explanation of what happened
    """
    error_log: str = ""
    architecture_summary: str = ""
    data_shape: str = ""
    training_logs: List[str] = []
    current_metrics: Dict[str, float] = {}
    done: bool = False
    reward: float = 0.0
    step: int = 0
    message: str = ""


# ─────────────────────────────────────────────
# RESET REQUEST / RESPONSE
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    """
    Sent to /reset to start a new episode.

    Fields:
      task_id     : which task to load  ("task1_easy", "task2_medium", "task3_hard")
      scenario_id : which scenario file within that task (default 1)
    """
    task_id: str = "task1_easy"
    scenario_id: int = 1


class ResetResponse(BaseModel):
    """
    The environment's reply to a reset — the initial observation.
    """
    observation: Observation
    task_description: str
    max_steps: int


# ─────────────────────────────────────────────
# STEP REQUEST / RESPONSE
# ─────────────────────────────────────────────

class StepRequest(BaseModel):
    """
    Sent to /step to take one action.
    """
    action: Action


class StepResponse(BaseModel):
    """
    The environment's reply after processing one action.

    OpenEnv spec requires top-level: observation, reward, done, info
    """
    observation: Observation
    reward: float = 0.0
    done: bool = False
    score: float = 0.0
    info: Dict[str, Any] = {}


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


# ─────────────────────────────────────────────
# STATE RESPONSE  (OpenEnv spec: /state)
# ─────────────────────────────────────────────

class StateResponse(BaseModel):
    """
    Returns current environment state.
    Required by OpenEnv spec: state() → current state.
    """
    task_id: str = ""
    scenario_id: int = 0
    step_count: int = 0
    max_steps: int = 0
    done: bool = False
    total_reward: float = 0.0
    actions_taken: List[str] = []
    current_metrics: Dict[str, float] = {}
    episode_active: bool = False