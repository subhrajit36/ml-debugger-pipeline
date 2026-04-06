"""
inference.py
------------
Baseline agent for the ML Pipeline Debugger OpenEnv environment.

HACKATHON REQUIREMENTS:
  - Must be in the root directory
  - Must use OpenAI client
  - Must run on all 3 tasks and print scores
  - Must complete in under 20 minutes on 2 vCPU + 8GB RAM
  - stdout MUST emit ONLY [START], [STEP], [END] JSON lines

HOW IT WORKS:
  1. Connect to our running FastAPI environment server
  2. For each task: call /reset to get the initial observation
  3. Build a prompt with the observation
  4. Ask the LLM what action to take
  5. Parse the LLM's response into an Action
  6. Call /step with that action
  7. Repeat until done or max steps reached
  8. Print the final score
"""

import os
import sys
import json
import requests
from openai import OpenAI
from typing import List


# ─────────────────────────────────────────────────────────────
# STRUCTURED LOGGING — stdout is ONLY for [START], [STEP], [END]
# ─────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    """Emit [START] structured log to stdout."""
    print(f"[START] {json.dumps({'task': task, 'env': env, 'model': model})}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    """Emit [STEP] structured log to stdout."""
    print(f"[STEP] {json.dumps({'step': step, 'action': action, 'reward': reward, 'done': done, 'error': error})}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    """Emit [END] structured log to stdout."""
    print(f"[END] {json.dumps({'success': success, 'steps': steps, 'score': score, 'rewards': rewards})}", flush=True)


def debug(msg: str):
    """All non-structured output goes to stderr so it never pollutes stdout."""
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────
# CONFIGURATION — read from environment variables
# ─────────────────────────────────────────────────────────────

API_BASE_URL   = os.environ.get("API_BASE_URL",   "https://router.huggingface.co/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME",      "google/gemma-4-31B-it:novita")
HF_TOKEN       = os.environ.get("HF_TOKEN",        "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY",  "")

# Use OPENAI_API_KEY if set, otherwise fall back to HF_TOKEN
API_KEY = OPENAI_API_KEY or HF_TOKEN

# The URL of OUR environment server (running on HF Spaces or locally)
ENV_BASE_URL   = os.environ.get("ENV_BASE_URL",   "http://localhost:7860")

# Create the OpenAI client (works with any OpenAI-compatible API)
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT CLIENT HELPERS
# ─────────────────────────────────────────────────────────────

def env_reset(task_id: str, scenario_id: int = 1) -> dict:
    """Call /reset on our environment server."""
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "scenario_id": scenario_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action_type: str, parameters: dict) -> dict:
    """Call /step on our environment server."""
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": {"action_type": action_type, "parameters": parameters}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────
# LLM AGENT
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert ML engineer debugging broken deep learning pipelines.

You will receive an observation describing the current state of a broken training pipeline.
Your job is to choose the best action to fix it.

Available actions (use EXACTLY these names):
  fix_tensor_reshape       - params: {"layer_name": str, "new_shape": int}
  modify_hyperparameters   - params: {"lr": float, "batch_size": int, "epochs": int}
  add_data_augmentation    - params: {"strategy": "mild" | "aggressive" | "flip" | "crop" | "rotate"}
  change_loss_function     - params: {"name": "CrossEntropy" | "Focal" | "Dice"}
  add_dropout              - params: {"layer_name": str, "rate": float}
  add_lr_scheduler         - params: {"type": "cosine" | "step" | "plateau", "params": {}}

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{"action_type": "...", "parameters": {...}}
"""


def build_user_prompt(observation: dict, task_description: str, step: int, max_steps: int) -> str:
    """Build the prompt we send to the LLM for each step."""
    obs = observation["observation"]
    metrics_str = json.dumps(obs.get("current_metrics", {}), indent=2)
    logs_str = "\n".join(obs.get("training_logs", []))

    return f"""TASK: {task_description}
STEP: {step} / {max_steps}

ERROR LOG:
{obs.get("error_log") or "(no error)"}

ARCHITECTURE:
{obs.get("architecture_summary")}

DATA SHAPE: {obs.get("data_shape")}

TRAINING LOGS:
{logs_str}

CURRENT METRICS:
{metrics_str}

LAST MESSAGE: {obs.get("message", "")}

What action should you take next? Respond with ONLY the JSON action."""


def parse_llm_action(llm_response: str) -> dict:
    """
    Parse the LLM's text response into an action dict.
    If parsing fails, return a safe default action.
    """
    try:
        # Strip any accidental markdown fences
        clean = llm_response.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        debug(f"Could not parse LLM response as JSON: {llm_response[:200]}")
        return {"action_type": "modify_hyperparameters", "parameters": {"lr": 0.001, "batch_size": 32, "epochs": 10}}


# ─────────────────────────────────────────────────────────────
# SINGLE TASK RUNNER
# ─────────────────────────────────────────────────────────────

def run_task(task_id: str, scenario_id: int = 1) -> float:
    """
    Run one full episode of a task and return the final score.
    Uses conversation history so the LLM learns from previous step feedback.
    """
    # 1. Reset the environment
    reset_data = env_reset(task_id, scenario_id)
    task_description = reset_data["task_description"]
    max_steps = reset_data["max_steps"]
    current_obs = reset_data

    log_start(task=task_id, env="ml-pipeline-debugger", model=MODEL_NAME)

    final_score = 0.0
    steps_taken = 0
    success = False
    rewards = []

    # Conversation history for multi-step reasoning
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        # 2. Run the episode
        for step_num in range(1, max_steps + 1):
            steps_taken = step_num

            # Build prompt and add to history
            prompt = build_user_prompt(current_obs, task_description, step_num, max_steps)
            messages.append({"role": "user", "content": prompt})

            try:
                llm_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=200,
                )
                action_text = llm_response.choices[0].message.content
            except Exception as exc:
                debug(f"Model request failed: {exc}")
                action_text = '{"action_type": "modify_hyperparameters", "parameters": {"lr": 0.001, "batch_size": 32, "epochs": 10}}'

            # Add assistant response to history
            messages.append({"role": "assistant", "content": action_text})

            action = parse_llm_action(action_text)

            # Take the action in the environment
            step_data = env_step(action["action_type"], action.get("parameters", {}))
            obs = step_data["observation"]
            final_score = step_data["score"]
            reward = step_data.get("reward", obs.get("reward", 0))
            done = step_data.get("done", obs.get("done", False))
            error = obs.get("error_log", "") if obs.get("error_log") else None

            rewards.append(reward)

            # Wrap the step result as current_obs for next iteration
            current_obs = step_data

            log_step(
                step=step_num,
                action=json.dumps(action),
                reward=reward,
                done=done,
                error=error,
            )

            # Check if episode is done
            if done:
                success = final_score >= 0.5
                break

    except Exception as exc:
        debug(f"Task {task_id} failed with exception: {exc}")

    # Compute score: use the environment's score (grader-based, 0.0-1.0)
    score = max(0.0, min(1.0, final_score))
    if not success:
        success = score >= 0.5

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ─────────────────────────────────────────────────────────────
# MAIN — run all 3 tasks
# ─────────────────────────────────────────────────────────────

def main():
    debug("ML Pipeline Debugger - Baseline Agent")
    debug(f"Using model: {MODEL_NAME}")
    debug(f"Environment: {ENV_BASE_URL}")

    # Check that the environment server is up
    try:
        health = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        health.raise_for_status()
        debug("Server: online")
    except Exception as e:
        debug(f"Server: not reachable - {e}")
        debug("Make sure to start the server first: uvicorn server.app:app --port 7860")
        return

    scores = {}

    # Task 1: Easy - Fix Tensor Dimension Mismatch
    scores["task1_easy"] = run_task("task1_easy", scenario_id=1)

    # Task 2: Medium - Fix Overfitting
    scores["task2_medium"] = run_task("task2_medium", scenario_id=1)

    # Task 3: Hard - Optimize Segmentation IoU
    scores["task3_hard"] = run_task("task3_hard", scenario_id=1)

    # Summary to stderr (not stdout)
    debug("=" * 60)
    debug("RESULTS SUMMARY")
    debug("=" * 60)
    for task, score in scores.items():
        debug(f"  {task:<20} {score:.3f}")
    avg = sum(scores.values()) / len(scores)
    debug(f"  Average Score: {avg:.3f}")
    debug("=" * 60)


if __name__ == "__main__":
    main()