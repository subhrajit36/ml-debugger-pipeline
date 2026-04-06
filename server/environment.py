"""
environment.py
--------------
The core of the simulation.

HOW IT WORKS:
1. On /reset  → loads a scenario JSON file into memory as the "current episode"
2. On /step   → looks at the agent's action, finds the right response in the JSON,
                updates state, calculates reward, returns Observation

KEY CONCEPTS:
  Episode  = one run of a task from start to finish
  Step     = one action taken by the agent
  Scenario = a JSON file describing a broken ML pipeline and all possible outcomes
"""

import json
import os
from pathlib import Path
from models import Action, Observation, ResetRequest, ResetResponse, StepResponse, StateResponse


# ─────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────

# __file__ is the path to this script itself
# .parent gives the directory containing it
# We find the scenarios folder relative to this file
BASE_DIR = Path(__file__).parent
SCENARIOS_DIR = BASE_DIR.parent / "scenarios"

# Map task_id strings to scenario folder names
TASK_TO_FOLDER = {
    "task1_easy":   "easy",
    "task2_medium": "medium",
    "task3_hard":   "hard",
}


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT CLASS
# ─────────────────────────────────────────────────────────────

class MLDebuggerEnvironment:
    """
    Manages a single episode of the ML Pipeline Debugger.

    An "episode" is one attempt to solve a task.
    It starts when reset() is called and ends when done=True.
    """

    def __init__(self):
        # These will be set when reset() is called
        self.scenario: dict = {}          # the loaded JSON scenario
        self.current_state: dict = {}     # mutable copy of the current observation state
        self.step_count: int = 0          # how many steps have been taken
        self.done: bool = False           # is the episode finished?
        self.total_reward: float = 0.0    # cumulative reward so far

        # Track what the agent has done (for combo actions like task2)
        self.actions_taken: set = set()

    # ─────────────────────────────────────────────
    # RESET — start a new episode
    # ─────────────────────────────────────────────

    def reset(self, request: ResetRequest) -> ResetResponse:
        """
        Load a scenario and return the initial observation.
        Called when the agent wants to start fresh.
        """
        folder = TASK_TO_FOLDER.get(request.task_id)
        if not folder:
            raise ValueError(f"Unknown task_id: {request.task_id}. Choose from: {list(TASK_TO_FOLDER.keys())}")

        scenario_path = SCENARIOS_DIR / folder / f"scenario_{request.scenario_id}.json"

        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

        # Load the JSON file into a Python dict
        with open(scenario_path, "r", encoding="utf-8") as f:
            self.scenario = json.load(f)

        # Reset episode state
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.actions_taken = set()

        # Deep copy the initial state so we can mutate it without touching the scenario
        init = self.scenario["initial_state"]
        self.current_state = {
            "error_log":             init.get("error_log", ""),
            "architecture_summary":  init.get("architecture_summary", ""),
            "data_shape":            init.get("data_shape", ""),
            "training_logs":         list(init.get("training_logs", [])),
            "current_metrics":       dict(init.get("current_metrics", {})),
        }

        # Build the initial Observation object
        obs = Observation(
            error_log=            self.current_state["error_log"],
            architecture_summary= self.current_state["architecture_summary"],
            data_shape=           self.current_state["data_shape"],
            training_logs=        self.current_state["training_logs"],
            current_metrics=      self.current_state["current_metrics"],
            done=False,
            reward=0.0,
            step=0,
            message=f"Episode started. Task: {self.scenario['title']}",
        )

        return ResetResponse(
            observation=obs,
            task_description=self.scenario["description"],
            max_steps=self.scenario["max_steps"],
        )

    # ─────────────────────────────────────────────
    # STEP — process one action
    # ─────────────────────────────────────────────

    def step(self, action: Action) -> StepResponse:
        """
        Take one action and return the resulting observation + reward.
        """
        if self.done:
            # Episode is over — tell the agent to call reset()
            obs = self._make_observation(
                reward=0.0,
                message="Episode already done. Call /reset to start a new episode.",
            )
            return StepResponse(observation=obs, reward=0.0, done=True, score=0.0)

        self.step_count += 1
        max_steps = self.scenario["max_steps"]

        # Check if we've run out of steps
        if self.step_count > max_steps:
            self.done = True
            obs = self._make_observation(
                reward=-1.0,
                message=f"❌ Ran out of steps ({max_steps} max). Episode failed.",
            )
            return StepResponse(observation=obs, reward=-1.0, done=True, score=self._calculate_score())

        # Route to the correct task handler
        task_id = self.scenario["task_id"]

        if task_id == "task1_easy":
            return self._handle_task1(action)
        elif task_id == "task2_medium":
            return self._handle_task2(action)
        elif task_id == "task3_hard":
            return self._handle_task3(action)
        else:
            return self._unknown_action_response(action)

    # ─────────────────────────────────────────────
    # TASK 1 HANDLER — Dimension Mismatch
    # ─────────────────────────────────────────────

    def _handle_task1(self, action: Action) -> StepResponse:
        """
        Task 1: The agent must call fix_tensor_reshape with the correct shape (25088 for scenario_1).
        """
        responses = self.scenario["responses"]

        if action.action_type == "fix_tensor_reshape":
            # Check if the agent provided the right shape value
            new_shape = action.parameters.get("new_shape")
            scenario_id = self.scenario["scenario_id"]

            # Scenario 1: correct shape is 25088 (128 channels * 14*14 spatial)
            # Scenario 2: correct value is 3 (RGB channels)
            correct_values = {1: 25088, 2: 3}
            correct = correct_values.get(scenario_id, 25088)

            if new_shape == correct:
                resp = responses["fix_tensor_reshape_correct"]
            else:
                resp = responses["fix_tensor_reshape_wrong_shape"]
        else:
            # Agent used a non-fix action — look it up or fall back to default
            resp = responses.get(action.action_type, responses["default"])

        return self._apply_response(resp)

    # ─────────────────────────────────────────────
    # TASK 2 HANDLER — Overfitting
    # ─────────────────────────────────────────────

    def _handle_task2(self, action: Action) -> StepResponse:
        """
        Task 2: The agent must apply BOTH dropout and augmentation.
        We track what the agent has done and check for combos.
        """
        responses = self.scenario["responses"]

        # Record this action
        self.actions_taken.add(action.action_type)

        has_dropout     = "add_dropout"           in self.actions_taken
        has_augmentation = "add_data_augmentation" in self.actions_taken

        # If agent has now done both → success
        if has_dropout and has_augmentation:
            resp = responses.get("both_dropout_and_augmentation", responses["default"])
            self.done = True

        elif action.action_type == "add_dropout" and not has_augmentation:
            resp = responses.get("add_dropout_only", responses["default"])

        elif action.action_type == "add_data_augmentation" and not has_dropout:
            strategy = action.parameters.get("strategy", "mild")
            scenario_id = self.scenario["scenario_id"]

            if scenario_id == 2:
                # scenario 2 differentiates mild vs aggressive
                if strategy == "aggressive":
                    resp = responses.get("add_augmentation_aggressive", responses["default"])
                else:
                    resp = responses.get("add_augmentation_mild", responses["default"])
            else:
                resp = responses.get("add_augmentation_only", responses["default"])

        elif action.action_type == "modify_hyperparameters":
            lr = action.parameters.get("lr", 0.001)
            if lr > 0.01:
                resp = responses.get("modify_hyperparameters_lr_high", responses["default"])
            else:
                resp = responses.get("modify_hyperparameters_lr_low", responses["default"])

        elif action.action_type == "add_lr_scheduler" and self.scenario["scenario_id"] == 2:
            # scenario 2 accepts scheduler as a correct action
            if has_augmentation:
                resp = responses.get("both_augmentation_and_scheduler", responses["default"])
                self.done = True
            else:
                resp = responses.get("add_lr_scheduler_only", responses["default"])

        else:
            resp = responses.get(action.action_type, responses["default"])

        return self._apply_response(resp)

    # ─────────────────────────────────────────────
    # TASK 3 HANDLER — Segmentation IoU
    # ─────────────────────────────────────────────

    def _handle_task3(self, action: Action) -> StepResponse:
        """
        Task 3: The agent must combine a good loss function AND a good LR scheduler.
        We track what the agent has applied and trigger success on the right combo.
        """
        responses = self.scenario["responses"]

        self.actions_taken.add(action.action_type)

        has_scheduler   = "add_lr_scheduler"    in self.actions_taken
        has_good_loss   = "change_loss_function" in self.actions_taken

        loss_name     = action.parameters.get("name", "").lower()
        scheduler_type = action.parameters.get("type", "").lower()

        # Always record the current action's parameters BEFORE evaluating combos
        if action.action_type == "change_loss_function":
            self._last_loss = loss_name
        elif action.action_type == "add_lr_scheduler":
            self._last_scheduler = scheduler_type

        # Determine if it's a good combo
        if has_scheduler and has_good_loss:
            prev_loss = self._get_last_loss_function()
            prev_sched = self._get_last_scheduler()

            if prev_loss in ("dice", "focal") and prev_sched == "cosine":
                if prev_loss == "dice":
                    resp = responses.get("dice_and_cosine", responses["default"])
                else:
                    resp = responses.get("focal_and_cosine", responses["default"])
                self.done = True

            elif prev_loss in ("dice", "focal") and prev_sched == "step":
                resp = responses.get("dice_and_step_scheduler", responses["default"])

            elif prev_loss == "crossentropy":
                resp = responses.get("crossentropy_and_scheduler", responses["default"])
            else:
                resp = responses["default"]

        elif action.action_type == "change_loss_function":
            if loss_name in ("dice",):
                resp = responses.get("dice_loss_only", responses["default"])
            elif loss_name in ("focal",):
                resp = responses.get("focal_loss_only", responses["default"])
            else:
                resp = responses.get("crossentropy_and_scheduler", responses["default"])

        elif action.action_type == "add_lr_scheduler":
            resp = responses.get("cosine_scheduler_only", responses["default"])

        else:
            resp = responses.get(action.action_type, responses["default"])

        return self._apply_response(resp)


    # ─────────────────────────────────────────────
    # HELPER: get last used loss/scheduler name
    # ─────────────────────────────────────────────

    def _get_last_loss_function(self) -> str:
        return getattr(self, "_last_loss", "crossentropy")

    def _get_last_scheduler(self) -> str:
        return getattr(self, "_last_scheduler", "step")

    # ─────────────────────────────────────────────
    # HELPER: apply a JSON response to current state
    # ─────────────────────────────────────────────

    def _apply_response(self, resp: dict) -> StepResponse:
        """
        Takes a response dict from the scenario JSON and:
        1. Updates current state (metrics, logs, done flag)
        2. Accumulates reward
        3. Returns a StepResponse
        """
        reward = resp.get("reward", 0.0)
        self.total_reward += reward

        # Update metrics and logs from the response
        if "metrics" in resp:
            self.current_state["current_metrics"].update(resp["metrics"])
        if "training_logs" in resp:
            self.current_state["training_logs"] = resp["training_logs"]

        # Update done flag if response says so
        if resp.get("done", False):
            self.done = True
        if resp.get("result") in ("success",):
            self.done = True

        obs = self._make_observation(reward=reward, message=resp.get("message", ""))
        score = self._calculate_score()

        return StepResponse(observation=obs, reward=reward, done=self.done, score=score)

    # ─────────────────────────────────────────────
    # HELPER: build an Observation from current state
    # ─────────────────────────────────────────────

    def _make_observation(self, reward: float, message: str) -> Observation:
        return Observation(
            error_log=            self.current_state.get("error_log", ""),
            architecture_summary= self.current_state.get("architecture_summary", ""),
            data_shape=           self.current_state.get("data_shape", ""),
            training_logs=        self.current_state.get("training_logs", []),
            current_metrics=      self.current_state.get("current_metrics", {}),
            done=                 self.done,
            reward=               reward,
            step=                 self.step_count,
            message=              message,
        )

    # ─────────────────────────────────────────────
    # HELPER: calculate final score (0.0 → 1.0)
    # ─────────────────────────────────────────────

    def _calculate_score(self) -> float:
        """
        Returns a score between 0.0 and 1.0 based on the task.
        """
        task_id = self.scenario["task_id"]
        metrics = self.current_state["current_metrics"]

        if task_id == "task1_easy":
            # Binary: did the forward pass succeed?
            return float(metrics.get("forward_pass_success", 0.0))

        elif task_id == "task2_medium":
            # Proportional: how much progress was made from initial towards the target val loss?
            target = self.scenario.get("target_val_loss", 0.8)
            current = metrics.get("val_loss", 999.0)
            initial = self.scenario.get("initial_state", {}).get("current_metrics", {}).get("val_loss", target + 1.0)
            
            if current <= target:
                return 1.0
            
            if initial <= target:
                return 1.0
                
            score = (initial - current) / (initial - target)
            return max(0.0, min(1.0, score))  # clamp to [0, 1]

        elif task_id == "task3_hard":
            # Proportional: how close to target IoU?
            target = self.scenario.get("target_iou", 0.55)
            current = metrics.get("iou", 0.0)
            score = current / target
            return max(0.0, min(1.0, score))

        return 0.0

    # ─────────────────────────────────────────────
    # HELPER: fallback for unknown actions
    # ─────────────────────────────────────────────

    def _unknown_action_response(self, action: Action) -> StepResponse:
        obs = self._make_observation(
            reward=0.0,
            message=f"Unknown action: '{action.action_type}'. Check the action space in README.",
        )
        return StepResponse(observation=obs, reward=0.0, done=self.done, score=0.0)

    # ─────────────────────────────────────────────
    # STATE — return current environment state
    # ─────────────────────────────────────────────

    def get_state(self) -> StateResponse:
        """
        Return the current environment state.
        Required by OpenEnv spec: state() → current state.
        """
        return StateResponse(
            task_id=self.scenario.get("task_id", ""),
            scenario_id=self.scenario.get("scenario_id", 0),
            step_count=self.step_count,
            max_steps=self.scenario.get("max_steps", 0),
            done=self.done,
            total_reward=self.total_reward,
            actions_taken=list(self.actions_taken),
            current_metrics=dict(self.current_state.get("current_metrics", {})),
            episode_active=bool(self.scenario),
        )