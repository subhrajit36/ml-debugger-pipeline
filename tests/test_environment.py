"""
test_environment.py
-------------------
Tests for the ML Pipeline Debugger environment.

Runs entirely in-process using FastAPI's TestClient — no need to start
the server separately.

Usage:
    pip install pytest httpx
    pytest tests/ -v
"""

import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from server.app import app


@pytest.fixture
def client():
    """Create a test client for our FastAPI app."""
    return TestClient(app)


# ─────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"


# ─────────────────────────────────────────────────────────────
# Reset Endpoint
# ─────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_task1_scenario1(self, client):
        resp = client.post("/reset", json={"task_id": "task1_easy", "scenario_id": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert data["max_steps"] == 5
        assert "RuntimeError" in data["observation"]["error_log"]

    def test_reset_task1_scenario2(self, client):
        resp = client.post("/reset", json={"task_id": "task1_easy", "scenario_id": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_steps"] == 5
        assert "channels" in data["observation"]["error_log"].lower()

    def test_reset_task2_scenario1(self, client):
        resp = client.post("/reset", json={"task_id": "task2_medium", "scenario_id": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_steps"] == 10
        assert data["observation"]["error_log"] == ""  # no crash, overfitting

    def test_reset_task2_scenario2(self, client):
        resp = client.post("/reset", json={"task_id": "task2_medium", "scenario_id": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_steps"] == 10

    def test_reset_task3_scenario1(self, client):
        resp = client.post("/reset", json={"task_id": "task3_hard", "scenario_id": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_steps"] == 15
        assert data["observation"]["current_metrics"]["iou"] == 0.21

    def test_reset_invalid_task(self, client):
        resp = client.post("/reset", json={"task_id": "nonexistent", "scenario_id": 1})
        assert resp.status_code == 400

    def test_reset_invalid_scenario(self, client):
        resp = client.post("/reset", json={"task_id": "task1_easy", "scenario_id": 999})
        assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────
# Task 1 — Fix Dimension Mismatch
# ─────────────────────────────────────────────────────────────

class TestTask1:
    def test_correct_reshape_scenario1(self, client):
        """Scenario 1: correct shape is 25088."""
        client.post("/reset", json={"task_id": "task1_easy", "scenario_id": 1})
        resp = client.post("/step", json={
            "action": {
                "action_type": "fix_tensor_reshape",
                "parameters": {"layer_name": "linear1", "new_shape": 25088}
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation"]["done"] is True
        assert data["score"] == 1.0
        assert data["observation"]["reward"] > 0

    def test_wrong_reshape_scenario1(self, client):
        """Scenario 1: wrong shape value."""
        client.post("/reset", json={"task_id": "task1_easy", "scenario_id": 1})
        resp = client.post("/step", json={
            "action": {
                "action_type": "fix_tensor_reshape",
                "parameters": {"layer_name": "linear1", "new_shape": 512}
            }
        })
        data = resp.json()
        assert data["observation"]["done"] is False
        assert data["score"] == 0.0

    def test_correct_reshape_scenario2(self, client):
        """Scenario 2: correct value is 3 (RGB channels)."""
        client.post("/reset", json={"task_id": "task1_easy", "scenario_id": 2})
        resp = client.post("/step", json={
            "action": {
                "action_type": "fix_tensor_reshape",
                "parameters": {"layer_name": "conv1", "new_shape": 3}
            }
        })
        data = resp.json()
        assert data["observation"]["done"] is True
        assert data["score"] == 1.0

    def test_irrelevant_action_task1(self, client):
        """Using augmentation on a shape mismatch should do nothing."""
        client.post("/reset", json={"task_id": "task1_easy", "scenario_id": 1})
        resp = client.post("/step", json={
            "action": {
                "action_type": "add_data_augmentation",
                "parameters": {"strategy": "mild"}
            }
        })
        data = resp.json()
        assert data["observation"]["done"] is False
        assert data["observation"]["reward"] == 0.0


# ─────────────────────────────────────────────────────────────
# Task 2 — Fix Overfitting
# ─────────────────────────────────────────────────────────────

class TestTask2:
    def test_dropout_then_augmentation(self, client):
        """Applying both dropout + augmentation should solve it."""
        client.post("/reset", json={"task_id": "task2_medium", "scenario_id": 1})

        # Step 1: add dropout
        resp = client.post("/step", json={
            "action": {
                "action_type": "add_dropout",
                "parameters": {"layer_name": "fc1", "rate": 0.5}
            }
        })
        data = resp.json()
        assert data["observation"]["done"] is False  # partial

        # Step 2: add augmentation → completes the combo
        resp = client.post("/step", json={
            "action": {
                "action_type": "add_data_augmentation",
                "parameters": {"strategy": "mild"}
            }
        })
        data = resp.json()
        assert data["observation"]["done"] is True
        assert data["score"] > 0

    def test_augmentation_then_dropout(self, client):
        """Order shouldn't matter — augmentation first, then dropout."""
        client.post("/reset", json={"task_id": "task2_medium", "scenario_id": 1})

        # Step 1: augmentation
        client.post("/step", json={
            "action": {
                "action_type": "add_data_augmentation",
                "parameters": {"strategy": "flip"}
            }
        })

        # Step 2: dropout → combo complete
        resp = client.post("/step", json={
            "action": {
                "action_type": "add_dropout",
                "parameters": {"layer_name": "fc1", "rate": 0.3}
            }
        })
        data = resp.json()
        assert data["observation"]["done"] is True

    def test_high_lr_causes_nan(self, client):
        """High learning rate should be penalized."""
        client.post("/reset", json={"task_id": "task2_medium", "scenario_id": 1})
        resp = client.post("/step", json={
            "action": {
                "action_type": "modify_hyperparameters",
                "parameters": {"lr": 0.1, "batch_size": 32, "epochs": 10}
            }
        })
        data = resp.json()
        assert data["observation"]["reward"] < 0


# ─────────────────────────────────────────────────────────────
# Task 3 — Segmentation IoU
# ─────────────────────────────────────────────────────────────

class TestTask3:
    def test_dice_then_cosine_scheduler(self, client):
        """Dice loss + cosine scheduler should solve it."""
        client.post("/reset", json={"task_id": "task3_hard", "scenario_id": 1})

        # Step 1: change loss to Dice
        resp = client.post("/step", json={
            "action": {
                "action_type": "change_loss_function",
                "parameters": {"name": "Dice"}
            }
        })
        data = resp.json()
        assert data["observation"]["done"] is False

        # Step 2: add cosine scheduler → success
        resp = client.post("/step", json={
            "action": {
                "action_type": "add_lr_scheduler",
                "parameters": {"type": "cosine", "params": {}}
            }
        })
        data = resp.json()
        assert data["observation"]["done"] is True
        assert data["score"] >= 1.0

    def test_focal_then_cosine_scheduler(self, client):
        """Focal loss + cosine scheduler should also solve it."""
        client.post("/reset", json={"task_id": "task3_hard", "scenario_id": 1})

        client.post("/step", json={
            "action": {
                "action_type": "change_loss_function",
                "parameters": {"name": "Focal"}
            }
        })

        resp = client.post("/step", json={
            "action": {
                "action_type": "add_lr_scheduler",
                "parameters": {"type": "cosine", "params": {}}
            }
        })
        data = resp.json()
        assert data["observation"]["done"] is True
        assert data["observation"]["current_metrics"]["iou"] >= 0.55

    def test_crossentropy_not_enough(self, client):
        """CrossEntropy + scheduler shouldn't reach the target."""
        client.post("/reset", json={"task_id": "task3_hard", "scenario_id": 1})

        client.post("/step", json={
            "action": {
                "action_type": "change_loss_function",
                "parameters": {"name": "CrossEntropy"}
            }
        })

        resp = client.post("/step", json={
            "action": {
                "action_type": "add_lr_scheduler",
                "parameters": {"type": "cosine", "params": {}}
            }
        })
        data = resp.json()
        assert data["observation"]["done"] is False

    def test_step_scheduler_not_enough(self, client):
        """Dice + StepLR should get close but not solve it."""
        client.post("/reset", json={"task_id": "task3_hard", "scenario_id": 1})

        client.post("/step", json={
            "action": {
                "action_type": "change_loss_function",
                "parameters": {"name": "Dice"}
            }
        })

        resp = client.post("/step", json={
            "action": {
                "action_type": "add_lr_scheduler",
                "parameters": {"type": "step", "params": {}}
            }
        })
        data = resp.json()
        assert data["observation"]["done"] is False


# ─────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_step_after_done(self, client):
        """Stepping after an episode is done should return a message."""
        client.post("/reset", json={"task_id": "task1_easy", "scenario_id": 1})
        # Solve it
        client.post("/step", json={
            "action": {
                "action_type": "fix_tensor_reshape",
                "parameters": {"layer_name": "linear1", "new_shape": 25088}
            }
        })
        # Try again after done
        resp = client.post("/step", json={
            "action": {
                "action_type": "fix_tensor_reshape",
                "parameters": {"layer_name": "linear1", "new_shape": 25088}
            }
        })
        data = resp.json()
        assert "reset" in data["observation"]["message"].lower() or "done" in data["observation"]["message"].lower()

    def test_max_steps_exceeded(self, client):
        """Running out of steps should end the episode."""
        client.post("/reset", json={"task_id": "task1_easy", "scenario_id": 1})
        # Use 5 wrong actions (max_steps = 5)
        for _ in range(5):
            client.post("/step", json={
                "action": {
                    "action_type": "modify_hyperparameters",
                    "parameters": {"lr": 0.001}
                }
            })
        # 6th step → over the limit
        resp = client.post("/step", json={
            "action": {
                "action_type": "fix_tensor_reshape",
                "parameters": {"layer_name": "linear1", "new_shape": 25088}
            }
        })
        data = resp.json()
        assert data["observation"]["done"] is True


# ─────────────────────────────────────────────────────────────
# State Endpoint (OpenEnv spec)
# ─────────────────────────────────────────────────────────────

class TestState:
    def test_state_before_reset(self, client):
        """State before any reset should show no active episode."""
        # Force a fresh state by resetting and completing
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "step_count" in data
        assert "done" in data
        assert "total_reward" in data

    def test_state_after_reset(self, client):
        """State after reset should show active episode."""
        client.post("/reset", json={"task_id": "task1_easy", "scenario_id": 1})
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "task1_easy"
        assert data["scenario_id"] == 1
        assert data["step_count"] == 0
        assert data["max_steps"] == 5
        assert data["done"] is False
        assert data["episode_active"] is True

    def test_state_after_step(self, client):
        """State after a step should update step_count."""
        client.post("/reset", json={"task_id": "task1_easy", "scenario_id": 1})
        client.post("/step", json={
            "action": {
                "action_type": "modify_hyperparameters",
                "parameters": {"lr": 0.001}
            }
        })
        resp = client.get("/state")
        data = resp.json()
        assert data["step_count"] == 1
        assert data["done"] is False

    def test_state_after_done(self, client):
        """State after solving should show done."""
        client.post("/reset", json={"task_id": "task1_easy", "scenario_id": 1})
        client.post("/step", json={
            "action": {
                "action_type": "fix_tensor_reshape",
                "parameters": {"layer_name": "linear1", "new_shape": 25088}
            }
        })
        resp = client.get("/state")
        data = resp.json()
        assert data["done"] is True
        assert data["total_reward"] > 0


# ─────────────────────────────────────────────────────────────
# Top-level reward/done in StepResponse (OpenEnv spec)
# ─────────────────────────────────────────────────────────────

class TestStepResponseFormat:
    def test_step_has_top_level_reward(self, client):
        """StepResponse must have top-level reward field."""
        client.post("/reset", json={"task_id": "task1_easy", "scenario_id": 1})
        resp = client.post("/step", json={
            "action": {
                "action_type": "fix_tensor_reshape",
                "parameters": {"layer_name": "linear1", "new_shape": 25088}
            }
        })
        data = resp.json()
        assert "reward" in data, "StepResponse missing top-level 'reward'"
        assert "done" in data, "StepResponse missing top-level 'done'"
        assert "score" in data, "StepResponse missing 'score'"
        assert "observation" in data
        assert data["reward"] > 0
        assert data["done"] is True

    def test_top_level_done_matches_observation(self, client):
        """Top-level done should match observation.done."""
        client.post("/reset", json={"task_id": "task2_medium", "scenario_id": 1})
        resp = client.post("/step", json={
            "action": {
                "action_type": "add_dropout",
                "parameters": {"layer_name": "fc1", "rate": 0.5}
            }
        })
        data = resp.json()
        assert data["done"] == data["observation"]["done"]


# ─────────────────────────────────────────────────────────────
# Hard Scenario 2
# ─────────────────────────────────────────────────────────────

class TestHardScenario2:
    def test_reset_hard_scenario2(self, client):
        """Scenario 2 should load with class imbalance context."""
        resp = client.post("/reset", json={"task_id": "task3_hard", "scenario_id": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_steps"] == 15
        assert data["observation"]["current_metrics"]["iou"] == 0.18

    def test_dice_cosine_solves_scenario2(self, client):
        """Dice + cosine should also solve the class imbalance scenario."""
        client.post("/reset", json={"task_id": "task3_hard", "scenario_id": 2})

        client.post("/step", json={
            "action": {
                "action_type": "change_loss_function",
                "parameters": {"name": "Dice"}
            }
        })

        resp = client.post("/step", json={
            "action": {
                "action_type": "add_lr_scheduler",
                "parameters": {"type": "cosine", "params": {}}
            }
        })
        data = resp.json()
        assert data["done"] is True
        assert data["observation"]["current_metrics"]["iou"] >= 0.55
