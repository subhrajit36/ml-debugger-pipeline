---
title: ML Pipeline Debugger
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
---

# 🔧 ML Pipeline Debugger

**Team:** DOT-DOT | **Hackathon:** Meta PyTorch × Hugging Face | **Track:** OpenEnv

An agent environment where an AI acts as an ML engineer debugging broken deep learning pipelines.

---

## 🧠 What Is This?

A **real-world simulation** of ML debugging — the most time-consuming part of every ML engineer's job. The agent receives a broken training pipeline — complete with error logs, architecture summaries, and training metrics — and must diagnose and fix it using a structured action space.

**Why this matters:** ML engineers spend 30–50% of their time debugging pipelines. Training an agent to handle common failure modes (tensor shape mismatches, overfitting, IoU plateaus) has immediate practical value for the RL/agent community.

**Key design decision:** All responses are pre-computed from JSON scenario files. No real GPU training happens. This makes the environment:
- ⚡ Fast — instant feedback, no training wait
- 🎯 Deterministic — reproducible scores
- 💻 Lightweight — runs on 2 vCPU + 8GB RAM

---

## 🗂️ Project Structure

```
ml_pipeline_debugger/
├── scenarios/
│   ├── easy/             # Tensor shape mismatch (2 scenarios)
│   ├── medium/           # Overfitting remediation (2 scenarios)
│   └── hard/             # Segmentation IoU optimization (2 scenarios)
├── server/
│   ├── app.py            # FastAPI server (4 endpoints)
│   └── environment.py    # Core simulation logic
├── tests/
│   └── test_environment.py  # 29 tests covering all tasks + edge cases
├── models.py             # Pydantic Action/Observation/Reward types
├── inference.py          # Baseline LLM agent (hackathon required)
├── client.py             # Reusable Python HTTP client
├── openenv.yaml          # OpenEnv metadata & spec
├── Dockerfile            # Docker container config
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🎯 Tasks

### Task 1 — Easy: Fix Tensor Dimension Mismatch
- **Problem:** A CNN's Conv layers output the wrong shape for the Linear layer, crashing at forward pass
- **Correct action:** `fix_tensor_reshape` with the right flatten dimension (e.g., 25088 = 128 × 14 × 14)
- **Grader:** Binary — did the forward pass succeed? (0.0 or 1.0)
- **Max steps:** 5
- **Scenarios:** 2 (different architectures, different correct shapes)

### Task 2 — Medium: Fix Overfitting
- **Problem:** Train loss drops but val loss keeps rising (classic overfitting)
- **Correct actions:** `add_dropout` + `add_data_augmentation` (both required, any order)
- **Grader:** `val_loss_ratio` — proportional to how much val_loss drops toward target (0.0–1.0)
- **Max steps:** 10
- **Scenarios:** 2 (with different augmentation strategies)

### Task 3 — Hard: Optimize Segmentation IoU
- **Problem:** U-Net stuck at IoU=0.21, must reach IoU ≥ 0.55 within 5 epochs
- **Correct actions:** `change_loss_function` (Dice/Focal) + `add_lr_scheduler` (cosine)
- **Grader:** `iou_ratio` — `clamp(achieved_iou / target_iou, 0, 1)`
- **Max steps:** 15
- **Scenarios:** 2 (standard + class imbalance variant)

---

## 🔌 Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `fix_tensor_reshape` | `layer_name: str, new_shape: int` | Fix dimension mismatch in Linear layers |
| `modify_hyperparameters` | `lr: float, batch_size: int, epochs: int` | Change training hyperparameters |
| `add_data_augmentation` | `strategy: str` (`mild`, `aggressive`, `flip`, `crop`, `rotate`) | Apply data augmentation |
| `change_loss_function` | `name: str` (`CrossEntropy`, `Focal`, `Dice`) | Switch loss function |
| `add_dropout` | `layer_name: str, rate: float` (0.0–1.0) | Add dropout regularization |
| `add_lr_scheduler` | `type: str` (`cosine`, `step`, `plateau`), `params: dict` | Add learning rate scheduler |

---

## 👁️ Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `error_log` | `string` | Current error message (empty if no error) |
| `architecture_summary` | `string` | Text description of model layers |
| `data_shape` | `string` | Input tensor shape e.g. `[32, 3, 224, 224]` |
| `training_logs` | `list[string]` | Lines of training output |
| `current_metrics` | `dict[string, float]` | `train_loss`, `val_loss`, `iou`, `accuracy`, etc. |
| `done` | `boolean` | Whether the episode is finished |
| `reward` | `float` | Reward for the last action |
| `step` | `integer` | Current step number |
| `message` | `string` | Human-readable explanation of last action result |

---

## 🏆 Reward Function

The reward function provides **dense signal** throughout the episode — not just sparse end-of-episode scores:

| Event | Reward | Rationale |
|-------|--------|-----------|
| Correct action applied | +0.5 to +2.0 | Encourages right diagnosis |
| Task solved | +5.0 to +7.0 | Big bonus for full solution |
| Epoch completes without crash | +0.1 | Partial progress signal |
| Validation loss drops | +1.0 | Rewards metric improvement |
| Wrong action (no effect) | 0.0 | No penalty for exploration |
| Action causes NaN loss | -0.5 | Penalizes destructive actions |
| Ran out of steps | -1.0 | Penalizes inefficiency |

---

## 🚀 Setup & Running Locally

### 1. Clone and install

```bash
git clone https://huggingface.co/spaces/subhrajit36/ml-pipeline-debugger
cd ml-pipeline-debugger
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="google/gemma-4-31B-it:novita"
export HF_TOKEN="your-hf-token"
export OPENAI_API_KEY="your-api-key"
```

### 3. Start the environment server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 4. Run the baseline agent

```bash
python inference.py
```

### 5. Run tests

```bash
pip install pytest httpx
pytest tests/ -v
```

### 6. Test manually with curl

```bash
# Health check
curl http://localhost:7860/health

# Reset (start task 1)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_easy", "scenario_id": 1}'

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "fix_tensor_reshape", "parameters": {"layer_name": "linear1", "new_shape": 25088}}}'

# Check state
curl http://localhost:7860/state
```

---

## 🐳 Docker

```bash
docker build -t ml-pipeline-debugger .
docker run -p 7860:7860 ml-pipeline-debugger
```

The environment server starts automatically and listens on port 7860.

---

## 🌍 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint (e.g. `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Yes | Model identifier (e.g. `google/gemma-4-31B-it:novita`) |
| `HF_TOKEN` | Yes | Hugging Face API token |
| `OPENAI_API_KEY` | Yes | API key for the LLM provider |
| `ENV_BASE_URL` | No | URL of this environment server (default: `http://localhost:7860`) |

---

## 📊 Baseline Scores

Tested with `google/gemma-4-31B-it:novita` via Hugging Face Router:

| Task | Difficulty | Score | Steps Used |
|------|-----------|-------|------------|
| Task 1 — Fix Tensor Mismatch | Easy | **1.0** | 1 |
| Task 2 — Fix Overfitting | Medium | **1.0** | 2 |
| Task 3 — Optimize Segmentation | Hard | **1.0** | 2 |
| **Average** | | **1.0** | |

The baseline agent achieves perfect scores because the LLM can correctly read the error logs/metrics and select the right actions. The tasks are designed to test ML debugging knowledge, not trick the agent.

---

## 📝 OpenEnv Spec Compliance

- ✅ `openenv.yaml` with full metadata
- ✅ Typed Pydantic models (`Action`, `Observation`, `Reward`)
- ✅ `step()` → returns observation, reward, done, info
- ✅ `reset()` → returns initial observation
- ✅ `state()` → returns current state
- ✅ 3 tasks with difficulty progression (easy → medium → hard)
- ✅ Graders return scores in [0.0, 1.0]
- ✅ Dockerfile builds and runs cleanly
- ✅ Baseline inference script with reproducible scores
- ✅ Dense reward function with partial progress signals