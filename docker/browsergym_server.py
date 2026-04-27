"""
BrowserGym OpenEnv Server

Exposes the BrowserGym MiniWoB++ environment as an HTTP API
that the GRPO trainer can connect to.

This runs inside the browsergym-env Docker container (CPU only).
The trainer container connects to this over the Docker network.
"""

import json
import logging
import os
import traceback
from typing import Optional

import browsergym.miniwob  # registers MiniWoB environments
import gymnasium as gym
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BrowserGym OpenEnv Server")

# Global environment reference
env: Optional[gym.Env] = None
current_obs = None
current_info = None


class ResetRequest(BaseModel):
    task_name: str = "browsergym/miniwob.click-test"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_str: str


class ObservationResponse(BaseModel):
    goal: Optional[str] = None
    axtree_txt: Optional[str] = None
    screenshot: Optional[list] = None  # Flattened pixel array
    error: Optional[str] = None
    last_action_error: bool = False


class ResetResponse(BaseModel):
    observation: ObservationResponse
    done: bool = False


class StepResponse(BaseModel):
    observation: ObservationResponse
    reward: float = 0.0
    done: bool = False


def extract_observation(obs: dict, info: dict) -> ObservationResponse:
    """Extract relevant fields from BrowserGym observation."""
    goal = obs.get("goal", "")
    axtree_txt = obs.get("axtree_txt", "")
    last_action_error = bool(obs.get("last_action_error", False))
    error = str(obs.get("last_action_error", "")) if last_action_error else None

    # Screenshot as list (optional — skip for text-only mode to save bandwidth)
    screenshot = None
    if "screenshot" in obs and obs["screenshot"] is not None:
        try:
            screenshot = obs["screenshot"].tolist()
        except Exception:
            screenshot = None

    return ObservationResponse(
        goal=goal,
        axtree_txt=axtree_txt,
        screenshot=screenshot,
        error=error,
        last_action_error=last_action_error,
    )


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "task": "browsergym/miniwob.click-test"}


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest):
    """Reset the BrowserGym environment and return initial observation."""
    global env, current_obs, current_info

    try:
        # Close existing environment if any
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        logger.info(f"Creating environment: {request.task_name}")

        # Create BrowserGym environment
        env = gym.make(
            request.task_name,
            headless=True,
            slow_mo=0,
        )

        # Reset and get initial observation
        seed = request.seed or int.from_bytes(os.urandom(4), "big") % (2**31)
        obs, info = env.reset(seed=seed)
        current_obs = obs
        current_info = info

        observation = extract_observation(obs, info)

        logger.info(f"Environment reset. Goal: {observation.goal}")

        return ResetResponse(observation=observation, done=False)

    except Exception as e:
        logger.error(f"Reset failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Execute an action in the BrowserGym environment."""
    global env, current_obs, current_info

    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
        logger.info(f"Executing action: {request.action_str}")

        # Execute the action
        obs, reward, terminated, truncated, info = env.step(request.action_str)
        current_obs = obs
        current_info = info

        done = terminated or truncated
        observation = extract_observation(obs, info)

        logger.info(f"Step result: reward={reward}, done={done}")

        return StepResponse(
            observation=observation,
            reward=float(reward),
            done=done,
        )

    except Exception as e:
        logger.error(f"Step failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
def shutdown():
    """Cleanup on server shutdown."""
    global env
    if env is not None:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting BrowserGym server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
