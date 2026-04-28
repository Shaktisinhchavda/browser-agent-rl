"""
BrowserGym OpenEnv Server

Exposes the BrowserGym MiniWoB++ environment as an HTTP API
that the GRPO trainer can connect to.

This runs inside the browsergym-env Docker container (CPU only).
The trainer container connects to this over the Docker network.

IMPORTANT: All Playwright operations run in a single dedicated thread
to avoid event loop conflicts with FastAPI's thread pool.
"""

import logging
import os
from browsergym.utils.obs import flatten_axtree_to_str
import traceback
import threading
import queue
from typing import Optional

import browsergym.miniwob  # registers MiniWoB environments
import gymnasium as gym
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BrowserGym OpenEnv Server")


# ---------------------------------------------------------------
# Playwright Thread — all gym/browser ops happen in ONE thread
# ---------------------------------------------------------------

class PlaywrightWorker:
    """
    Runs all BrowserGym/Playwright operations in a single dedicated thread.
    This prevents 'no running event loop' errors caused by FastAPI's
    thread pool recycling threads between requests.
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.env = None

    def _run(self):
        """Worker loop — processes one task at a time."""
        while True:
            func, args, result_queue = self._queue.get()
            try:
                result = func(*args)
                result_queue.put(("ok", result))
            except Exception as e:
                result_queue.put(("error", e))

    def submit(self, func, *args):
        """Submit work to the Playwright thread and wait for result."""
        result_queue = queue.Queue()
        self._queue.put((func, args, result_queue))
        status, result = result_queue.get(timeout=120)
        if status == "error":
            raise result
        return result

    def do_reset(self, task_name: str, seed: int):
        """Reset the environment (runs in Playwright thread)."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass

        logger.info(f"Creating environment: {task_name}")
        self.env = gym.make(task_name, headless=True, slow_mo=0)
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def do_step(self, action_str: str):
        """Execute an action (runs in Playwright thread)."""
        if self.env is None:
            raise RuntimeError("Environment not initialized")
        obs, reward, terminated, truncated, info = self.env.step(action_str)
        done = terminated or truncated
        return obs, reward, done, info

    def do_close(self):
        """Close the environment (runs in Playwright thread)."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None


# Create the single worker
worker = PlaywrightWorker()


# ---------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "browsergym/miniwob.click-test"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_str: str


class ObservationResponse(BaseModel):
    goal: Optional[str] = None
    axtree_txt: Optional[str] = None
    screenshot: Optional[list] = None
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

    # Convert axtree_object (dict) to readable text like:
    # RootWebArea 'Click Test Task'
    #     [13] button 'Click Me!'
    axtree_object = obs.get("axtree_object", None)
    if axtree_object:
        axtree_txt = flatten_axtree_to_str(axtree_object)
    else:
        axtree_txt = ""

    last_action_error = bool(obs.get("last_action_error", False))
    error = str(obs.get("last_action_error", "")) if last_action_error else None

    # Skip screenshots to save bandwidth
    screenshot = None

    return ObservationResponse(
        goal=goal,
        axtree_txt=axtree_txt,
        screenshot=screenshot,
        error=error,
        last_action_error=last_action_error,
    )


# ---------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "task": "browsergym/miniwob.click-test"}


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest):
    """Reset the BrowserGym environment and return initial observation."""
    try:
        seed = request.seed or int.from_bytes(os.urandom(4), "big") % (2**31)
        obs, info = worker.submit(worker.do_reset, request.task_name, seed)

        observation = extract_observation(obs, info)
        logger.info(f"Environment reset. Goal: {observation.goal}")

        return ResetResponse(observation=observation, done=False)

    except Exception as e:
        logger.error(f"Reset failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Execute an action in the BrowserGym environment."""
    try:
        logger.info(f"Executing action: {request.action_str}")
        obs, reward, done, info = worker.submit(worker.do_step, request.action_str)

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
    try:
        worker.submit(worker.do_close)
    except Exception:
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting BrowserGym server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
