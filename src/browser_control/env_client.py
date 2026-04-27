"""
BrowserGym HTTP Client

Communicates with the BrowserGym FastAPI server (browsergym_server.py)
running in the browsergym-env Docker container.

Replaces the reference repo's `envs.browsergym_env` module with a
simple HTTP client that talks to our own REST API.
"""

import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class Observation:
    """Parsed observation from the BrowserGym environment."""
    goal: Optional[str] = None
    axtree_txt: Optional[str] = None
    screenshot: Optional[list] = None
    error: Optional[str] = None
    last_action_error: bool = False


@dataclass
class StepResult:
    """Result from a reset or step call."""
    observation: Observation
    reward: float = 0.0
    done: bool = False


class BrowserGymClient:
    """
    HTTP client for the BrowserGym FastAPI server.

    Usage:
        client = BrowserGymClient("http://localhost:7860")
        result = client.reset()
        result = client.step("click('13')")
    """

    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        """Check if the server is running."""
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def reset(self, task_name: str = "browsergym/miniwob.click-test", seed: int = None) -> StepResult:
        """Reset the environment and return the initial observation."""
        payload = {"task_name": task_name}
        if seed is not None:
            payload["seed"] = seed

        resp = requests.post(
            f"{self.base_url}/reset",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        obs_data = data["observation"]
        observation = Observation(
            goal=obs_data.get("goal"),
            axtree_txt=obs_data.get("axtree_txt"),
            screenshot=obs_data.get("screenshot"),
            error=obs_data.get("error"),
            last_action_error=obs_data.get("last_action_error", False),
        )

        return StepResult(
            observation=observation,
            reward=0.0,
            done=data.get("done", False),
        )

    def step(self, action_str: str) -> StepResult:
        """Execute an action and return the new observation + reward."""
        resp = requests.post(
            f"{self.base_url}/step",
            json={"action_str": action_str},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        obs_data = data["observation"]
        observation = Observation(
            goal=obs_data.get("goal"),
            axtree_txt=obs_data.get("axtree_txt"),
            screenshot=obs_data.get("screenshot"),
            error=obs_data.get("error"),
            last_action_error=obs_data.get("last_action_error", False),
        )

        return StepResult(
            observation=observation,
            reward=float(data.get("reward", 0.0)),
            done=data.get("done", False),
        )
