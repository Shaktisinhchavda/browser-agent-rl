from pathlib import Path


def get_path_to_configs() -> str:
    """Returns path to the configs directory."""
    path = str(Path(__file__).parent.parent.parent / "configs")
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_path_to_media() -> str:
    """Returns path to the media directory."""
    path = str(Path(__file__).parent.parent.parent / "media")
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_path_model_checkpoints(experiment_name: str) -> str:
    """
    Returns path to model checkpoints directory.
    Uses /model_checkpoints volume mount in Docker, falls back to local dir.
    """
    # Try Docker volume mount first
    docker_path = Path("/model_checkpoints") / experiment_name.replace("/", "--")
    if docker_path.parent.exists():
        docker_path.mkdir(parents=True, exist_ok=True)
        return str(docker_path)

    # Fallback to local directory
    local_path = Path(__file__).parent.parent.parent / "checkpoints" / experiment_name.replace("/", "--")
    local_path.mkdir(parents=True, exist_ok=True)
    return str(local_path)
