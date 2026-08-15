from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(filename: str) -> dict:
    """
    Load a YAML configuration file from the project's configs directory.
    """

    config_path = PROJECT_ROOT / "configs" / filename

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_config():
    """
    Load the main project configuration.
    """

    return load_yaml("config.yaml")


def get_model_config():
    """
    Load the model configuration.
    """

    return load_yaml("model.yaml")