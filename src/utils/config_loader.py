import yaml
from utils.logger import LOG

def load_config(config_file="config.yaml"):
    """Loads the YAML configuration file."""
    try:
        with open(config_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        LOG.error(f"Configuration file {config_file} not found!")
        raise
    except yaml.YAMLError as e:
        LOG.error(f"Error parsing YAML file {config_file}: {e}")
        raise

def get_active_model_config():
    """Gets the configuration for the active model specified in the config file."""
    config = load_config()
    active_model_key = config.get("active_model")
    if not active_model_key:
        raise ValueError("'active_model' key not found in config.yaml")

    for provider, details in config.get("providers", {}).items():
        if active_model_key in details.get("models", {}):
            model_config = details["models"][active_model_key]
            # Ensure the provider key is in the model's config for easy access
            if 'provider' not in model_config:
                model_config['provider'] = provider
            return model_config

    raise ValueError(f"Configuration for active model '{active_model_key}' not found in config.yaml")