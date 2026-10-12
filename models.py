import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# First key found (top to bottom) wins. OmniRoute points at a local proxy.
PROVIDER_CONFIGS: list[dict] = [
    {
        "env_var": "OPENAI_API_KEY",
        "label": "OpenAI",
        "provider": "openai",
        "model": "gpt-5-nano",
    },
    {
        "env_var": "ANTHROPIC_API_KEY",
        "label": "Anthropic",
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
    },
    {
        "env_var": "GROQ_API_KEY",
        "label": "Groq",
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
    },
    {
        "env_var": "OMNIROUTE_API_KEY",
        "label": "OmniRoute",
        "provider": "openai",
        "model": "free_combo",
        "base_url": "http://localhost:20128/v1",
    },
]


def get_active_provider_config() -> dict | None:
    for config in PROVIDER_CONFIGS:
        if os.getenv(config["env_var"]):
            return config
    return None


def build_model():
    config = get_active_provider_config()
    if config is None:
        raise RuntimeError(
            "No LLM API key found. Set one of: "
            + ", ".join(c["env_var"] for c in PROVIDER_CONFIGS)
        )
    kwargs = {
        "model": config["model"],
        "model_provider": config["provider"],
        "api_key": os.getenv(config["env_var"]),
    }
    if "base_url" in config:
        kwargs["base_url"] = config["base_url"]
    return init_chat_model(**kwargs)


model = build_model()
