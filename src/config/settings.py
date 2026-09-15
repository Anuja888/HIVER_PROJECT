"""Central configuration loaded from config.yaml + .env (gitignored)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # src/config -> src -> project root


def _load_yaml() -> dict[str, Any]:
    path = REPO_ROOT / "config.yaml"
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    """Runtime settings. Environment variables override config.yaml."""

    # ---- Paths ----
    data_dir: Path = Field(default=REPO_ROOT / "data", alias="DATA_DIR")
    raw_data_file: Path = Field(default=REPO_ROOT / "data" / "raw" / "twitter_customer_support.csv", alias="RAW_DATA_FILE")
    cache_dir: Path = Field(default=REPO_ROOT / "data" / "cache", alias="CACHE_DIR")
    gold_dir: Path = Field(default=REPO_ROOT / "data" / "gold", alias="GOLD_DIR")
    reports_dir: Path = Field(default=REPO_ROOT / "reports", alias="REPORTS_DIR")
    docs_dir: Path = Field(default=REPO_ROOT / "docs", alias="DOCS_DIR")

    # ---- LLM ----
    llm_provider: str = Field(default="mock", alias="LLM_PROVIDER")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", alias="ANTHROPIC_MODEL")

    # ---- Kaggle (one-time download only) ----
    kaggle_username: str = Field(default="", alias="KAGGLE_USERNAME")
    kaggle_key: str = Field(default="", alias="KAGGLE_KEY")

    # ---- Sampling / eval ----
    sample_seed: int = Field(default=42, alias="SAMPLE_SEED")
    sample_size: int = Field(default=200000, alias="SAMPLE_SIZE")
    train_fraction: float = Field(default=0.7, alias="TRAIN_FRACTION")
    val_fraction: float = Field(default=0.1, alias="VAL_FRACTION")
    golden_size: int = Field(default=200, alias="GOLDEN_SIZE")

    # ---- Thresholds (tuned on validation split) ----
    intent_confidence_threshold: float = Field(default=0.55, alias="INTENT_CONFIDENCE_THRESHOLD")
    retrieval_similarity_threshold: float = Field(default=0.55, alias="RETRIEVAL_SIMILARITY_THRESHOLD")
    grounding_strength_threshold: float = Field(default=0.6, alias="GROUNDING_STRENGTH_THRESHOLD")

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def yaml_config(self) -> dict[str, Any]:
        return _load_yaml()


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        # Ensure directories exist
        for d in [_settings.data_dir, _settings.cache_dir, _settings.gold_dir,
                  _settings.reports_dir, _settings.docs_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
    return _settings


# Convenience singleton
settings = get_settings()