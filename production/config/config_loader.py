"""Configuration loader for production system."""
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration loader and accessor."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dots).

        Args:
            key: Configuration key (e.g., 'data.train_file')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config.get('data', {})

    @property
    def preprocessing(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._config.get('preprocessing', {})

    @property
    def models(self) -> Dict[str, Any]:
        """Get models configuration."""
        return self._config.get('models', {})

    @property
    def thresholds(self) -> Dict[str, Any]:
        """Get thresholds configuration."""
        return self._config.get('thresholds', {})

    @property
    def business_rules(self) -> Dict[str, Any]:
        """Get business rules configuration."""
        return self._config.get('business_rules', {})

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get metrics configuration."""
        return self._config.get('metrics', {})

    @property
    def paths(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self._config.get('paths', {})

    @property
    def api(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self._config.get('api', {})

    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})

    @property
    def monitoring(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self._config.get('monitoring', {})

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"
