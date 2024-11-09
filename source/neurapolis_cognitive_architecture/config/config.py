from neurapolis_common.config.environment import Environment
from neurapolis_common.config.get_environment import get_environment

from .base_config import BaseConfig
from .config_development import DevelopmentConfig
from .config_production import ProductionConfig
from .config_staging import StagingConfig

environment = get_environment()

config: BaseConfig
if environment == Environment.DEVELOPMENT:
    config = DevelopmentConfig()
elif environment == Environment.STAGING:
    config = StagingConfig()
elif environment == Environment.PRODUCTION:
    config = ProductionConfig()
else:
    raise ValueError(f"Invalid environment: {environment}")
