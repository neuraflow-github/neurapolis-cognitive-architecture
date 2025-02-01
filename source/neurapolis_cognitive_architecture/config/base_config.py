from pydantic import Field
from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    max_context_window_token_count: int = Field(
        190_000
    )  # 200k would be the max for Claude 3.5 Sonnet or o3-mini
