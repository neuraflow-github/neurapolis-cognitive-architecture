from pydantic import Field
from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    retrieved_file_limit: int = Field(10)
    context_window_token_limit: int = Field(
        150_0000
    )  # 200k would be the max for Claude 3.5 Sonnet
