from pydantic import BaseModel, Field


class CognitiveArchitectureConfig(BaseModel):
    max_reference_count: int = Field(40)
    max_llm_context_reference_count: int = Field(40)
