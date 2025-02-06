from pydantic import BaseModel, Field


class CognitiveArchitectureConfig(BaseModel):
    max_reference_count: int = Field()
    max_llm_context_reference_count: int = Field()
    openai_reasoning_effort: str = Field()
