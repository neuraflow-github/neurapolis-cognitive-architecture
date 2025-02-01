from neurapolis_cognitive_architecture.models import CognitiveArchitectureConfig
from neurapolis_retriever.enums import QualityPreset


def get_cognitive_architecture_config_by_quality_preset(
    quality_preset: QualityPreset,
) -> CognitiveArchitectureConfig:
    if quality_preset == QualityPreset.DEEP_RESEARCH:
        return CognitiveArchitectureConfig(
            max_reference_count=50,
            max_llm_context_reference_count=50,
        )
    if quality_preset == QualityPreset.HIGH:
        return CognitiveArchitectureConfig(
            max_reference_count=50,
            max_llm_context_reference_count=40,
        )
    elif quality_preset == QualityPreset.MEDIUM:
        return CognitiveArchitectureConfig(
            max_reference_count=50,
            max_llm_context_reference_count=30,
        )
    elif quality_preset == QualityPreset.LOW:
        return CognitiveArchitectureConfig(
            max_reference_count=50,
            max_llm_context_reference_count=20,
        )
    else:
        raise Exception(f"Invalid quality preset: {quality_preset}")
