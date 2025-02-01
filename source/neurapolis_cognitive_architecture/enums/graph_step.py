from enum import StrEnum


class GraphStep(StrEnum):
    MENTIONED_FILES_RETRIEVER = "MENTIONED_FILES_RETRIEVER"
    AGENT = "AGENT"
    RETRIEVER = "RETRIEVER"
