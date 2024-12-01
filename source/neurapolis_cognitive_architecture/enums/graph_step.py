from enum import StrEnum


class GraphStep(StrEnum):
    DECIDER = "DECIDER"
    RETRIEVER = "RETRIEVER"
    REPLIER = "REPLIER"
