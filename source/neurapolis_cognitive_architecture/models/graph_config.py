from typing import Callable, TypedDict

from neurapolis_retriever import LoaderUpdate


class GraphConfig(TypedDict):
    send_loader_update_to_client: Callable[[LoaderUpdate], None]
    configurable: dict
