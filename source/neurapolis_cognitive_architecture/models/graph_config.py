from typing import Callable, TypedDict

from neurapolis_retriever import LoaderUpdate


class GraphConfig(TypedDict):
    thread_id: str
    send_loader_update_to_client: Callable[[LoaderUpdate], None]
