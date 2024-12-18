import asyncio
import logging
from uuid import uuid4

from neurapolis_cognitive_architecture import (
    MyAiMessage,
    MyHumanMessage,
    NeurapolisCognitiveArchitecture,
)
from neurapolis_retriever import DateLoaderLogEntry, QualityPreset, TextLoaderLogEntry


async def run_cognitive_architecture():
    cognitive_architecture = NeurapolisCognitiveArchitecture()

    async def send_loader_update_to_client(loader_update):
        logging.info("---")
        logging.info(f"Retriever step: {loader_update.graph_step}")
        logging.info(f"Search count: {loader_update.search_count}")
        logging.info(f"Hit count: {loader_update.hit_count}")
        logging.info(f"Relevant hit count: {loader_update.relevant_hit_count}")
        logging.info("Log entries:")
        for x_log_entry in loader_update.log_entries:
            if isinstance(x_log_entry, TextLoaderLogEntry):
                logging.info(f"  - {x_log_entry.text}")
            elif isinstance(x_log_entry, DateLoaderLogEntry):
                logging.info(f"  - {x_log_entry.date}")
        logging.info("---")

    async def send_ai_message_to_client(ai_message: MyAiMessage):
        logging.info("---")
        logging.info("AI Response:")
        logging.info(ai_message)
        logging.info("---")

    human_message = MyHumanMessage(
        id=str(uuid4()),
        content="Welche Spielplätze wurden gebaut oder geplant?",
        quality_preset=QualityPreset.LOW,
    )

    await cognitive_architecture.query(
        thread_id=str(uuid4()),
        human_message=human_message,
        send_loader_update_to_client=send_loader_update_to_client,
        send_ai_message_to_client=send_ai_message_to_client,
    )


if __name__ == "__main__":
    asyncio.run(run_cognitive_architecture())
