import asyncio
import logging
from uuid import uuid4

from neurapolis_cognitive_architecture import (
    MyAiMessage,
    MyHumanMessage,
    NeurapolisCognitiveArchitecture,
)
from neurapolis_retriever import DateLoaderLogEntry, QualityPreset, TextLoaderLogEntry

logger = logging.getLogger(__name__)


async def run_cognitive_architecture():
    cognitive_architecture = NeurapolisCognitiveArchitecture()

    async def send_loader_update_to_client(loader_update):
        logger.info("---")
        logger.info(f"Retriever step: {loader_update.graph_step}")
        logger.info(f"Search count: {loader_update.search_count}")
        logger.info(f"Hit count: {loader_update.hit_count}")
        logger.info(f"Relevant hit count: {loader_update.relevant_hit_count}")
        logger.info("Log entries:")
        for x_log_entry in loader_update.log_entries:
            if isinstance(x_log_entry, TextLoaderLogEntry):
                logger.info(f"  - {x_log_entry.text}")
            elif isinstance(x_log_entry, DateLoaderLogEntry):
                logger.info(f"  - {x_log_entry.date}")
        logger.info("---")

    async def send_ai_message_to_client(ai_message: MyAiMessage):
        logger.info("---")
        logger.info("AI Response:")
        logger.info(ai_message)
        logger.info("---")

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
