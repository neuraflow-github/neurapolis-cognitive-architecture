import asyncio
import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from neurapolis_common import config as common_config

logger = logging.getLogger()


async def setup():
    try:
        logger.info("Started creating checkpointer Postgres tables...")

        async with AsyncPostgresSaver.from_conn_string(
            common_config.db_connection_string
        ) as async_postgres_saver:
            await async_postgres_saver.setup()

        logger.info("Finished creating checkpointer Postgres tables")
    except Exception as exception:
        logger.error(
            f"Error creating checkpointer Postgres tables: {exception}", exc_info=True
        )


if __name__ == "__main__":
    asyncio.run(setup())
