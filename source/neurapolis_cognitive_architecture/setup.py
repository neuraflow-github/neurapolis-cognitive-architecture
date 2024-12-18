import asyncio
import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from neurapolis_common import config as common_config


async def setup():
    logging.info("Started creating checkpointer DB tables...")

    async with AsyncPostgresSaver.from_conn_string(
        common_config.db_connection_string
    ) as async_postgres_saver:
        await async_postgres_saver.setup()

    logging.info("Finished creating checkpointer DB tables")


if __name__ == "__main__":
    asyncio.run(setup())
