import asyncio
from uuid import uuid4

from neurapolis_cognitive_architecture import (
    MyAiMessage,
    MyHumanMessage,
    NeurapolisCognitiveArchitecture,
)


async def run_cognitive_architecture():
    cognitive_architecture = NeurapolisCognitiveArchitecture()

    def send_loader_update_to_client(loader_update):
        print("---")
        print(f"Retriever step: {loader_update.retriever_step}")
        print(f"Search count: {loader_update.search_count}")
        print(f"Hit count: {loader_update.hit_count}")
        print(f"Relevant hit count: {loader_update.relevant_hit_count}")
        print("Log entries:")
        for x_log_entry in loader_update.log_entries:
            print(f"  - {x_log_entry.text}")
        print("---")

    def send_ai_message_to_client(ai_message: MyAiMessage):
        print("---")
        print("AI Response:")
        print(ai_message.content)
        print("---")

    user_message = MyHumanMessage(
        id=str(uuid4()), content="Welche Spielplätze wurden gebaut oder geplant?"
    )

    await cognitive_architecture.query(
        thread_id=str(uuid4()),
        user_message=user_message,
        send_loader_update_to_client=send_loader_update_to_client,
        send_ai_message_to_client=send_ai_message_to_client,
    )


if __name__ == "__main__":
    asyncio.run(run_cognitive_architecture())