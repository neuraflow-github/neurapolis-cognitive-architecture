from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
import uuid
from dotenv import load_dotenv
from langchain_core.runnables import chain
from test import graph

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def middleware(config, req: Request):
    body = await req.json()
    config["configurable"] = {"thread_id": str(uuid.uuid4())}
    return config


# @chain
# async def graph_chain(inputs, config):
#     messages = inputs.get("messages", [])
#     state = {"messages": messages}

#     async for event in graph.astream_events(state, config, version="v1"):
#         kind = event["event"]
#         if kind == "on_chat_model_stream" and event["name"] == "agent":
#             chunk = event["data"]["chunk"]
#             if isinstance(chunk.content, list):
#                 for item in chunk.content:
#                     if isinstance(item, dict) and item.get("type") == "text":
#                         yield item["text"]
#             else:
#                 yield chunk.content
# elif kind == "on_tool_start":
#     yield f"\nStarting tool: {event['name']} with inputs: {event['data'].get('input')}\n"
# elif kind == "on_tool_end":
#     yield f"\nTool {event['name']} finished. Output: {event['data'].get('output')}\n"


add_routes(
    app,
    graph,
    path="/chat",
    playground_type="chat",
    per_req_config_modifier=middleware,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
