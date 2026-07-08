from fastapi import FastAPI
from pydantic import BaseModel
from util import *

app = FastAPI()


class IngestModel(BaseModel):
    text: str

class ChatModel(BaseModel):
    query: str


@app.get("/health")
async def health_check():
    return {"message": "Server running!"}


@app.post("/chat")
async def chat(chat_data: ChatModel):
    try:
        data_dict = chat_data.model_dump()
        user_query = data_dict.get("query")
        chat_response = llm_chat(user_query)
        return {"message": "LLM Response", "requestStatus": 1, "data": chat_response}
    except Exception as e:
        return {"message": "Some error occurred!!", "requestStatus": 0, "error": str(e)}


@app.post("/ingest")
async def ingest(ingest_data: IngestModel):
    try:
        data_dict = ingest_data.model_dump()
        ingest_str = data_dict.get("text")
        add_document_to_collection(ingest_str)
        return {"message": "Document added successfully!", "requestStatus": 1}
    except Exception as e:
        return {"message": "Some error occurred!!", "requestStatus": 0, "error": str(e)}

@app.post("/agent-chat")
async def agent_chat(data: ChatModel):
    try:
        data_dict = data.model_dump()
        agent_response = agent_chat_with_tools(data_dict)
        return {"message": "LLM Response", "requestStatus": 1, "data": agent_response}
    except Exception as e:
        return {"message": "Some error occurred!!", "requestStatus": 0, "error": str(e)}

