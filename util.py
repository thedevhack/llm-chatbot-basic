
import chromadb
from datetime import datetime
from openai import OpenAI
import json
import math
import logging
import time
from langsmith import traceable

import os
print("LANGSMITH_API_KEY:", os.environ.get("LANGSMITH_API_KEY"))
print("LANGSMITH_TRACING:", os.environ.get("LANGSMITH_TRACING"))
print("LANGSMITH_PROJECT:", os.environ.get("LANGSMITH_PROJECT"))

# from Week3 import total_token_used
from custom_logger import JSONLogger

json_logger = JSONLogger(__name__, "structured_server.log")

logging.basicConfig(filename="server.log",
                    level=logging.INFO,
                    format="%(asctime)s- %(levelname)s - %(message)s"
                    )

logger = logging.getLogger(__name__)

client = chromadb.PersistentClient(path="./chroma_db")

epoch = (datetime.now()).timestamp()

collection_name = f"ultimate_collection"

collection = client.get_or_create_collection(name=collection_name)

document_identifier: int = 1

with open("..\\OPENAI.txt", "r") as f:
    key = f.read()

with open("DOCUMENT_ID.txt", "r") as file:
    try:
        document_identifier: int = int(file.read())
    except:
        document_identifier: int = 1

openai = OpenAI(api_key=key)


def get_weather(city: str) -> str:
    city_weather = {
        "london": "Cloudy, 14°C",
        "new york": "Sunny, 22°C",
        "tokyo": "Rainy, 18°C",
        "mumbai": "Humid, 32°C",
        "bangalore": "Pleasant, 24°C",
    }

    return city_weather.get(city.lower(), f"The {city}'s weather cannot be determined!")

def calculate(expr: str) -> str:
    return eval(expr, {"__builtins__":{}}, {"sqrt": math.sqrt, "pow": math.pow})

def get_capital(country: str) -> str:
    country_capital = {
        "india": "New Delhi",
        "france": "Paris",
        "japan": "Tokyo",
        "usa": "Washington D.C.",
        "germany": "Berlin",
        "australia": "Canberra",
    }
    return country_capital.get(country.lower(), f"{country}'s Capital cannot be defined!")


TOOLS = [{
    "type": "function",
    "name": "get_weather",
    "description": "Returns current weather info for a given city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Name of the city for which the weather is to be fetched."
            }
        }
    }
},{
    "type": "function",
    "name": "calculate",
    "description": "Evaluates  a math expression",
    "parameters": {
        "type": "object",
        "properties": {
            "expr": {
                "type": "string",
                "description": "Expression to be evaluated."
            }
        }
    }
}, {
    "type": "function",
    "name": "get_capital",
    "description": "Returns the capital of a given country",
    "parameters": {
        "type": "object",
        "properties": {
            "country": {
                "type": "string",
                "description": "Country for which the capital is to be returned as output."
            }
        }
    }
}]

TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_capital": get_capital
}

def add_document_to_collection(text: str):
    global document_identifier

    collection.add(
        documents=[text],
        ids=f"doc_{document_identifier}"
    )
    document_identifier += 1
    with open("DOCUMENT_ID.txt", "w") as file:
        try:
            file.write(str(document_identifier))
        except:
            pass


def get_relevant_documents(query: str):
    relevant_documents = collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "distances"]
    )
    print(relevant_documents)
    return relevant_documents["documents"][0]

@traceable(name="llm_chat")
def llm_chat(user_query: str):
    try:
        relevant_documents = get_relevant_documents(user_query)

        user_llm_query = "Documents: \n\n"
        user_llm_query += "".join([f"Document {i+1}: {doc}" + "\n" for i, doc in enumerate(relevant_documents)])
        user_llm_query += "\n Question: " + str(user_query)
        print(user_llm_query)
        llm_response = openai.responses.create(
            model="gpt-4o-mini",
            instructions="""
                You are a helpful assistant.
                You answer based on the documents provided.
                If the answer is truly not available in any of the document, respond with:
                "I don't have any information regarding this."
                Don't use your general training knowledge — only use documents provided.
                """,
            input=[{"role":"user", "content":user_llm_query}]
        )

        return {"llm_resp": llm_response.output_text, "relevant_docs": relevant_documents}
    except Exception as e:
        print("Error while fetching llm response for user query it failed ,because of ", str(e))
        return None

@traceable(name="agent_chat_with_tools")
def agent_chat_with_tools(data: dict):

    startTime = time.time()
    messages = []

    query = data.get("query")

    # logger.info(f"REQUEST | QUERY: {query}")

    json_logger.log(event="REQUEST",
                    query=query)
    messages.append({"role":"user", "content":query})

    resp = openai.responses.create(
        model="gpt-4o-mini",
        instructions="You are a helpful agent. Use tools when needed.",
        input=messages,
        tools=TOOLS
    )
    while True:
        if "function_call" in [item.type for item in resp.output]:

            messages = []

            for item in resp.output:
                if item.type == "function_call":
                    funct_name = item.name
                    funct_args = json.loads(item.arguments)

                    print(f"Calling {funct_name} tool with args {funct_args}")
                    tool_startTime = time.time()
                    tool_output = TOOL_MAP[str(funct_name)](**funct_args)
                    # logger.info(F"TOOL_OUTPUT | QUERY {query} | tool_name {funct_name} | tool_args {funct_args} | tool_time_taken {time.time() - tool_startTime}s")

                    json_logger.log(event="tool_call",
                                    tool_name=funct_name,
                                    tool_args=funct_args,
                                    total_tool_output_time_taken=f"{time.time() - tool_startTime}s")

                    # print(f"Output for {funct_name} tool with args {funct_args}: ",tool_output)


                    messages.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_output
                    })

            resp = openai.responses.create(
                model="gpt-4o-mini",
                instructions="You are a helpful agent. Use tools when needed.",
                input=messages,
                previous_response_id=resp.id,
                tools=TOOLS
            )

        else:
            break

    total_tokens = resp.usage.total_tokens
    # logger.info(f"RESPONSE | TOTAL TOKENS USED {total_tokens} | total_time_taken {time.time() - startTime}s")
    json_logger.log(event="RESPONSE",
                    total_token_used=total_tokens,
                    total_time_taken_request=f"{time.time()-startTime}s")

    return resp.output_text



