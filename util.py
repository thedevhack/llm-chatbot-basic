
import chromadb
from datetime import datetime
from openai import OpenAI

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



