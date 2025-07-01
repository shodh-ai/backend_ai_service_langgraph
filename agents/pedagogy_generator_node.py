import logging
import json
import httpx

pedagogy_logger = logging.getLogger(__name__) # Changed to __name__ for consistency
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

query_columns = [
    "Goal",
    "Feeling",
    "Confidence",
    "Estimated Overall English Comfort Level",
    "Initial Impression",
    "Speaking Strengths",
    "Fluency",
    "Grammar",
    "Vocabulary",
]

vectorstore = None
embedding_model = None # Will be initialized by get_pedagogy_vectorstore

def get_pedagogy_vectorstore():
    global vectorstore, embedding_model
    if vectorstore is None:
        # Build an absolute path to the persist directory to ensure it's always found
        script_dir = os.path.dirname(__file__)
        base_dir = os.path.dirname(script_dir)  # This should be the project root
        persist_directory = os.path.join(base_dir, "data", "pedagogy_chroma")

        pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Initializing GoogleGenerativeAIEmbeddings...")
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Initialized GoogleGenerativeAIEmbeddings.")

        if os.path.exists(persist_directory):
            pedagogy_logger.info(f"PEDAGOGY_GENERATOR.PY: Loading existing vectorstore from {persist_directory}...")
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
            pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Finished loading vectorstore.")
        else:
            pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: No existing vectorstore found. Building from scratch...")
            script_dir = os.path.dirname(__file__)
            file_path = os.path.join(script_dir, '..', 'data', 'pedagogy_data.csv')
            
            pedagogy_logger.info(f"PEDAGOGY_GENERATOR.PY: Reading CSV from {file_path}...")
            df = pd.read_csv(file_path)
            pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Finished reading CSV.")

            df.rename(columns={
                "Answer One": "Goal",
                "Answer Two": "Feeling",
                "Answer Three": "Confidence"
            }, inplace=True)

            df["combined_text_for_embedding"] = df[query_columns].astype(str).agg(" ".join, axis=1)
            langchain_documents = []
            pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Preparing documents for Chroma...")
            for index, row in df.iterrows():
                page_content = row["combined_text_for_embedding"]
                metadata = row.drop("combined_text_for_embedding").to_dict()
                langchain_documents.append(Document(page_content=page_content, metadata=metadata))
            pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Finished preparing documents.")

            pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Calling Chroma.from_documents() to build and persist vectorstore...")
            try:
                vectorstore = Chroma.from_documents(
                    documents=langchain_documents,
                    embedding=embedding_model,
                    persist_directory=persist_directory,
                )
                pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Finished building and persisting vectorstore.")
            except Exception as e:
                pedagogy_logger.error(f"PEDAGOGY_GENERATOR.PY: An error occurred during Chroma.from_documents: {e}", exc_info=True)
                raise
    return vectorstore

def query_similar_documents(query_values):
    print(query_values)
    current_vectorstore = get_pedagogy_vectorstore()
    query_text = " ".join([str(query_values[col]) for col in query_columns])
    similar_documents = current_vectorstore.similarity_search(query_text, k=10)
    print(f"Top 10 similar rows to your query:\n---")
    metadata_list = []
    for i, doc in enumerate(similar_documents):
        print(f"Result {i+1}:")
        metadata_list.append(doc.metadata)
        print("---")
    return metadata_list


logger = logging.getLogger(__name__)


async def pedagogy_generator_node(state: AgentGraphState) -> dict:
    logger.info(
        f"PedagogyGeneratorNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )

    initial_report = state.get("initial_report_content", {})
    current_context = state.get("current_context", {})

    # Get diagnostic info from the initial report
    estimated_estimated_overall_english_comfort_level = initial_report.get(
        "estimated_overall_english_comfort_level"
    )
    initial_impression = initial_report.get("initial_impression")
    speaking_strengths = initial_report.get("speaking_strengths")
    fluency = initial_report.get("fluency")
    grammar = initial_report.get("grammar")
    vocabulary = initial_report.get("vocabulary")

    # Get user-provided context from the nested current_context object
    goal = current_context.goal if current_context else None
    feeling = current_context.feeling if current_context else None
    confidence = current_context.confidence if current_context else None

    query_values = {
        "Goal": goal,
        "Feeling": feeling,
        "Confidence": confidence,
        "Estimated Overall English Comfort Level": estimated_estimated_overall_english_comfort_level,
        "Initial Impression": initial_impression,
        "Speaking Strengths": speaking_strengths,
        "Fluency": fluency,
        "Grammar": grammar,
        "Vocabulary": vocabulary,
    }
    metadata_list = query_similar_documents(query_values)
    prompt = f"""
    You are an expert pedagogue in English. A student who we judged as:
    {query_values}
    wants to study for TOFEL. You need to design a pedagogy for this student.

    A few examples of the pedagogy for similar cases are:
    {metadata_list}
    
    Return a JSON object:
    {{
        "task_suggestion_tts": "A friendly, user-facing message that introduces the learning plan. For example: 'Based on what you've told me, I've created a personalized learning plan for you. We'll start with some grammar and vocabulary basics to build a strong foundation, and then move on to speaking and writing practice. How does that sound?'",
        "reasoning": "The overall reasoning behind the pedagogy. Why did you choose the order you did? Why not any other order? Is the response personalised for the student, etc.",
        "steps": [
            {{
                "type": "Module type out of Teaching, Modelling, Scaffolding, Cowriting and Test",
                "task": "speaking or writing",
                "topic": "If teaching which topic out of learning objective. If any other the topic the student will be expected to write or speak upon. Try to make it varied topics so the student doesn't feel comfort or discomfort.",
                "level": "Level of the task: Basic, Intermediate, Advanced",
            }}
        ]
    }}
    """

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = model.generate_content(prompt)
        response_json = json.loads(response.text)
        logger.info(f"LLM Response for pedagogy: {response_json}")

        # Save the generated pedagogy to the Pronity backend
        user_id = state.get("user_id")
        auth_token = state.get("user_token")

        if not user_id or not auth_token:
            logger.error(
                "User ID or Auth Token not found in state. Cannot save pedagogy to backend."
            )
        else:
            try:
                pronity_backend_url = os.getenv(
                    "PRONITY_BACKEND_URL", "http://localhost:8000"
                )
                api_endpoint = f"{pronity_backend_url}/user/save-flow"

                payload = {
                    "analysis": response_json.get("reasoning", ""),
                    "flowElements": response_json.get("steps", []),
                }

                headers = {
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json",
                }

                async with httpx.AsyncClient() as client:
                    logger.info(
                        f"Sending pedagogy flow to Pronity backend for user {user_id} at {api_endpoint}"
                    )
                    api_response = await client.post(
                        api_endpoint, json=payload, headers=headers
                    )

                    api_response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                    logger.info(
                        f"Successfully saved pedagogy flow for user {user_id}. Status: {api_response.status_code}"
                    )

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error occurred while saving pedagogy flow for user {user_id}: {e.response.status_code} - {e.response.text}"
                )
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during pedagogy save for user {user_id}: {e}"
                )

        return {"task_suggestion_llm_output": response_json}

    except Exception as e:
        logger.error(f"Error processing with GenerativeModel or saving pedagogy: {e}")
        return {}
