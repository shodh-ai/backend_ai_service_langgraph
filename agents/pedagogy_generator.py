import logging
import json

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
    "Answer One",
    "Answer Two",
    "Answer Three",
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
        pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Initializing pedagogy vectorstore for the first time...")
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, '..', 'data', 'pedagogy_data.csv')
        
        pedagogy_logger.info(f"PEDAGOGY_GENERATOR.PY: Reading CSV from {file_path}...")
        df = pd.read_csv(file_path)
        pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Finished reading CSV.")

        pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Initializing GoogleGenerativeAIEmbeddings...")
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Initialized GoogleGenerativeAIEmbeddings.")

        df["combined_text_for_embedding"] = df[query_columns].astype(str).agg(" ".join, axis=1)
        langchain_documents = []
        pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Preparing documents for Chroma...")
        for index, row in df.iterrows():
            page_content = row["combined_text_for_embedding"]
            metadata = row.drop("combined_text_for_embedding").to_dict()
            langchain_documents.append(Document(page_content=page_content, metadata=metadata))
        pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Finished preparing documents.")

        pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Calling Chroma.from_documents()...")
        vectorstore = Chroma.from_documents(
            documents=langchain_documents,
            embedding=embedding_model,
            persist_directory="data/pedagogy_chroma",
        )
        pedagogy_logger.info("PEDAGOGY_GENERATOR.PY: Finished Chroma.from_documents(). Vectorstore initialized.")
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

    estimated_english_comfort_level = state.get(
        "estimated_overall_english_comfort_level"
    )
    initial_impression = state.get("initial_impression")
    speaking_strengths = state.get("speaking_strengths")
    fluency = state.get("fluency")
    grammar = state.get("grammar")
    vocabulary = state.get("vocabulary")
    question_one_answer = state.get("question_one_answer")
    question_two_answer = state.get("question_two_answer")
    question_three_answer = state.get("question_three_answer")

    print(
        estimated_english_comfort_level,
        initial_impression,
        fluency,
        grammar,
        vocabulary,
        question_one_answer,
        question_two_answer,
        question_three_answer,
    )

    query_values = {
        "Answer One": question_one_answer,
        "Answer Two": question_two_answer,
        "Answer Three": question_three_answer,
        "Estimated Overall English Comfort Level": estimated_english_comfort_level,
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
        logger.info(f"Response: {response_json}")
        return {"task_suggestion_llm_output": response_json}
    except Exception as e:
        logger.error(f"Error processing with GenerativeModel: {e}")
        return None
