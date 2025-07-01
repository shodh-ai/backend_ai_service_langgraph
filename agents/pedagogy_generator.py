# agents/pedagogy_generator.py

import logging
import json
import os
import httpx
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

# These are the keys for the student's context, which are still needed for the prompt.
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

async def pedagogy_generator_node(state: AgentGraphState) -> dict:
    """
    Generates a personalized pedagogy plan, returning a structured object with both the
    plan itself and the layered content to present it to the user.
    """
    logger.info("---Executing Pedagogy Generator Node (Layered Content Version)---")
    try:

        rag_data = state.get("rag_document_data")
        if not rag_data:
            raise ValueError("Pedagogy generator received no RAG documents from state.")

        # 1. Get initial report content from state
        initial_report = state.get("initial_report_content", "")
        if not initial_report:
            raise ValueError("No initial report content found in state.")


        student_context = {col: state.get(col, 'Not specified') for col in query_columns}

        json_output_example = """
{
    "pedagogy_plan": [
        {
            "type": "Modelling",
            "task": "speaking",
            "topic": "Describing a personal experience",
            "level": "Intermediate"
        }
    ],
    "layered_content": {
        "main_explanation": "Based on your goal to improve fluency, I suggest we start with a modeling exercise where you'll describe a personal experience. This will help you practice structuring your thoughts.",
        "simplified_explanation": "Let's start by having you tell a short story. It's great practice!",
        "clarifications": {
            "Why this task?": "This helps with storytelling, a key part of fluent conversation.",
            "What if I can't think of a story?": "No worries! I can give you a prompt to get started."
        },
        "sequence": [
            {"type": "tts", "content": "I've looked at your progress and I have a great next step for us."},
            {"type": "tts", "content": "I suggest we work on telling a story about a personal experience. How does that sound?"},
            {"type": "listen", "expected_intent": "user_agrees_to_plan", "prompt_if_silent": "Ready to start?"}
        ]
    }
}
"""
        prompt = f"""
        You are an expert pedagogue designing a personalized English learning step for a TOEFL student.

        **Student Context:**
        {json.dumps(student_context, indent=2)}

        **Initial Assessment Report:**
        {initial_report}

        **Your Task:**
        Design the *very next* learning step for this student. Return a SINGLE JSON object with two main keys: `pedagogy_plan` and `layered_content`.

        1.  `pedagogy_plan`: A list containing a SINGLE JSON object for the next task. The object must have `type`, `task`, `topic`, and `level`.
        2.  `layered_content`: A JSON object containing the content to present this plan to the student. It must have these four keys:
            - `main_explanation`: Your reasoning for suggesting this plan.
            - `simplified_explanation`: A simple, encouraging way to phrase the suggestion.
            - `clarifications`: 2-3 pre-written answers to likely questions about the plan.
            - `sequence`: An interactive sequence to propose the plan. It must end with a `listen` step to get the student's agreement.

        **Example JSON Output:**
        {json_output_example}

        **Instructions:**
        - The `pedagogy_plan` should only contain ONE step for now.
        - The tone should be encouraging and collaborative.
        - The `expected_intent` in the `listen` step should be `user_agrees_to_plan`.

        Generate the JSON payload now.
        """

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(prompt)
        response_json = json.loads(response.text)
        logger.info(f"LLM Response for pedagogy: {response_json}")

        pedagogy_plan = response_json.get("pedagogy_plan", [])
        layered_content_payload = response_json.get("layered_content", {})

        user_id = state.get("user_id")
        auth_token = state.get("user_token")

        if user_id and auth_token and pedagogy_plan:
            pronity_backend_url = os.getenv("PRONITY_BACKEND_URL", "http://localhost:8000")
            api_endpoint = f"{pronity_backend_url}/user/save-flow"
            payload = {
                "analysis": layered_content_payload.get("main_explanation", ""),
                "flowElements": pedagogy_plan,
            }
            headers = {
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient() as client:
                logger.info(f"Sending pedagogy flow to Pronity backend for user {user_id}")
                api_response = await client.post(api_endpoint, json=payload, headers=headers)
                api_response.raise_for_status()
                logger.info(f"Successfully saved pedagogy flow for user {user_id}.")
        else:
            logger.warning("User ID, Auth Token, or pedagogy_plan not found. Skipping pedagogy save.")

        return {
            "intermediate_pedagogy_payload": layered_content_payload,
            "current_pedagogy_plan": pedagogy_plan
        }

    except Exception as e:
        logger.error(f"PedagogyGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"intermediate_pedagogy_payload": {"error": True, "error_message": str(e)}}
