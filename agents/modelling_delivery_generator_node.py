# langgraph-service/agents/modelling_delivery_generator_node.py
import logging
import json
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Configure Gemini client
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def modelling_delivery_node(state: AgentGraphState) -> dict:
    """
    Executes a SINGLE step from the modelling plan, using RAG documents from the state
    to generate the content for the user just-in-time.
    """
    logger.info("--- Executing Modelling Delivery Node (Simplified) ---")
    try:
        # --- 1. Get current state ---
        plan = state.get("modelling_plan")
        current_index = state.get("current_plan_step_index", 0)
        
        if not plan or current_index >= len(plan):
            logger.error("No plan or invalid step index.")
            return {"error_message": "Cannot deliver content without a valid plan step."}

        current_step = plan[current_index]
        step_focus = current_step.get("focus")
        active_persona = state.get("active_persona", "The Structuralist")
        student_submission = state.get("student_task_submission", "")
        original_task_prompt = state.get("current_task_details", {}).get("description", "the task")

        # --- 2. Get RAG documents from state ---
        rag_docs = state.get("rag_document_data", [])
        logger.info(f"Delivery node received {len(rag_docs)} documents from state.")

        # --- 3. Construct the LLM Prompt with RAG context ---
        llm_prompt = f"""
        You are '{active_persona}' AI Tutor. Your task is to deliver one specific part of a modelling session.

        **Context from Knowledge Base (use this for inspiration):**
        {json.dumps(rag_docs, indent=2)}

        **Original Task:** {original_task_prompt}
        **Student's Submission:** {student_submission}
        **Current Session Step:** "{step_focus}"

        **Your Task:**
        Generate the content for this specific step. Your response should be engaging and match your persona.
        - If the step is 'DELIVER_FEEDBACK', provide the specific feedback mentioned in the focus, using the RAG docs as a style guide.
        - If the step is 'SHOW_MODEL_ANSWER', present a model answer clearly. You can use a `show_model_answer` UI action. The RAG docs contain examples of model answers.
        - If the step is 'EXPLAIN_KEY_CONCEPT', explain the concept clearly in the context of the student's submission and the model answer, referencing similar explanations in the RAG docs.
        - Conclude your spoken text with a brief check for understanding (e.g., 'Does that make sense?').
        
        Return a single JSON object with keys "text_for_tts" and "ui_actions".
        Example for showing a model answer:
        {{
            "text_for_tts": "Now, let's look at a model answer together. I'll display it on the screen.",
            "ui_actions": [{{"action": "show_model_answer", "content": "... a well-written model answer based on the RAG examples..."}}]
        }}
        """

        # --- 4. Generate Content ---
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        output_data = json.loads(response.text)
        
        logger.info(f"Generated output for modelling step {current_index + 1}: {output_data}")

        return {
            "output_content": output_data,
            "last_action_was": "MODELLING_DELIVERY"
        }

    except Exception as e:
        logger.error(f"ModellingDeliveryNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to deliver modelling step: {e}"}
