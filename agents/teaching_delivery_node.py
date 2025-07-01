# agents/teaching_delivery_node.py
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


async def teaching_delivery_generator_node(state: AgentGraphState) -> dict:
    """
    Executes a SINGLE step from the plan, generating content "just-in-time"
    using the RAG results provided.
    """
    logger.info("--- Executing Teaching Delivery Generator Node ---")
    try:
        rag_documents = state.get("rag_document_data")
        if not rag_documents:
            logger.warning("Teaching Delivery Generator has no RAG documents for this step.")
            rag_context_examples = "[No specific examples found, generate based on general persona knowledge.]"
        else:
            rag_context_examples = json.dumps(rag_documents, indent=2)

        plan = state.get("pedagogical_plan")
        current_index = state.get("current_plan_step_index", 0)
        current_step = plan[current_index]
        step_focus = current_step.get("focus")
        active_persona = state.get("active_persona", "Structuralist")
        student_profile = state.get("student_memory_context", {})

        llm_prompt = f"""
        You are '{active_persona}' AI Tutor. Your task is to deliver one specific part of a lesson.

        Student Profile: {json.dumps(student_profile, indent=2)}
        Current Lesson Step Focus: "{step_focus}"

        Expert Examples (how '{active_persona}' explains this, from knowledge base):
        {rag_context_examples}
        
        Based on this, generate the content for this lesson step.
        - Your spoken text should be clear, engaging, and match the persona.
        - If the expert examples describe visual aids, create `ui_actions` to represent them.
        - Conclude your spoken text with a brief comprehension check question.
        
        Return a single JSON object with keys "text_for_tts" and "ui_actions".
        """

        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        output_data = json.loads(response.text)
        
        lesson_id = state.get("current_context", {}).get("lesson_id", "UNKNOWN_LESSON")
        output_data["statement_id"] = f"lesson_{lesson_id}_step_{current_index}"
        
        logger.info(f"Generated output for step {current_index + 1}: {output_data}")

        return {
            "output_content": output_data,
            "last_action_was": "TEACHING_DELIVERY", # Flag for router
            "rag_document_data": None # Clear this after use
        }

    except Exception as e:
        logger.error(f"TeachingDeliveryGeneratorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to deliver lesson step: {e}", "route_to_fallback": True}