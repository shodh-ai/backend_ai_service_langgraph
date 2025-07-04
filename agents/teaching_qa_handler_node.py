# agents/teaching_qa_handler_node.py
import logging
import json
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState
from graph.utils import query_knowledge_base # Corrected import

logger = logging.getLogger(__name__)

# Configure Gemini client
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def teaching_qa_handler_node(state: AgentGraphState) -> dict:
    """
    Handles a student's question asked during a lesson, providing a contextual answer.
    """
    logger.info("--- Executing Teaching Q&A Handler Node ---")
    # --- Debugging: Log incoming state ---
    plan_on_entry = state.get("pedagogical_plan")
    logger.info(f"QA_HANDLER_ENTRY_DEBUG: Plan on entry: {plan_on_entry}")
    logger.info(f"QA_HANDLER_ENTRY_DEBUG: Type of plan on entry: {type(plan_on_entry)}")
    # --- End Debugging ---
    try:
        student_question = state.get("transcript")
        # The lesson context is already loaded in the state from the start of the turn
        plan = state.get("pedagogical_plan")
        current_index = state.get("current_plan_step_index", 0)
        active_persona = state.get("active_persona", "Structuralist")
        
        if not student_question:
            # Should not happen if routed correctly, but good to handle
            return {"output_content": {"text_for_tts": "Sorry, I didn't catch your question. Could you please repeat it?"}}
            
        # Get the focus of the step the student was on when they asked
        step_focus = "General Topic"
        if plan and current_index < len(plan):
            step_focus = plan[current_index].get("focus", "General Topic")

        # 1. RAG for context related to the question and lesson step
        # Use the generic KB query function. The student's question is the best query text.
        # The lesson_id provides specific context for the RAG query.
        current_context = state.get("current_context", {})
        lesson_id = current_context.get("lesson_id") # Get lesson_id, may be None

        if not lesson_id:
            logger.warning("lesson_id not found in context. Query will be against the general 'teaching' category.")

        # Query the knowledge base using the 'teaching' category and the specific lesson_id if available.
        rag_context = await query_knowledge_base(
            query_string=student_question,
            category="teaching",
            lesson_id=lesson_id
        )
        
        # 2. Construct LLM Prompt to answer the question
        llm_prompt = f"""
        You are '{active_persona}' AI Tutor. You are in the middle of a lesson.
        The current lesson step focus is: "{step_focus}".
        The student has interrupted to ask the following question: "{student_question}"
        
        Relevant teaching material for context:
        {json.dumps(rag_context, indent=2)}
        
        Your task:
        1. Provide a clear and helpful answer to the student's question.
        2. Maintain your '{active_persona}' persona and tone.
        3. After answering, gently prompt the student to confirm if they understand your answer
           and are ready to continue the lesson.
        
        Return JSON: {{"text_for_tts": "...", "ui_actions": []}}
        """
        
        # 3. Call the LLM
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        output_data = json.loads(response.text)

        logger.info(f"Generated Q&A response: {output_data}")
        
        # 4. Return the output using the correct key for the teaching payload.
        #    This ensures the state is updated correctly without overwriting other values.
        return {
            "intermediate_teaching_payload": output_data
        }

    except Exception as e:
        logger.error(f"TeachingQAHandlerNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to handle Q&A: {e}", "route_to_fallback": True}