# langgraph-service/agents/modelling_qa_handler_node.py
import logging
import json
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState
from graph.utils import query_knowledge_base

logger = logging.getLogger(__name__)

# Configure Gemini client
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def modelling_qa_handler_node(state: AgentGraphState) -> dict:
    """
    Handles a student's question asked during a modelling session.
    """
    logger.info("--- Executing Modelling Q&A Handler Node ---")
    try:
        student_question = state.get("transcript")
        plan = state.get("modelling_plan")
        current_index = state.get("current_plan_step_index", 0)
        active_persona = state.get("active_persona", "The Structuralist")
        
        if not student_question:
            return {"intermediate_modelling_payload": {"text_for_tts": "Sorry, I didn't catch that. Could you repeat your question?"}}
            
        # Get the focus of the current step in the modelling plan
        step_focus = "General Topic"
        if plan and current_index < len(plan):
            step_focus = plan[current_index].get("focus", "General Topic")

        # RAG for context. We can use the original task prompt as a good filter.
        current_task_details = state.get("current_task_details", {})
        task_id = current_task_details.get("prompt_id", "modelling") # Use prompt_id or fallback

        rag_context = await query_knowledge_base(
            query_string=student_question,
            category=task_id
        )
        
        # Construct LLM Prompt to answer the question
        llm_prompt = f"""
        You are '{active_persona}' AI Tutor. You are in the middle of a modelling session to demonstrate a skill.
        The current session step focus is: "{step_focus}".
        The student has interrupted to ask the following question: "{student_question}"
        
        Relevant material for context:
        {json.dumps(rag_context, indent=2)}
        
        Your task:
        1. Provide a clear, helpful answer to the student's question in the context of the modelling session.
        2. Maintain your '{active_persona}' persona.
        3. After answering, gently prompt to see if they are ready to continue.
        
        Return JSON: {{"text_for_tts": "...", "ui_actions": []}}
        """
        
        # Call the LLM
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        output_data = json.loads(response.text)

        logger.info(f"Generated Modelling Q&A response: {output_data}")
        
        # Return the output in a dedicated key to avoid overwriting the main plan state.
        return {
            "intermediate_modelling_payload": output_data
        }

    except Exception as e:
        logger.error(f"ModellingQAHandlerNode: CRITICAL FAILURE: {e}", exc_info=True)
        fallback_payload = {"text_for_tts": "I seem to have encountered an issue. Could you please rephrase your question?"}
        return {"intermediate_modelling_payload": fallback_payload}
