# langgraph-service/agents/modelling_planner_node.py
import logging
import json
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Configure the Gemini client
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
else:
    logger.error("GOOGLE_API_KEY not found in environment for modelling_planner_node.")

async def modelling_planner_node(state: AgentGraphState) -> dict:
    """
    Analyzes a student's submission, consumes RAG results from the state,
    and generates a multi-step plan for delivering feedback and a model answer.
    """
    logger.info("--- Executing Modelling Planner Node (Simplified) ---")
    try:
        # --- 1. Get current state ---
        student_submission = state.get("student_task_submission")
        if not student_submission:
            raise ValueError("Modelling planner requires 'student_task_submission' in state.")

        original_task_prompt = state.get("current_task_details", {}).get("description", "the task")
        active_persona = state.get("active_persona", "The Structuralist")
        
        # --- 2. Get RAG documents from state ---
        rag_docs = state.get("rag_document_data", [])
        logger.info(f"Planner received {len(rag_docs)} documents from RAG node.")

        # --- 3. Construct the LLM Prompt with RAG context ---
        llm_prompt = f"""
        You are an expert AI Pedagogical Strategist for the '{active_persona}' AI Tutor persona.
        Your task is to analyze a student's submission and create a high-level plan to model a better response and provide feedback, using expert examples as a guide.

        **Reference Pedagogical Examples from Knowledge Base:**
        {json.dumps(rag_docs, indent=2)}

        **Original Task Prompt:**
        {original_task_prompt}

        **Student's Submission:**
        {student_submission}

        **THE PLAN:**
        Devise an ordered pedagogical plan based on the student's submission and inspired by the reference examples.
        The plan should be a JSON list of step objects. Each object must have a "step_type" and a short, descriptive "focus".
        Possible step_types: 'DELIVER_FEEDBACK', 'SHOW_MODEL_ANSWER', 'EXPLAIN_KEY_CONCEPT', 'CHECK_UNDERSTANDING_QA'.
        
        **Example JSON Output (adapt based on the RAG examples and student work):**
        [
          {{"step_type": "DELIVER_FEEDBACK", "focus": "Overall impression and key strengths"}},
          {{"step_type": "SHOW_MODEL_ANSWER", "focus": "Presenting a complete model answer for comparison"}},
          {{"step_type": "EXPLAIN_KEY_CONCEPT", "focus": "Deep dive on the use of transition words in the model"}},
          {{"step_type": "CHECK_UNDERSTANDING_QA", "focus": "Check if the student has questions about the model or feedback"}}
        ]
        
        Return ONLY the valid JSON list.
        """
        
        # --- 4. Generate Plan ---
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        modelling_plan_json = json.loads(response.text)
        logger.info(f"LLM generated modelling plan with {len(modelling_plan_json)} steps.")

        return {
            "modelling_plan": modelling_plan_json,
            "current_plan_step_index": 0,
            "current_plan_active": True
        }

    except Exception as e:
        logger.error(f"ModellingPlannerNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to plan modelling session: {e}"}
