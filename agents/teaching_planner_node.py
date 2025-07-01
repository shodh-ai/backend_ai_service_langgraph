# agents/teaching_planner_node.py
import logging
import json
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Configure Gemini client (or do this once in a central config file)
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
else:
    logger.error("GOOGLE_API_KEY not found in environment for teaching_planner_node.")

async def teaching_planner_node(state: AgentGraphState) -> dict:
    """
    Uses the RAG results to generate a lightweight, step-by-step lesson PLAN.
    This node runs AFTER the flexible_RAG_retrieval_node.
    """
    logger.info("--- Executing Teaching Planner Node (using RAG results) ---")
    try:
        rag_documents = state.get("rag_document_data")
        if not rag_documents:
            # This is a fallback if RAG returns nothing. We can still try to plan without it.
            logger.warning("Teaching planner has no RAG documents to use as inspiration.")
            rag_context_examples = "[No specific examples found, use general persona knowledge.]"
        else:
            rag_context_examples = json.dumps(rag_documents, indent=2)

        current_context = state.get("current_context")
        student_profile = state.get("student_memory_context", {})
        active_persona = state.get("active_persona", "Structuralist")

        # Construct the LLM Prompt for Planning
        llm_prompt = f"""
        You are an expert AI Pedagogical Strategist for the '{active_persona}' AI Tutor persona.
        Your task is to create a high-level lesson plan.

        CONTEXT:
        - Learning Objective: '{current_context.get('Learning_Objective_Focus')}'
        - Student's Profile: {json.dumps(student_profile, indent=2)}

        INSPIRATION from expert data (how '{active_persona}' typically structures this lesson):
        {rag_context_examples}

        THE PLAN:
        Devise an ordered pedagogical plan as a JSON list of step objects.
        Each object must have a "step_type" and a short, descriptive "focus".
        Possible step_types: 'EXPLAIN_CONCEPT', 'SHOW_MODEL', 'PROVIDE_SCAFFOLDED_PRACTICE', 'CHECK_UNDERSTANDING_QA'.
        
        Example JSON Output:
        [
          {{"step_type": "EXPLAIN_CONCEPT", "focus": "Core principles of essay coherence"}},
          {{"step_type": "CHECK_UNDERSTANDING_QA", "focus": "Check understanding of core principles"}},
          {{"step_type": "SHOW_MODEL", "focus": "Modeling coherence in an introductory paragraph"}},
          {{"step_type": "ASSIGN_INDEPENDENT_PRACTICE", "focus": "Write a coherent paragraph on a given topic"}}
        ]
        
        Return ONLY the valid JSON list.
        """
        
        # Call the LLM
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        lesson_plan_json = json.loads(response.text)
        logger.info(f"LLM generated lesson plan with {len(lesson_plan_json)} steps.")

        return {
            "pedagogical_plan": lesson_plan_json,
            "current_plan_step_index": 0,
            "current_plan_active": True
        }

    except Exception as e:
        logger.error(f"TeachingPlannerNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to plan lesson: {e}", "route_to_fallback": True}