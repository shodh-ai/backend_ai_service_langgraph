import logging
import json
from typing import Dict, Any, List
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from state import AgentGraphState
from graph.utils import query_knowledge_base # Import the robust RAG query utility

logger = logging.getLogger(__name__)
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def pedagogical_strategy_planner_node(state: AgentGraphState) -> dict:
    """
    The MESO PLANNER.
    Takes a single, approved Learning Objective (LO) and creates a multi-step,
    multi-modality lesson plan for it.
    """
    logger.info("--- Executing Pedagogical Strategy Planner Node (Meso Planner) ---")
    try:
        user_id = state["user_id"]
        student_profile = state.get("student_memory_context", {})
        active_persona = state.get("active_persona", "The Structuralist")
        lo_to_plan = state.get("current_lo_to_address") # This should be set by the curriculum navigator

        if not lo_to_plan:
            raise ValueError("PedagogicalStrategyPlannerNode received no LO to plan for.")
        
        lo_id = lo_to_plan.get("id")
        lo_title = lo_to_plan.get("title")

        # 1. RAG on your detailed Teaching/Modeling/Scaffolding CTA data for this specific LO
        rag_query = (f"Persona: {active_persona}, Learning Objective: {lo_title}, "
                     f"Student Proficiency: {student_profile.get('skill_level')}, "
                     f"Student Affect: {student_profile.get('affective_state')}. "
                     "What is the best sequence of teaching modalities (TEACH, MODEL, SCAFFOLD, PRACTICE)?")
        
        # 2. Retrieve relevant pedagogical sequences from the knowledge base using our robust RAG utility.
        # This query will find teaching strategies matching the student's persona and the current learning objective.
        retrieved_teaching_sequences = await query_knowledge_base(rag_query, category="pedagogical_sequencing")

        # 2. Construct prompt for the LLM lesson planner
        prompt = f"""
        You are an expert AI Lesson Designer for the '{active_persona}' AI Tutor persona.
        Your task is to create a detailed, multi-step lesson plan to achieve a single Learning Objective (LO) for a student.

        LEARNING OBJECTIVE TO PLAN:
        {lo_id}: {lo_title}

        STUDENT PROFILE:
        {json.dumps(student_profile, indent=2)}

        EXPERT STRATEGY EXAMPLES (how '{active_persona}' typically sequences teaching for similar topics):
        {json.dumps(retrieved_teaching_sequences, indent=2)}
        
        INSTRUCTIONS:
        Design an ordered lesson plan as a JSON list of step objects.
        Each step object must have a `modality` and a short descriptive `focus`.
        The `modality` must be one of: 'TEACH', 'MODEL', 'SCAFFOLDED_PRACTICE', 'INDEPENDENT_PRACTICE', 'REVIEW', 'COMPREHENSION_CHECK'.
        Your sequence should be pedagogically sound and match the '{active_persona}' style. For example, 'The Structuralist' might always teach a concept before modeling it. 'The Interactive Explorer' might start with a question.
        The plan should be a logical progression to help the student master the LO.
        
        Example JSON Output for "ThesisStatement":
        [
          {{"step_id": 1, "modality": "TEACH", "focus": "Introduce the core definition and purpose of a thesis statement.", "target_page": "P7"}},
          {{"step_id": 2, "modality": "TEACH", "focus": "Explain the difference between an arguable claim and a simple fact.", "target_page": "P7"}},
          {{"step_id": 3, "modality": "MODEL", "focus": "Demonstrate writing a strong thesis for a sample prompt.", "target_page": "P8"}},
          {{"step_id": 4, "modality": "SCAFFOLDED_PRACTICE", "focus": "Guide student in writing a thesis using a template.", "target_page": "P4"}},
          {{"step_id": 5, "modality": "REVIEW", "focus": "Review the student's scaffolded practice attempt.", "target_page": "P6"}}
        ]

        Return ONLY the valid JSON list of plan steps.
        """

        # 3. Call the LLM
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = await model.generate_content_async(prompt, generation_config=GenerationConfig(response_mime_type="application/json"))
        lesson_plan_steps = json.loads(response.text)

        logger.info(f"PedagogicalStrategyPlannerNode: Generated a {len(lesson_plan_steps)}-step lesson plan for LO '{lo_title}'.")
        
        # 4. Update the state with the new, active lesson plan
        return {
            "pedagogical_plan": lesson_plan_steps,
            "current_plan_step_index": 0, # Start at the beginning
            "current_plan_active": True, # Flag that we are now executing this micro-plan
        }
        
    except Exception as e:
        logger.error(f"PedagogicalStrategyPlannerNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to create lesson plan: {e}"}