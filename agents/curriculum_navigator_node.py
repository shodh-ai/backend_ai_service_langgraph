import logging
import json
from typing import Dict, Any, List
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from state import AgentGraphState
# Assume these utility modules exist and are accessible
from memory.mem0_client import shared_mem0_client as mem0_client # Your Mem0 client instance
from knowledge.knowledge_client import knowledge_client # Your client for LO Tree/Curriculum Map
from graph.utils import query_knowledge_base # The correct RAG utility for the knowledge base

logger = logging.getLogger(__name__)

# It's better to configure the model once, but for modularity, we can do it here.
# Ensure GOOGLE_API_KEY is loaded via load_dotenv() in your main app.py
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def curriculum_navigator_node(state: AgentGraphState) -> dict:
    """
    The MACRO PLANNER.
    Determines the next high-level Learning Objective (LO) for the student.
    """
    logger.info("--- Executing Curriculum Navigator Node (Macro Planner) ---")
    try:
        user_id = state["user_id"]
        student_profile = state.get("student_memory_context", {})
        recent_history = state.get("recent_interaction_history", []) # Get recent history
        active_persona = state.get("active_persona", "The Structuralist")
        student_preference_override = state.get("student_preference_override")

        if not student_profile:
            logger.warning(f"CurriculumNavigatorNode: student_memory_context is empty for user '{user_id}'. Planning as a new student.")
            # Proceed with an empty profile, the LLM can handle this.

        # 1. Fetch necessary "world model" information
        master_curriculum_map = await knowledge_client.get_full_curriculum_map()

        # 2. RAG on high-level pedagogy (optional but powerful)
        rag_query = (f"Student profile: {student_profile}. Recent interactions: {recent_history}. "
                     f"Student preference: {student_preference_override}. "
                     f"Which general skill area should be prioritized?")
        retrieved_planning_strategies = await query_knowledge_base(rag_query, category="curriculum_planning")

        # 3. Construct prompt for the LLM planner
        prompt = f"""
        You are an expert AI Curriculum Navigator for a TOEFL tutor, embodying the '{active_persona}' persona.
        Your task is to determine the single most impactful Learning Objective (LO) for the student to work on next.

        STUDENT'S LONG-TERM PROFILE:
        {json.dumps(student_profile, indent=2)}

        STUDENT'S RECENT INTERACTION HISTORY (last 5 turns):
        {json.dumps(recent_history, indent=2)}

        STUDENT'S EXPLICIT PREFERENCE (if any):
        {student_preference_override or "None given."}

        AVAILABLE CURRICULUM & DEPENDENCIES (lo_id: title, prerequisites):
        {json.dumps(master_curriculum_map, indent=2)}

        EXPERT STRATEGY EXAMPLES for curriculum planning:
        {json.dumps(retrieved_planning_strategies, indent=2)}

        INSTRUCTIONS:
        1.  Analyze the student's long-term profile AND their recent interaction history. The recent history is very important as it may show immediate struggles or interests.
        2.  If the recent history shows the student struggling with a specific concept, consider proposing an LO that addresses it, even if it wasn't in the long-term plan.
        3.  Consider the curriculum dependencies. A student CANNOT be assigned an LO if they have not mastered all of its `prerequisites`.
        4.  Prioritize the student's explicit preference (`{student_preference_override}`) if it's a valid and unlocked LO.
        5.  If no preference or recent struggle, choose the most logical next LO based on their long-term weaknesses and the curriculum path.
        6.  Generate a brief, persona-aligned statement to propose this LO to the student.
        
        Return a single, valid JSON object with the following keys:
        - "reasoning": Your step-by-step reasoning for choosing this LO, explicitly mentioning how recent history influenced your choice.
        - "chosen_lo_id": The unique ID of the selected Learning Objective (e.g., "LO_ThesisStatement").
        - "chosen_lo_title": The human-readable title of the LO (e.g., "Crafting a Strong Thesis Statement").
        - "proposal_script_for_student": The text you will use to propose this to the student.
        
        Example Output:
        {{
            "reasoning": "The student's goal is writing improvement. Their profile shows a weakness in 'Essay Structure'. They have mastered the prerequisite 'IdentifyingMainIdea'. Therefore, the next logical LO is 'ThesisStatement' (LO_002). This aligns with their goals and unlocks further writing skills.",
            "chosen_lo_id": "LO_002",
            "chosen_lo_title": "Writing a Clear Topic Sentence",
            "proposal_script_for_student": "Based on your focus on writing, I think the most impactful skill we can work on next is how to write clear topic sentences for your paragraphs. This is key for well-organized essays. How does that sound?"
        }}
        """

        # 4. Call the LLM
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = await model.generate_content_async(prompt, generation_config=GenerationConfig(response_mime_type="application/json"))
        planner_output = json.loads(response.text)

        logger.info(f"CurriculumNavigatorNode: Planned next LO for user '{user_id}': {planner_output.get('chosen_lo_title')}")
        logger.debug(f"CurriculumNavigatorNode: Reasoning: {planner_output.get('reasoning')}")

        # 5. Update the state with the planner's decision
        return {
            # This is the high-level decision, the "what"
            "current_lo_to_address": {
                "id": planner_output.get("chosen_lo_id"),
                "title": planner_output.get("chosen_lo_title"),
            },
            # This is the AI's spoken output for this turn
            "output_content": {
                "text_for_tts": planner_output.get("proposal_script_for_student"),
                "ui_actions": [] # UI Actions can be added to show options etc.
            },
            "last_ai_action": "PROPOSED_NEW_LO" # A flag for the next conversational turn
        }

    except Exception as e:
        logger.error(f"CurriculumNavigatorNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": f"Failed to determine next learning objective: {e}"}