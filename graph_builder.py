import logging
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END
from memory.mem0_memory import Mem0Memory
from state import AgentGraphState
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import json

# Import all the agent node functions
from graph.teaching_flow import create_teaching_subgraph
from graph.feedback_flow import create_feedback_subgraph
from graph.scaffolding_flow import create_scaffolding_subgraph
from graph.modeling_flow import create_modeling_subgraph
from graph.cowriting_flow import create_cowriting_subgraph
from graph.pedagogy_flow import create_pedagogy_subgraph

# Import shared agent node functions
from agents import (
    save_interaction_node,
    format_final_output_for_client_node,
    handle_welcome_node,
    student_data_node,
    welcome_prompt_node,
    conversation_handler_node,
    load_student_data_node,
    motivational_support_node,
    progress_reporter_node,
    inactivity_prompt_node,
    tech_support_acknowledger_node,
    prepare_navigation_node,
    session_wrap_up_node,
    finalize_session_in_mem0_node,
    initial_report_generation_node,
    error_generator_node,
    feedback_student_data_node,
    query_document_node,
    RAG_document_node,
    feedback_planner_node,
    feedback_generator_node,
    scaffolding_student_data_node,
    struggle_analyzer_node,
    scaffolding_retriever_node,
    scaffolding_planner_node,
    scaffolding_generator_node,
    pedagogy_generator_node,
    modelling_query_document_node,
    modelling_RAG_document_node,
    modelling_generator_node,
    modelling_output_formatter_node,
    teaching_rag_node,
    teaching_delivery_node,
    teaching_generator_node,
)

# Define node names for clarity
NODE_SAVE_INTERACTION = "save_interaction"
NODE_LOAD_STUDENT_DATA = "load_student_data"
NODE_CONVERSATION_HANDLER = "conversation_handler"
NODE_FORMAT_FINAL_OUTPUT = "format_final_output"
NODE_HANDLE_WELCOME = "handle_welcome"
NODE_STUDENT_DATA = "student_data"
NODE_WELCOME_PROMPT = "welcome_prompt"
NODE_MOTIVATIONAL_SUPPORT = "motivational_support" # Added motivational node name
NODE_PROGRESS_REPORTER = "progress_reporter" # Added progress reporter node name
NODE_INACTIVITY_PROMPT = "inactivity_prompt" # Added inactivity prompt node name
NODE_TECH_SUPPORT_ACKNOWLEDGER = "tech_support_acknowledger"
NODE_PREPARE_NAVIGATION = "prepare_navigation"
NODE_SESSION_WRAP_UP = "session_wrap_up"
NODE_FINALIZE_SESSION_IN_MEM0 = "finalize_session_in_mem0"
NODE_INITIAL_REPORT_GENERATION = "initial_report_generation"
NODE_ERROR_GENERATION = "error_generation"
NODE_FEEDBACK_STUDENT_DATA = "feedback_student_data"
NODE_QUERY_DOCUMENT = "query_document"
NODE_RAG_DOCUMENT = "rag_document"
NODE_FEEDBACK_PLANNER = "feedback_planner"
NODE_FEEDBACK_GENERATOR = "feedback_generator"
NODE_SCAFFOLDING_STUDENT_DATA = "scaffolding_student_data"
NODE_STRUGGLE_ANALYZER = "struggle_analyzer"
NODE_SCAFFOLDING_RETRIEVER = "scaffolding_retriever"
NODE_SCAFFOLDING_PLANNER = "scaffolding_planner"
NODE_SCAFFOLDING_GENERATOR = "scaffolding_generator"
NODE_INITIAL_REPORT_GENERATION = "initial_report_generation"
NODE_PEDAGOGY_GENERATION = "pedagogy_generation"
NODE_MODELLING_QUERY_DOCUMENT = "modelling_query_document"
NODE_MODELLING_RAG_DOCUMENT = "modelling_rag_document"
NODE_MODELLING_GENERATOR = "modelling_generator"
NODE_MODELLING_OUTPUT_FORMATTER = "modelling_output_formatter"
NODE_TEACHING_RAG = "teaching_rag"
NODE_TEACHING_DELIVERY = "teaching_delivery"
NODE_TEACHING_GENERATOR = "teaching_generator"

# Define node names for subgraphs
NODE_TEACHING_MODULE = "TEACHING_MODULE"
NODE_FEEDBACK_MODULE = "FEEDBACK_MODULE"
NODE_SCAFFOLDING_MODULE = "SCAFFOLDING_MODULE"
NODE_MODELING_MODULE = "MODELING_MODULE"
NODE_COWRITING_MODULE = "COWRITING_MODULE"
NODE_PEDAGOGY_MODULE = "PEDAGOGY_MODULE"
NODE_P1_CURRICULUM_NAVIGATOR = "p1_curriculum_navigator"


# Define a router node (empty function that doesn't modify state)
async def router_node(state: AgentGraphState) -> dict:
    logger.info(
        f"Router node entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    return {}


# Router function after saving an interaction
async def route_after_save_interaction(state: AgentGraphState) -> str:
    if state.get("session_is_ending", False):
        logger.info("Session is ending. Routing from SaveInteraction to FinalizeSessionInMem0.")
        return NODE_FINALIZE_SESSION_IN_MEM0
    else:
        logger.info("Session not ending. Interaction saved. Routing to END.")
        return END


# Placeholder for P1 Curriculum Navigator node
async def p1_curriculum_navigator_node(state: AgentGraphState) -> dict:
    logger.info(
        f"P1 Curriculum Navigator activated for user {state.get('user_id', 'unknown_user')}. Deciding next steps after teaching module."
    )
    # This node would typically set state to guide the next actions, e.g., next lesson, or switch to another mode.
    # For now, it prepares for a general conversation handler.
    return {"message": "Teaching module completed. Navigating to next steps."}

# Define the initial router function based on task_stage
# Router function after motivational support
async def route_after_motivation(state: AgentGraphState) -> str:
    hint = state.get("next_node_hint_from_motivation")
    logger.info(f"Routing after motivational support. Hint: {hint}")
    if hint and hint in [NODE_CONVERSATION_HANDLER, NODE_WELCOME_PROMPT, NODE_STUDENT_DATA, NODE_HANDLE_WELCOME]: # Add other valid node names
        return hint
    # Add more sophisticated logic if the hint can be more complex
    # For example, if hint is 'REVIEW_COHERENCE_NOTES', map it to a specific graph node
    # if hint == "REVIEW_COHERENCE_NOTES":
    #     return "NODE_KNOWLEDGE_REVIEW_COHERENCE" # Assuming such a node exists
    logger.warning(f"No specific route or unknown hint '{hint}' after motivational support. Defaulting to conversation handler.")
    return "DEFAULT_FALLBACK_AFTER_MOTIVATION" # Fallback to conversation_handler via the conditional edge map

async def initial_router_logic(state: AgentGraphState) -> str:
    next_task_details = state.get("next_task_details", {})
    user_id = state.get('user_id', 'unknown_user')

    # Priority routing for specific task types like LESSON
    # Ensure next_task_details is not None before trying to access its 'type'
    if next_task_details and next_task_details.get("type") == "LESSON":
        logger.info(f"Routing to TEACHING_MODULE for user {user_id}")
        return NODE_TEACHING_MODULE

    context = state.get("current_context")
    task_stage_from_context = getattr(context, "task_stage", None) # Get task_stage from context object if it's an object
    if isinstance(context, dict):
        task_stage_from_context = context.get("task_stage") # Get task_stage if context is a dict

    # Handle inactivity prompt first
    if task_stage_from_context == "SYSTEM_USER_INACTIVITY_DETECTED":
        logger.info(f"Detected user inactivity. Routing to NODE_INACTIVITY_PROMPT.")
        return NODE_INACTIVITY_PROMPT

    transcript = state.get("transcript")
    chat_history = state.get("chat_history")

    # Access task_stage using getattr since context is an object
    task_stage = getattr(context, "task_stage", None) if context else None

    # Route both ROX_WELCOME_INIT and welcome_flow to the welcome handler
    if task_stage in ["ROX_WELCOME_INIT", "welcome_flow"]:
        return NODE_HANDLE_WELCOME
    if task_stage == "FEEDBACK_GENERATION":
        return NODE_FEEDBACK_MODULE
    if task_stage == "SCAFFOLDING_GENERATION":
        return NODE_SCAFFOLDING_MODULE
    if task_stage == "MODELLING_ACTIVITY_REQUESTED":
        return NODE_MODELING_MODULE
    if task_stage == "COWRITING_GENERATION": # Corrected task stage name
        return NODE_COWRITING_MODULE
    if task_stage == "TEACHING_LESSON_REQUESTED":
        return NODE_TEACHING_MODULE

    if task_stage_from_context == "INITIAL_REPORT_GENERATION":
        return NODE_INITIAL_REPORT_GENERATION
    if task_stage_from_context == "PEDAGOGY_GENERATION":
        return NODE_PEDAGOGY_MODULE

    # Default to NLU based routing if no specific task_stage matches above
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set. Cannot perform NLU routing.")
        # Fallback if API key is missing, perhaps to a general handler or error state
        return NODE_CONVERSATION_HANDLER 

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        prompt = (
            f"""
        You are an NLU assistant for the Rox AI Tutor on its welcome page.
        The student responded: '{transcript}'.
        Chat History: {chat_history}
        Categorize the student's intent from the possible intents. Possible intents:
        - 'CONFIRM_START_SUGGESTED_TASK'
        - 'REJECT_SUGGESTED_TASK_REQUEST_ALTERNATIVE'
        - 'ASK_CLARIFYING_QUESTION_ABOUT_SUGGESTED_TASK'
        - 'ASK_GENERAL_KNOWLEDGE_QUESTION' (e.g., about a grammar topic, TOEFL strategy)
        - 'REQUEST_STATUS_DETAIL'
        - 'GENERAL_CHITCHAT'
        - 'REPORT_TECHNICAL_ISSUE'  # New intent
        - 'INTENT_TO_QUIT_SESSION' # New intent for ending the session
        - 'OTHER_OFF_TOPIC'
        If 'ASK_GENERAL_KNOWLEDGE_QUESTION', extract the core topic/question.
        If 'REPORT_TECHNICAL_ISSUE', extract 'issue_description' (what the student says is wrong, e.g., 'my audio is not working') and 'reported_emotion' (e.g., 'frustration', 'confusion', 'annoyance' based on their language).
        """
            + "Return JSON: {'intent': '<INTENT_NAME>', 'extracted_topic': '<topic if ASK_GENERAL_KNOWLEDGE_QUESTION>', 'extracted_entities': {'issue_description': '<description if REPORT_TECHNICAL_ISSUE>', 'reported_emotion': '<emotion if REPORT_TECHNICAL_ISSUE>'}}"
        )
        response = model.generate_content(prompt)
        logger.debug(f"NLU Response Text for user {user_id}: {response.text}")

        # Check for progress-related queries before JSON parsing, as they are keyword-based
        progress_queries = [
            "how am i doing", "what's my progress", "am i improving", 
            "my score went down", "check my progress", "show me my progress"
        ]
        if transcript and any(query in transcript.lower() for query in progress_queries):
            logger.info(f"Detected progress query in transcript for user {user_id}: {transcript}")
            state['triggering_event_for_motivation'] = None # Clear motivational trigger if any
            return NODE_PROGRESS_REPORTER

        # Check for motivational triggers before JSON parsing
        if transcript and ("frustrated" in transcript.lower() or "nervous" in transcript.lower() or "bored" in transcript.lower() or "i can't do this" in transcript.lower()):
            logger.info(f"Detected potential need for motivational support in transcript for user {user_id}: {transcript}")
            state['triggering_event_for_motivation'] = f"Detected sentiment in user transcript: {transcript}"
            return NODE_MOTIVATIONAL_SUPPORT

        response_json = json.loads(response.text)
        intent = response_json.get("intent", "GENERAL_CHITCHAT")
        extracted_topic = response_json.get("extracted_topic", "")
        extracted_entities_from_nlu = response_json.get("extracted_entities", {})
        state['nlu_intent'] = intent

        if extracted_entities_from_nlu and isinstance(extracted_entities_from_nlu, dict) and (extracted_entities_from_nlu.get('issue_description') or extracted_entities_from_nlu.get('reported_emotion')):
            state['extracted_entities'] = extracted_entities_from_nlu
            logger.info(f"NLU extracted entities for technical issue for user {user_id}: {extracted_entities_from_nlu}")
        else:
            state['extracted_entities'] = {}

        logger.info(f"NLU Intent for user {user_id}: {intent}, Extracted Topic: {extracted_topic}, Extracted Entities: {state.get('extracted_entities')}")

        if intent == "INTENT_TO_QUIT_SESSION":
            logger.info(f"Routing to NODE_SESSION_WRAP_UP for user {user_id}.")
            return NODE_SESSION_WRAP_UP
        elif intent == "REPORT_TECHNICAL_ISSUE":
            logger.info(f"Routing to NODE_TECH_SUPPORT_ACKNOWLEDGER for user {user_id}.")
            return NODE_TECH_SUPPORT_ACKNOWLEDGER
        elif intent == "CONFIRM_START_SUGGESTED_TASK":
            next_task_details = state.get("next_task_details")
            if next_task_details and isinstance(next_task_details, dict) and next_task_details.get("page_target"):
                logger.info(f"Routing to NODE_PREPARE_NAVIGATION for user {user_id}.")
                return NODE_PREPARE_NAVIGATION
            else:
                logger.warning(f"Intent is CONFIRM_START_SUGGESTED_TASK for user {user_id}, but 'next_task_details' are missing/invalid. Routing to NODE_CONVERSATION_HANDLER.")
                state['missing_next_task_for_confirmation'] = True
                return NODE_CONVERSATION_HANDLER
        # Fallback for other intents or if logic above doesn't return
        logger.info(f"Routing to NODE_CONVERSATION_HANDLER for user {user_id} with intent: {intent}")
        return NODE_CONVERSATION_HANDLER

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError in initial_router_logic for user {user_id}: {e}. Response text: {response.text if 'response' in locals() else 'Response object not available'}")
        return NODE_CONVERSATION_HANDLER # Fallback on JSON parsing error
    except Exception as e:
        logger.error(f"Error in initial_router_logic NLU processing for user {user_id}: {e}")
        return NODE_CONVERSATION_HANDLER # Fallback for any other NLU error



def build_graph():
    """Builds and compiles the LangGraph application with the P1 and P2 submission flows."""
    logger.info("Building LangGraph with AgentGraphState...")
    NODE_ROUTER = "router"

    # Instantiate the checkpointer for state persistence
    checkpointer = Mem0Memory()

    workflow = StateGraph(AgentGraphState)

    # Add nodes for core components
    workflow.add_node(NODE_SAVE_INTERACTION, save_interaction_node)
    workflow.add_node(NODE_LOAD_STUDENT_DATA, load_student_data_node)
    workflow.add_node(NODE_CONVERSATION_HANDLER, conversation_handler_node)
    workflow.add_node(NODE_FORMAT_FINAL_OUTPUT, format_final_output_for_client_node)
    workflow.add_node(NODE_HANDLE_WELCOME, handle_welcome_node)
    workflow.add_node(NODE_STUDENT_DATA, student_data_node)
    workflow.add_node(NODE_WELCOME_PROMPT, welcome_prompt_node)

    workflow.add_node(NODE_MOTIVATIONAL_SUPPORT, motivational_support_node) # Added motivational node
    workflow.add_node(NODE_PROGRESS_REPORTER, progress_reporter_node) # Added progress reporter node
    workflow.add_node(NODE_INACTIVITY_PROMPT, inactivity_prompt_node) # Added inactivity prompt node
    workflow.add_node(NODE_TECH_SUPPORT_ACKNOWLEDGER, tech_support_acknowledger_node) # Added tech support node
    workflow.add_node(NODE_PREPARE_NAVIGATION, prepare_navigation_node) # Added prepare_navigation_node
    workflow.add_node(NODE_SESSION_WRAP_UP, session_wrap_up_node) # Added session_wrap_up_node
    workflow.add_node(NODE_FINALIZE_SESSION_IN_MEM0, finalize_session_in_mem0_node) # Added finalize_session_in_mem0_node
    workflow.add_node(NODE_ERROR_GENERATION, error_generator_node)
    workflow.add_node(NODE_FEEDBACK_STUDENT_DATA, feedback_student_data_node)
    workflow.add_node(NODE_QUERY_DOCUMENT, query_document_node)
    workflow.add_node(NODE_RAG_DOCUMENT, RAG_document_node)
    workflow.add_node(NODE_FEEDBACK_PLANNER, feedback_planner_node)
    workflow.add_node(NODE_FEEDBACK_GENERATOR, feedback_generator_node)

    # Add scaffolding system nodes
    workflow.add_node(NODE_SCAFFOLDING_STUDENT_DATA, scaffolding_student_data_node)
    workflow.add_node(NODE_STRUGGLE_ANALYZER, struggle_analyzer_node)
    workflow.add_node(NODE_SCAFFOLDING_RETRIEVER, scaffolding_retriever_node)
    workflow.add_node(NODE_SCAFFOLDING_PLANNER, scaffolding_planner_node)
    workflow.add_node(NODE_SCAFFOLDING_GENERATOR, scaffolding_generator_node)

    workflow.add_node(NODE_INITIAL_REPORT_GENERATION, initial_report_generation_node)
    workflow.add_node(NODE_PEDAGOGY_GENERATION, pedagogy_generator_node)

    # Modelling System Nodes
    workflow.add_node(NODE_MODELLING_QUERY_DOCUMENT, modelling_query_document_node)
    workflow.add_node(NODE_MODELLING_RAG_DOCUMENT, modelling_RAG_document_node)
    workflow.add_node(NODE_MODELLING_GENERATOR, modelling_generator_node)
    workflow.add_node(NODE_MODELLING_OUTPUT_FORMATTER, modelling_output_formatter_node)

    # Teaching System Nodes
    workflow.add_node(NODE_TEACHING_RAG, teaching_rag_node)
    workflow.add_node(NODE_TEACHING_DELIVERY, teaching_delivery_node) # This might become obsolete or used for non-LLM paths
    workflow.add_node(NODE_TEACHING_GENERATOR, teaching_generator_node)

    workflow.add_node(NODE_ROUTER, router_node)

    # Instantiate and add all subgraphs
    teaching_subgraph_instance = create_teaching_subgraph()
    feedback_subgraph_instance = create_feedback_subgraph()
    scaffolding_subgraph_instance = create_scaffolding_subgraph()
    modeling_subgraph_instance = create_modeling_subgraph()
    cowriting_subgraph_instance = create_cowriting_subgraph()
    pedagogy_subgraph_instance = create_pedagogy_subgraph()

    workflow.add_node(NODE_TEACHING_MODULE, teaching_subgraph_instance)
    workflow.add_node(NODE_FEEDBACK_MODULE, feedback_subgraph_instance)
    workflow.add_node(NODE_SCAFFOLDING_MODULE, scaffolding_subgraph_instance)
    workflow.add_node(NODE_MODELING_MODULE, modeling_subgraph_instance)
    workflow.add_node(NODE_COWRITING_MODULE, cowriting_subgraph_instance)
    workflow.add_node(NODE_PEDAGOGY_MODULE, pedagogy_subgraph_instance)

    workflow.add_node(NODE_P1_CURRICULUM_NAVIGATOR, p1_curriculum_navigator_node)

    workflow.set_entry_point(NODE_ROUTER)

    # Conditional Edges from the main router
    path_map = {
        NODE_HANDLE_WELCOME: NODE_HANDLE_WELCOME,
        NODE_TEACHING_MODULE: NODE_TEACHING_MODULE,
        NODE_FEEDBACK_MODULE: NODE_FEEDBACK_MODULE,
        NODE_SCAFFOLDING_MODULE: NODE_SCAFFOLDING_MODULE,
        NODE_MODELING_MODULE: NODE_MODELING_MODULE,
        NODE_COWRITING_MODULE: NODE_COWRITING_MODULE,
        NODE_PEDAGOGY_MODULE: NODE_PEDAGOGY_MODULE,
        NODE_INITIAL_REPORT_GENERATION: NODE_INITIAL_REPORT_GENERATION,
        NODE_INACTIVITY_PROMPT: NODE_INACTIVITY_PROMPT,
        NODE_CONVERSATION_HANDLER: NODE_LOAD_STUDENT_DATA, # Now routes to load memory first
        NODE_MOTIVATIONAL_SUPPORT: NODE_MOTIVATIONAL_SUPPORT,
        NODE_PROGRESS_REPORTER: NODE_PROGRESS_REPORTER,
        NODE_TECH_SUPPORT_ACKNOWLEDGER: NODE_TECH_SUPPORT_ACKNOWLEDGER,
        NODE_PREPARE_NAVIGATION: NODE_PREPARE_NAVIGATION,
        NODE_SESSION_WRAP_UP: NODE_SESSION_WRAP_UP,
        "DEFAULT_FALLBACK": NODE_LOAD_STUDENT_DATA # Now the fallback routes to load memory first
    }
    workflow.add_conditional_edges(NODE_ROUTER, initial_router_logic, path_map)

    # Conditional Edges from Motivational Support Node
    motivational_support_path_map = {
        NODE_CONVERSATION_HANDLER: NODE_LOAD_STUDENT_DATA, # Now routes to load memory first
        NODE_WELCOME_PROMPT: NODE_WELCOME_PROMPT,
        NODE_STUDENT_DATA: NODE_STUDENT_DATA,
        NODE_HANDLE_WELCOME: NODE_HANDLE_WELCOME,
        "DEFAULT_FALLBACK_AFTER_MOTIVATION": NODE_CONVERSATION_HANDLER  # As per route_after_motivation logic
    }
    workflow.add_conditional_edges(
        NODE_MOTIVATIONAL_SUPPORT,
        route_after_motivation,
        motivational_support_path_map
    )

    # Edges for flows that are NOT subgraphs
    workflow.add_edge(NODE_HANDLE_WELCOME, NODE_STUDENT_DATA)
    workflow.add_edge(NODE_STUDENT_DATA, NODE_WELCOME_PROMPT)
    workflow.add_edge(NODE_WELCOME_PROMPT, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_INITIAL_REPORT_GENERATION, NODE_FORMAT_FINAL_OUTPUT)
    
    # Updated conversation flow with memory loading first
    workflow.add_edge(NODE_LOAD_STUDENT_DATA, NODE_CONVERSATION_HANDLER)
    workflow.add_edge(NODE_CONVERSATION_HANDLER, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_INACTIVITY_PROMPT, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_TECH_SUPPORT_ACKNOWLEDGER, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_PREPARE_NAVIGATION, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_SESSION_WRAP_UP, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_PROGRESS_REPORTER, NODE_FORMAT_FINAL_OUTPUT)

    # Session Wrap Up Path
    workflow.add_edge(NODE_SESSION_WRAP_UP, NODE_FORMAT_FINAL_OUTPUT) # Wrap-up message needs formatting

    # Common path after output is formatted
    workflow.add_edge(NODE_FORMAT_FINAL_OUTPUT, NODE_SAVE_INTERACTION)

    # Conditional path after interaction is saved
    workflow.add_conditional_edges(
        NODE_SAVE_INTERACTION,
        route_after_save_interaction, # Decides if session finalization is needed
        {
            NODE_FINALIZE_SESSION_IN_MEM0: NODE_FINALIZE_SESSION_IN_MEM0,
            END: END, # If route_after_save_interaction returns END (i.e., __end__), then terminate the graph.
                                         # This was an error in previous logic, FormatFinalOutput should not be hit again here.
                                         # Corrected to END for non-session-ending paths.
        },
    )
    # If session is NOT ending, route_after_save_interaction returns NODE_FORMAT_FINAL_OUTPUT.
    # This should be END, as formatting and saving are done.
    # Let's refine route_after_save_interaction to return END for the 'else' case.

    # Path after session data is finalized in Mem0
    workflow.add_edge(NODE_FINALIZE_SESSION_IN_MEM0, END)

    # Edges from subgraphs back to the main flow
    workflow.add_edge(NODE_FEEDBACK_MODULE, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_SCAFFOLDING_MODULE, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_MODELING_MODULE, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_COWRITING_MODULE, NODE_FORMAT_FINAL_OUTPUT)

    # Edges for the teaching module flow
    workflow.add_edge(NODE_TEACHING_MODULE, NODE_P1_CURRICULUM_NAVIGATOR)
    workflow.add_edge(NODE_P1_CURRICULUM_NAVIGATOR, NODE_CONVERSATION_HANDLER) # Or to another router/handler if needed

    # Initial report generation flow
    workflow.add_edge(NODE_INITIAL_REPORT_GENERATION, NODE_FORMAT_FINAL_OUTPUT)

    # Pedagogy generation flow
    workflow.add_edge(NODE_PEDAGOGY_MODULE, NODE_FORMAT_FINAL_OUTPUT)
    workflow.add_edge(NODE_INITIAL_REPORT_GENERATION, NODE_FORMAT_FINAL_OUTPUT)


    # Modelling system flow
    workflow.add_edge(NODE_MODELLING_QUERY_DOCUMENT, NODE_MODELLING_RAG_DOCUMENT)
    workflow.add_edge(NODE_MODELLING_RAG_DOCUMENT, NODE_MODELLING_GENERATOR)
    workflow.add_edge(NODE_MODELLING_GENERATOR, NODE_MODELLING_OUTPUT_FORMATTER)
    workflow.add_edge(NODE_MODELLING_OUTPUT_FORMATTER, NODE_FORMAT_FINAL_OUTPUT)

    # Teaching System Edges (LLM-based flow)
    workflow.add_edge(NODE_TEACHING_RAG, NODE_TEACHING_GENERATOR) # RAG output goes to the new generator
    workflow.add_edge(NODE_TEACHING_GENERATOR, NODE_FORMAT_FINAL_OUTPUT) # Generator output goes to formatter

    # Compile the graph with the checkpointer
    toefl_tutor_graph = workflow.compile(checkpointer=checkpointer)
    logger.info("LangGraph compiled successfully with Mem0Memory checkpointer.")
    return toefl_tutor_graph


# To test the graph building process (optional, can be run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        graph = build_graph()
        logger.info("Graph compiled successfully in __main__.")
    except Exception as e_build:
        logger.error(f"Error building graph in __main__: {e_build}", exc_info=True)
