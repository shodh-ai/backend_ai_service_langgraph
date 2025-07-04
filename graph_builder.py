# FINAL, PERFECTED graph/graph_builder.py

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from state import AgentGraphState
from agents import save_interaction_node
from agents.acknowledge_interrupt_node import acknowledge_interrupt_node
from agents.conversation_handler import conversation_handler_node
from agents.pedagogical_strategy_planner_node import pedagogical_strategy_planner_node
from agents.plan_advancer_node import plan_advancer_node
from agents.initiate_module_execution_node import initiate_module_execution_node
from agents.student_model_node import (
    load_student_data_node, save_interaction_node, 
    NODE_LOAD_STUDENT_DATA, NODE_SAVE_INTERACTION
)
from agents.context_merger_node import context_merger_node

# Commenting out the unused import
# from memory import Mem0Checkpointer

# Configure logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# --- 1. Import all your flows AND simple nodes ---
from graph.modeling_flow import create_modeling_subgraph
from graph.teaching_flow import create_teaching_subgraph
from graph.scaffolding_flow import create_scaffolding_subgraph
from graph.feedback_flow import create_feedback_subgraph
from graph.cowriting_flow import create_cowriting_subgraph
from graph.rox_flow import create_rox_welcome_subgraph
from graph.pedagogy_flow import create_pedagogy_subgraph

# --- 2. Define ALL node names ---
NODE_CONTEXT_MERGER = "context_merger"
NODE_MODELING_MODULE = "modeling_module"
NODE_TEACHING_MODULE = "teaching_module"
NODE_SCAFFOLDING_MODULE = "scaffolding_module"
NODE_FEEDBACK_MODULE = "feedback_module"
NODE_COWRITING_MODULE = "cowriting_module"
NODE_PEDAGOGY_MODULE = "pedagogy_module"
NODE_HANDLE_WELCOME = "handle_welcome"
NODE_ACKNOWLEDGE_INTERRUPT = "acknowledge_interrupt"
NODE_ROUTER_ENTRY = "router_entry_point"
NODE_CONVERSATION_HANDLER = "conversation_handler"
NODE_PEDAGOGICAL_STRATEGY_PLANNER = "pedagogical_strategy_planner"
NODE_PLAN_ADVANCER = "plan_advancer"
NODE_INITIATE_MODULE_EXECUTION = "initiate_module_execution"
NODE_SAVE_INTERACTION = "save_interaction"

# This node is a placeholder, it can be an empty function
async def router_entry_node(state: AgentGraphState) -> dict:
    logger.info("--- Entering Main Graph Router ---")
    return {}

async def initial_router_logic(state: AgentGraphState) -> str:
    task_name = state.get("task_name")
    current_context = state.get("current_context", {})

    if not task_name:
        # Use .get() for safety
        task_name = current_context.get("task_stage")

    # If still nothing, default to conversation
    if not task_name:
        task_name = "handle_conversation"
    
    logger.info(f"INITIAL ROUTER: Evaluating route. Final determined Task Name: '{task_name}'")

    routing_map = {
        "handle_student_response": NODE_CONVERSATION_HANDLER,
        "user_wants_to_interrupt": NODE_ACKNOWLEDGE_INTERRUPT,
        "acknowledge_interruption": NODE_ACKNOWLEDGE_INTERRUPT,
        "handle_page_load": NODE_HANDLE_WELCOME,
        "start_modelling_activity": NODE_MODELING_MODULE,
        "request_teaching_lesson": NODE_TEACHING_MODULE, # Old, for reference
        "initiate_teaching_session": NODE_TEACHING_MODULE, # New task name
        "TEACHING_PAGE_INIT": NODE_TEACHING_MODULE, # From task_stage
        "TEACHING_PAGE_QA": NODE_TEACHING_MODULE, # From task_stage during QA
        "scaffolding_needed": NODE_SCAFFOLDING_MODULE,
        "feedback_needed": NODE_FEEDBACK_MODULE,
        "initiate_cowriting": NODE_COWRITING_MODULE,
        "initiate_pedagogy": NODE_PEDAGOGY_MODULE,
        "handle_student_clarification_question": NODE_CONVERSATION_HANDLER,
    }
    # --- Start Debug Logging ---
    logger.info(f"DEBUG: task_name is '{task_name}' (type: {type(task_name)})" )
    logger.info(f"DEBUG: routing_map keys are: {list(routing_map.keys())}")
    key_exists = task_name in routing_map
    logger.info(f"DEBUG: Does task_name exist as a key in routing_map? {key_exists}")
    # --- End Debug Logging ---

    route_destination = routing_map.get(task_name, NODE_CONVERSATION_HANDLER)

    # --- Add lesson_id to context if routing to teaching module ---
    if route_destination == NODE_TEACHING_MODULE:
        lesson_id = current_context.get("lesson_id")
        if lesson_id:
            state["current_context"] = {**current_context, "lesson_id": lesson_id}
            logger.info(f"Extracted lesson_id '{lesson_id}' for teaching module.")

    logger.info(f"INITIAL ROUTER: Final decision. Routing to -> [{route_destination}]")
    return route_destination


async def route_after_nlu(state: AgentGraphState) -> str:
    """Router that decides the path after the NLU/conversation handler node."""
    intent = state.get("classified_intent") or state.get("student_intent_for_rox_turn")
    if intent in ("CONFIRM_PROCEED_WITH_LO", "CONFIRM_START_TASK"):
        logger.info("NLU Router: Student confirmed LO. Routing to Pedagogical Strategy Planner.")
        return NODE_PEDAGOGICAL_STRATEGY_PLANNER
    else:
        # For any other intent (questions, rejections), we end the turn.
        # The AI's response is already in 'output_content' from the NLU node.
        logger.info(f"NLU Router: Intent is '{intent}'. Ending turn.")
        return END


async def main_modality_router(state: AgentGraphState) -> str:
    """The main router for executing the pedagogical plan step-by-step."""
    logger.info("--- Main Modality Router: Executing Micro-Plan ---")
    plan = state.get("pedagogical_plan")
    index = state.get("current_plan_step_index", 0)

    if not plan or index >= len(plan):
        logger.info("Modality Router: Plan complete or not found. Routing to Welcome Handler to propose next LO.")
        return NODE_HANDLE_WELCOME

    next_step = plan[index]
    modality = next_step.get("modality")
    logger.info(f"Modality Router: Executing step {index + 1}. Modality is '{modality}'.")

    # Map modality to the correct subgraph node name
    modality_map = {
        "TEACH": NODE_TEACHING_MODULE,
        "MODEL": NODE_MODELING_MODULE,
        "SCAFFOLD": NODE_SCAFFOLDING_MODULE,
        "FEEDBACK": NODE_FEEDBACK_MODULE,
    }

    destination = modality_map.get(modality)
    if destination:
        logger.info(f"Modality Router: Routing to {destination}.")
        return destination
    else:
        logger.warning(f"Modality Router: Unknown modality '{modality}'. Ending turn as a fallback.")
        return END


def build_graph(memory: AsyncSqliteSaver):
    """Builds the main application graph."""
    
    logger.info("--- Building Main TOEFL Tutor Graph ---")
    workflow = StateGraph(AgentGraphState)

    # --- 3. Add ALL Nodes and Subgraphs ---
    workflow.add_node(NODE_MODELING_MODULE, create_modeling_subgraph())
    workflow.add_node(NODE_TEACHING_MODULE, create_teaching_subgraph())
    workflow.add_node(NODE_SCAFFOLDING_MODULE, create_scaffolding_subgraph())
    workflow.add_node(NODE_FEEDBACK_MODULE, create_feedback_subgraph())
    workflow.add_node(NODE_COWRITING_MODULE, create_cowriting_subgraph())
    workflow.add_node(NODE_HANDLE_WELCOME, create_rox_welcome_subgraph())
    workflow.add_node(NODE_PEDAGOGY_MODULE, create_pedagogy_subgraph())

    # Add the simple nodes
    workflow.add_node(NODE_CONTEXT_MERGER, context_merger_node)
    workflow.add_node(NODE_ROUTER_ENTRY, router_entry_node)
    workflow.add_node(NODE_CONVERSATION_HANDLER, conversation_handler_node)
    workflow.add_node(NODE_PEDAGOGICAL_STRATEGY_PLANNER, pedagogical_strategy_planner_node)
    workflow.add_node(NODE_PLAN_ADVANCER, plan_advancer_node)
    workflow.add_node(NODE_INITIATE_MODULE_EXECUTION, initiate_module_execution_node)
    workflow.add_node(NODE_SAVE_INTERACTION, save_interaction_node)
    workflow.add_node(NODE_ACKNOWLEDGE_INTERRUPT, acknowledge_interrupt_node)

    # --- 4. Define the Graph's Flow ---
    workflow.set_entry_point(NODE_CONTEXT_MERGER)
    workflow.add_edge(NODE_CONTEXT_MERGER, NODE_ROUTER_ENTRY)

    # The initial router sends the user to the correct starting point
    workflow.add_conditional_edges(
        NODE_ROUTER_ENTRY,
        initial_router_logic,
        {
            NODE_ACKNOWLEDGE_INTERRUPT: NODE_ACKNOWLEDGE_INTERRUPT,
            NODE_MODELING_MODULE: NODE_MODELING_MODULE, # Legacy direct entry
            NODE_TEACHING_MODULE: NODE_TEACHING_MODULE, # Legacy direct entry
            NODE_SCAFFOLDING_MODULE: NODE_SCAFFOLDING_MODULE, # Legacy direct entry
            NODE_FEEDBACK_MODULE: NODE_FEEDBACK_MODULE, # Legacy direct entry
            NODE_COWRITING_MODULE: NODE_COWRITING_MODULE,
            NODE_PEDAGOGY_MODULE: NODE_PEDAGOGY_MODULE,
            NODE_HANDLE_WELCOME: NODE_HANDLE_WELCOME,
            NODE_CONVERSATION_HANDLER: NODE_CONVERSATION_HANDLER,
        }
    )

    # After the NLU handler, decide whether to start a lesson or end the turn
    workflow.add_conditional_edges(
        NODE_CONVERSATION_HANDLER,
        route_after_nlu,
        {
            NODE_PEDAGOGICAL_STRATEGY_PLANNER: NODE_PEDAGOGICAL_STRATEGY_PLANNER,
            END: END  # End the turn if the student asks a question or rejects the plan
        }
    )

    # The pedagogical planner creates the plan, then announces it.
    workflow.add_edge(NODE_PEDAGOGICAL_STRATEGY_PLANNER, NODE_INITIATE_MODULE_EXECUTION)

    # After announcing the plan, the turn is saved and ended.
    workflow.add_edge(NODE_INITIATE_MODULE_EXECUTION, NODE_SAVE_INTERACTION)

    # --- 5. Define the Exit Paths ---
    # Nodes that are not part of the main pedagogical loop but should still save their interaction.
    nodes_leading_to_save = [
        NODE_ACKNOWLEDGE_INTERRUPT,
        NODE_HANDLE_WELCOME, # Saves the curriculum proposal interaction
        NODE_COWRITING_MODULE,
        NODE_PEDAGOGY_MODULE,
        # The main modality execution loop is now disabled in favor of the 'announce' node.
        # Re-enable these if you revert to the continuous execution model.
        # NODE_TEACHING_MODULE,
        # NODE_MODELING_MODULE,
        # NODE_SCAFFOLDING_MODULE,
        # NODE_FEEDBACK_MODULE
    ]
    for node_name in nodes_leading_to_save:
        workflow.add_edge(node_name, NODE_SAVE_INTERACTION)

    workflow.add_edge(NODE_SAVE_INTERACTION, END)

    # --- Compile the Final Graph ---
    logger.info("--- Main Graph Compilation ---")
    compiled_graph = workflow.compile(checkpointer=memory)
    logger.info("--- Main Graph Compiled Successfully ---")
    
    return compiled_graph