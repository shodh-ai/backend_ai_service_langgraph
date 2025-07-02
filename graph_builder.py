# FINAL, PERFECTED graph/graph_builder.py

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from state import AgentGraphState
from agents import save_interaction_node
from agents.acknowledge_interrupt_node import acknowledge_interrupt_node
from agents.conversation_handler import conversation_handler_node
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
from graph.welcome_flow import create_welcome_subgraph
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
    workflow.add_node(NODE_HANDLE_WELCOME, create_welcome_subgraph())
    workflow.add_node(NODE_PEDAGOGY_MODULE, create_pedagogy_subgraph())

    # Add the simple nodes
    workflow.add_node(NODE_CONTEXT_MERGER, context_merger_node)
    workflow.add_node(NODE_ROUTER_ENTRY, router_entry_node)
    workflow.add_node(NODE_CONVERSATION_HANDLER, conversation_handler_node)
    workflow.add_node(NODE_SAVE_INTERACTION, save_interaction_node)
    workflow.add_node(NODE_ACKNOWLEDGE_INTERRUPT, acknowledge_interrupt_node)

    # --- 4. Define the Graph's Flow ---
    workflow.set_entry_point(NODE_CONTEXT_MERGER)

    # The merger node unconditionally proceeds to the main router
    workflow.add_edge(NODE_CONTEXT_MERGER, NODE_ROUTER_ENTRY)

    workflow.add_conditional_edges(
        NODE_ROUTER_ENTRY,
        initial_router_logic,
        {
            NODE_ACKNOWLEDGE_INTERRUPT: NODE_ACKNOWLEDGE_INTERRUPT,
            NODE_MODELING_MODULE: NODE_MODELING_MODULE,
            NODE_TEACHING_MODULE: NODE_TEACHING_MODULE,
            NODE_SCAFFOLDING_MODULE: NODE_SCAFFOLDING_MODULE,
            NODE_FEEDBACK_MODULE: NODE_FEEDBACK_MODULE,
            NODE_COWRITING_MODULE: NODE_COWRITING_MODULE,
            NODE_PEDAGOGY_MODULE: NODE_PEDAGOGY_MODULE,
            NODE_HANDLE_WELCOME: NODE_HANDLE_WELCOME,
            NODE_CONVERSATION_HANDLER: NODE_CONVERSATION_HANDLER,
        }
    )

    # --- 5. Define the Exit Paths ---
    all_flow_nodes = [
        NODE_MODELING_MODULE, NODE_TEACHING_MODULE, NODE_SCAFFOLDING_MODULE,
        NODE_FEEDBACK_MODULE, NODE_COWRITING_MODULE, NODE_PEDAGOGY_MODULE,
        NODE_HANDLE_WELCOME, NODE_CONVERSATION_HANDLER, NODE_ACKNOWLEDGE_INTERRUPT
    ]
    for node_name in all_flow_nodes:
        workflow.add_edge(node_name, NODE_SAVE_INTERACTION)

    workflow.add_edge(NODE_SAVE_INTERACTION, END)

    # --- Compile the Final Graph ---
    logger.info("--- Main Graph Compilation ---")

    # Compile the graph with the checkpointer
    # All nodes will eventually lead to the save_interaction_node, which is the end of a turn.
    compiled_graph = workflow.compile(checkpointer=memory)

    logger.info("--- Main Graph Compiled Successfully ---")
    
    return compiled_graph