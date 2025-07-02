# graph/teaching_flow.py
import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState

logger = logging.getLogger(__name__)

# Import node functions
from agents.teaching_planner_node import teaching_planner_node
from agents.teaching_delivery_node import teaching_delivery_generator_node
from agents.teaching_plan_advancer_node import teaching_plan_advancer_node
from agents.teaching_qa_handler_node import teaching_qa_handler_node
from agents.teaching_RAG_document_node import teaching_RAG_document_node

# --- Node Names ---
NODE_TEACHING_RAG = "teaching_rag_document"
NODE_TEACHING_PLANNER = "teaching_planner"
NODE_TEACHING_DELIVERY_GENERATOR = "teaching_delivery_generator"
NODE_TEACHING_QA = "teaching_qa"
NODE_TEACHING_PLAN_ADVANCER = "teaching_plan_advancer"
NODE_CHECK_PLAN_COMPLETION = "check_plan_completion" # New placeholder node

# --- Placeholder & Router Functions ---

async def check_plan_completion_node(state: AgentGraphState) -> dict:
    """A placeholder node to attach the delivery_or_end_router to."""
    logger.info("--- Checking if lesson plan is complete --- ")
    return {}

async def entry_router(state: AgentGraphState) -> str:
    """Routes based on the task and plan existence."""
    task_name = state.get("task_name")
    logger.info(f"Teaching Subgraph: Routing for task '{task_name}'.")

    # If the task is a follow-up Q&A, go directly to the handler
    if task_name == "TEACHING_PAGE_QA":
        logger.info("Teaching Subgraph: QA task. Routing to QA handler.")
        return NODE_TEACHING_QA

    # For initial tasks, check if a plan already exists
    if state.get("pedagogical_plan"):
        logger.info("Teaching Subgraph: Active plan found. Routing to check completion.")
        return NODE_CHECK_PLAN_COMPLETION
    else:
        logger.info("Teaching Subgraph: No plan found. Routing to RAG for planning.")
        return NODE_TEACHING_RAG

async def delivery_or_end_router(state: AgentGraphState) -> str:
    """Checks if the lesson plan is complete before delivering the next step."""
    plan = state.get("pedagogical_plan")
    current_index = state.get("current_plan_step_index", 0)

    # --- Enhanced Debug Logging ---
    logger.info(f"ROUTER_DEBUG: Type of plan: {type(plan)}")
    logger.info(f"ROUTER_DEBUG: Plan content: {plan}")
    logger.info(f"ROUTER_DEBUG: Plan length: {len(plan) if isinstance(plan, list) else 'N/A'}")
    logger.info(f"ROUTER_DEBUG: Current index: {current_index} (type: {type(current_index)})")
    # --- End Debug Logging ---

    if not plan or current_index >= len(plan):
        logger.info("Plan is complete or absent. Ending teaching module.")
        return END
    else:
        logger.info(f"Plan active at step {current_index + 1}. Routing to RAG for content.")
        return NODE_TEACHING_RAG

async def post_rag_router(state: AgentGraphState) -> str:
    """After RAG, routes to the planner if no plan exists, or delivery if one does."""
    if state.get("pedagogical_plan"):
        logger.info("Post-RAG: Plan exists. Routing to delivery generator.")
        return NODE_TEACHING_DELIVERY_GENERATOR
    else:
        logger.info("Post-RAG: No plan. Routing to planner.")
        return NODE_TEACHING_PLANNER

async def after_interaction_router(state: AgentGraphState) -> str:
    """Routes after student interaction (QA). Decides whether to advance or re-explain."""
    intent = state.get("student_intent_for_lesson_turn", "CONFIRM_UNDERSTANDING")
    logger.info(f"Post-Interaction Router: Student intent is '{intent}'")
    if intent == "CONFIRM_UNDERSTANDING":
        return NODE_TEACHING_PLAN_ADVANCER
    else: # If student is confused, re-run the delivery generator for the same step.
        logger.info("Student expressed confusion. Re-running RAG and delivery for the current step.")
        return NODE_TEACHING_RAG

# --- Graph Definition ---

def create_teaching_subgraph():
    """Creates the simplified LangGraph subgraph for the teaching module."""
    workflow = StateGraph(AgentGraphState)

    # Add Nodes
    workflow.add_node(NODE_TEACHING_PLANNER, teaching_planner_node)
    workflow.add_node(NODE_TEACHING_RAG, teaching_RAG_document_node)
    workflow.add_node(NODE_TEACHING_DELIVERY_GENERATOR, teaching_delivery_generator_node)
    workflow.add_node(NODE_TEACHING_QA, teaching_qa_handler_node)
    workflow.add_node(NODE_TEACHING_PLAN_ADVANCER, teaching_plan_advancer_node)
    workflow.add_node(NODE_CHECK_PLAN_COMPLETION, check_plan_completion_node)

    # --- Define Flow ---

    # 1. Entry Point: Plan, QA, or Deliver?
    workflow.set_conditional_entry_point(
        entry_router,
        {
            NODE_TEACHING_RAG: NODE_TEACHING_RAG,
            NODE_CHECK_PLAN_COMPLETION: NODE_CHECK_PLAN_COMPLETION,
            NODE_TEACHING_QA: NODE_TEACHING_QA # New entry for follow-up
        }
    )

    # 2. Post-RAG Routing: To Planner or to Delivery?
    workflow.add_conditional_edges(
        NODE_TEACHING_RAG,
        post_rag_router,
        {
            NODE_TEACHING_PLANNER: NODE_TEACHING_PLANNER,
            NODE_TEACHING_DELIVERY_GENERATOR: NODE_TEACHING_DELIVERY_GENERATOR
        }
    )

    # 3. Planning Flow
    workflow.add_edge(NODE_TEACHING_PLANNER, NODE_CHECK_PLAN_COMPLETION)

    # 4. Delivery Loop Entry: Check if plan is complete
    workflow.add_conditional_edges(
        NODE_CHECK_PLAN_COMPLETION, 
        delivery_or_end_router,
        {
            NODE_TEACHING_RAG: NODE_TEACHING_RAG, # If not done, get content for next step
            END: END # If done, exit
        }
    )

    # 5. Delivery Content Generation -> ENDS the flow for this turn.
    workflow.add_edge(NODE_TEACHING_DELIVERY_GENERATOR, END)

    # 6. Interaction Handling (from a new entry point)
    # The QA handler should be a terminal node for the turn. It answers the question,
    # and then the graph should wait for the next user input.
    workflow.add_edge(NODE_TEACHING_QA, END)

    workflow.add_edge(NODE_TEACHING_PLAN_ADVANCER, NODE_CHECK_PLAN_COMPLETION)

    return workflow.compile()