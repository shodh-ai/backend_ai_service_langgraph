# graph/modeling_flow.py
import logging
from langgraph.graph import StateGraph, END
from state import AgentGraphState
from typing import Literal

# 1. Import the agent node functions
from agents.modelling_RAG_document_node import modelling_RAG_document_node
from agents.modelling_planner_node import modelling_planner_node
from agents.modelling_delivery_generator_node import modelling_delivery_node
from agents.modelling_plan_advancer_node import modelling_plan_advancer_node
from agents.modelling_qa_handler_node import modelling_qa_handler_node

logger = logging.getLogger(__name__)

# 2. Define standardized node names
NODE_MODELING_RAG = "modelling_rag"
NODE_MODELING_PLANNER = "modelling_planner"
NODE_MODELING_DELIVERY = "modelling_delivery"
NODE_MODELING_PLAN_ADVANCER = "modelling_plan_advancer"
NODE_MODELING_QA = "modelling_qa"

# 3. Define conditional routing functions
def should_continue_modeling(state: AgentGraphState) -> Literal["deliver_step", "__end__"]:
    """Router to decide if the modeling plan has more steps to deliver."""
    logger.debug("--- Modeling Router: Checking if plan should continue ---")
    plan = state.get("modelling_plan", []) # Use modelling_plan
    if not plan:
        logger.warning("Modeling Router: No modeling plan found. Ending flow.")
        return "__end__"

    current_step_index = state.get("current_plan_step_index", 0)
    total_steps = len(plan)

    if current_step_index >= total_steps:
        logger.info(f"Modeling Router: Plan complete ({current_step_index}/{total_steps}). Ending flow.")
        return "__end__"
    else:
        logger.info(f"Modeling Router: Plan continues to step {current_step_index + 1}/{total_steps}.")
        return "deliver_step"

# 4. Define the subgraph creation function
def create_modeling_subgraph():
    """
    Creates a LangGraph subgraph for the plan-based modeling flow.
    This flow uses a single RAG query, then orchestrates a planner, delivery, QA, and plan advancer.
    """
    workflow = StateGraph(AgentGraphState)

    # Add the nodes
    workflow.add_node(NODE_MODELING_RAG, modelling_RAG_document_node)
    workflow.add_node(NODE_MODELING_PLANNER, modelling_planner_node)
    workflow.add_node(NODE_MODELING_DELIVERY, modelling_delivery_node)
    workflow.add_node(NODE_MODELING_QA, modelling_qa_handler_node)
    workflow.add_node(NODE_MODELING_PLAN_ADVANCER, modelling_plan_advancer_node)

    # Define the flow
    # Start with a single RAG query to fetch all context needed for the session.
    workflow.set_entry_point(NODE_MODELING_RAG)
    # The RAG results are then passed to the planner.
    workflow.add_edge(NODE_MODELING_RAG, NODE_MODELING_PLANNER)

    # After planning, decide whether to start delivery or end
    workflow.add_conditional_edges(
        NODE_MODELING_PLANNER,
        should_continue_modeling,
        {
            "deliver_step": NODE_MODELING_DELIVERY,
            "__end__": END
        }
    )

    # The core modeling loop
    workflow.add_edge(NODE_MODELING_DELIVERY, NODE_MODELING_QA)
    workflow.add_edge(NODE_MODELING_QA, NODE_MODELING_PLAN_ADVANCER)

    # After advancing the plan, loop back to the router to check the next step
    workflow.add_conditional_edges(
        NODE_MODELING_PLAN_ADVANCER,
        should_continue_modeling,
        {
            "deliver_step": NODE_MODELING_DELIVERY,
            "__end__": END
        }
    )

    # Compile and return the subgraph
    return workflow.compile()
