# agents/__init__.py

# This file serves as the central index for all agent nodes in the application.
# It has been refactored to support the standardized RAG -> Generator -> Formatter architecture.

# ===================================
# Core System Nodes
# ===================================
from .student_model_node import (load_student_data_node, save_interaction_node)
from .initiate_module_execution_node import initiate_module_execution_node
from .conversational_manager_node import handle_home_greeting_node as handle_welcome_node
from .curriculum_navigator_node import curriculum_navigator_node
from .pedagogical_strategy_planner_node import pedagogical_strategy_planner_node
from .plan_advancer_node import plan_advancer_node
from .conversation_handler import conversation_handler_node

# ===================================
# Standardized Flow Nodes
# ===================================

# --- Cowriting Flow ---
from .cowriting_RAG_document_node import cowriting_RAG_document_node
from .cowriting_generator import cowriting_generator_node
from .cowriting_output_formatter import cowriting_output_formatter_node

# --- Feedback Flow ---
from .feedback_RAG_document_node import feedback_RAG_document_node
from .feedback_generator import feedback_generator_node
from .feedback_output_formatter import feedback_output_formatter_node

# --- Scaffolding Flow ---
from .scaffolding_RAG_document_node import scaffolding_RAG_document_node
from .scaffolding_generator import scaffolding_generator_node
from .scaffolding_output_formatter import scaffolding_output_formatter_node

# --- Modeling Flow ---
from .modelling_RAG_document_node import modelling_RAG_document_node
from .modelling_generator import modelling_generator_node as modelling_generator
from .modelling_output_formatter import modelling_output_formatter_node as modelling_output_formatter

# --- Teaching Flow ---
from .teaching_RAG_document_node import teaching_RAG_document_node
from .teaching_generator import teaching_generator_node
from .teaching_output_formatter import teaching_output_formatter_node

# --- Pedagogy Flow ---


# ===================================
# General Purpose & Legacy Nodes
# ===================================
from .error_generator import error_generator_node
from .initial_report_generation import initial_report_generation_node

# ===================================
# Public API for the 'agents' package
# ===================================
__all__ = [
    # Core System
    "load_student_data_node",
    "save_interaction_node",
    "handle_welcome_node",
    "curriculum_navigator_node",
    "initiate_module_execution_node",


    # General Conversation
    "conversation_handler_node",

    # Cowriting Flow
    "cowriting_RAG_document_node",
    "cowriting_generator_node",
    "cowriting_output_formatter_node",

    # Feedback Flow
    "feedback_RAG_document_node",
    "feedback_generator_node",
    "feedback_output_formatter_node",

    # Scaffolding Flow
    "scaffolding_RAG_document_node",
    "scaffolding_generator_node",
    "scaffolding_output_formatter_node",

    # Modeling Flow
    "modelling_RAG_document_node",
    "modelling_generator",
    "modelling_output_formatter",

    # Teaching Flow
    "teaching_RAG_document_node",
    "teaching_generator_node",
    "teaching_output_formatter_node",



    # General & Legacy
    "error_generator_node",
    "initial_report_generation_node",
]
