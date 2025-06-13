from typing import TypedDict, List, Optional, Dict, Any, Union
from models import InteractionRequestContext  # Import your Pydantic model


class AgentGraphState(TypedDict, total=False):
    # Teaching System Specific State
    teacher_persona: Optional[str]
    learning_objective_id: Optional[str] # To match CSV's LEARNING_OBJECTIVE (e.g., SPK_GEN_FLU_001)
    student_proficiency_level: Optional[str] # To match CSV's STUDENT_PROFICIENCY (e.g., Beginner)
    current_student_affective_state: Optional[str] # To match CSV's STUDENT_AFFECTIVE_STATE (e.g., Frustration)
    current_lesson_step_number: Optional[int] # 1-indexed, to select the specific part of the lesson

    retrieved_teaching_row: Optional[Dict[str, Any]] # The full row from CSV, retrieved by RAG
    
    teaching_output_content: Optional[Dict[str, Any]] # Output from TeachingDeliveryNode, e.g., {"text_for_tts": "...", "ui_actions": [...]}

    # User and session identifiers
    user_id: str
    user_token: str
    session_id: str

    # Input data from the current interaction
    transcript: Optional[str]  # The most recent transcript from the user
    full_submitted_transcript: Optional[
        str
    ]  # Complete transcript when a task is submitted (P2->P5)
    current_context: (
        InteractionRequestContext  # Current UI state and context information
    )
    chat_history: Optional[List[Dict[str, str]]]  # Previous conversation turns
    question_stage: Optional[str]  # Current question stage

    # Student model and memory data
    student_memory_context: Optional[
        Dict[str, Any]
    ]  # Data retrieved from memory system

    # Task management
    next_task_details: Optional[
        Dict[str, Any]
    ]  # Information about the next task to present

    # Processing results
    diagnosis_result: Optional[Dict[str, Any]]  # Results from diagnostic analysis
    error_details: Optional[Dict[str, Any]] # Detailed error information
    document_query_result: Optional[Dict[str, Any]] # Result from document querying
    rag_query_result: Optional[Dict[str, Any]] # Result from RAG querying

    # Feedback flow specific fields
    feedback_plan: Optional[Dict[str, Any]]
    feedback_output: Optional[Dict[str, Any]]

    # Scaffolding flow specific fields (additional)
    scaffolding_analysis: Optional[Dict[str, Any]]
    scaffolding_retrieval_result: Optional[Dict[str, Any]]
    scaffolding_plan: Optional[Dict[str, Any]]

    # Teaching module state
    teaching_module_state: Optional[Dict[str, Any]]

    # Curriculum / P1 specific outputs
    p1_curriculum_navigator_output: Optional[Dict[str, Any]]

    # General conversation response holder
    conversation_response: Optional[Dict[str, Any]]

    # Task stage (also in current_context, but explicitly set in app.py)
    task_stage: Optional[str]

    # Data from specific welcome flow nodes
    greeting_data: Optional[Dict[str, str]]  # Expected: {"greeting_tts": "..."}
    task_suggestion_llm_output: Optional[
        Dict[str, str]
    ]  # Expected: {"task_suggestion_tts": "..."}

    # Output data to be returned to the frontend
    output_content: Optional[
        Dict[str, Any]
    ]  # Contains text_for_tts, ui_actions, and frontend_rpc_calls

    # Raw outputs from various systems, passed through by OutputFormatterNode
    # These are intended for the client response but also need to be in state 
    # if subsequent internal nodes (e.g. for saving to memory) need them.
    raw_teaching_output: Optional[Dict[str, Any]]
    raw_modelling_output: Optional[Dict[str, Any]]
    raw_feedback_output: Optional[Dict[str, Any]]
    raw_pedagogy_output: Optional[Dict[str, Any]]
    initial_report_content: Optional[Dict[str, Any]]

    # For backward compatibility during transition
    feedback_content: Optional[
        Dict[str, Any]
    ]  # To be deprecated in favor of output_content


    # Scaffolding-specific fields
    primary_struggle: Optional[str]  # Main struggle identified for scaffolding
    scaffolding_strategies: Optional[
        List[Dict[str, Any]]
    ]  # List of possible scaffolding strategies
    selected_scaffold_type: Optional[str]  # Type of scaffold chosen for this interaction
    scaffold_adaptation_plan: Optional[str]  # Plan for adapting the scaffold
    scaffold_content_type: Optional[str]  # Type of content (template, bank, etc.)
    scaffold_content_name: Optional[str]  # Name of the scaffold content
    scaffold_content: Optional[
        Dict[str, Any]
    ]  # The generated scaffolding content details
    scaffolding_output: Optional[Dict[str, Any]]  # Formatted output for the UI

    # Cowriting-specific fields
    student_written_chunk: Optional[str]  # Current chunk of text written by student
    student_articulated_thought: Optional[str]  # Student's verbalized thought about their writing
    writing_task_context: Optional[Dict[str, str]]  # Details about the writing task section
    cowriting_lo_focus: Optional[str]  # Learning objective focus for cowriting
    student_comfort_level: Optional[str]  # Student's comfort with the task
    student_affective_state: Optional[str]  # Student's current emotional/affective state
    cowriting_strategies: Optional[List[Dict[str, Any]]]  # List of possible cowriting strategies
    selected_cowriting_intervention: Optional[Dict[str, Any]]  # Selected intervention plan
    cowriting_output: Optional[Dict[str, Any]]  # Formatted cowriting output for UI
    # Modelling System Data
    modelling_document_data: Optional[List[Dict[str, Any]]] # Data from modelling_query_document_node
    example_prompt_text: Optional[str]
    student_goal_context: Optional[str]
    student_confidence_context: Optional[str]
    teacher_initial_impression: Optional[str]
    student_struggle_context: Optional[str]
    english_comfort_level: Optional[str]
    intermediate_modelling_payload: Optional[Dict[str, Any]] # Payload from modelling_generator_node
    modelling_output_content: Optional[Dict[str, Any]] # Payload from modelling_output_formatter_node

    # Keys from OutputFormatterNode for final client response
    final_text_for_tts: Optional[str]
    final_ui_actions: Optional[List[Any]] # Can be list of dicts or ReactUIAction Pydantic models
    final_next_task_info: Optional[Dict[str, Any]]
    final_navigation_instruction: Optional[Dict[str, Any]]
    raw_modelling_output: Optional[Dict[str, Any]] # Raw output from modelling_generator_node, passed through
    # Flattened keys from modelling_output_content, also passed through by OutputFormatterNode
    think_aloud_sequence: Optional[List[Dict[str, Any]]]
    pre_modeling_setup_script: Optional[str]
    post_modeling_summary_and_key_takeaways: Optional[str]
    comprehension_check_or_reflection_prompt_for_student: Optional[str]
    adaptation_notes: Optional[str]

    llm_instruction: Optional[
        str
    ]  # Instruction for the LLM, typically from a prompt node
    user_data: Optional[Dict[str, Any]]  # User data, typically from the frontend

    navigation_tts: Optional[str]  # TTS from prepare_navigation_node
    ui_actions_for_formatter: Optional[List[Dict[str, Any]]] # UI actions to be consolidated by formatter
    conversational_tts: Optional[str]  # TTS from conversation_handler_node or similar

    # For motivational support node
    triggering_event_for_motivation: Optional[str] # Context for why motivational support is being triggered
    next_node_hint_from_motivation: Optional[str] # Suggestion from motivational node for the next graph step

    # For session wrap-up
    session_is_ending: Optional[bool] = False
    final_session_data_to_save: Optional[Dict[str, Any]] = None

    primary_error: Optional[str]  # Primary error from the LLM
    explanation: Optional[str]  # Explanation for the primary error
    document_data: Optional[List[Dict]]

    estimated_overall_english_comfort_level: Optional[str]
    initial_impression: Optional[str]
    speaking_strengths: Optional[str]
    fluency: Optional[str]
    grammar: Optional[str]
    vocabulary: Optional[str]
    question_one_answer: Optional[str]
    question_two_answer: Optional[str]
    question_three_answer: Optional[str]

    prioritized_issue: Optional[str]
    chosen_pedagogical_strategy: Optional[str]
