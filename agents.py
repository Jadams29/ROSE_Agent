from typing import Literal

from langgraph.graph import StateGraph, END

from state import AgentState
from tools import ROSETools


class ROSEAgent:
    """Constructs and compiles the LangGraph for the ROSE agent."""

    def __init__(self, llm_model_name="gemini-2.5-pro"):
        """Initialize the ROSE agent with tools."""
        self.tools = ROSETools(llm_model_name=llm_model_name)

    # --- Node Functions ---
    # These wrapper functions call the corresponding tool methods

    def decompose_goal_node(self, state: AgentState) -> dict:
        """Node wrapper for goal decomposition tool."""
        return self.tools.decompose_goal(state)

    def strategize_revision_node(self, state: AgentState) -> dict:
        """Node wrapper for revision strategy tool."""
        return self.tools.strategize_revision(state)

    def generate_prompt_node(self, state: AgentState) -> dict:
        """Node wrapper for prompt generation tool."""
        return self.tools.generate_prompt(state)

    def evaluate_improvement_node(self, state: AgentState) -> dict:
        """Node wrapper for evaluation tool."""
        return self.tools.evaluate_improvement(state)

    # --- Conditional Logic ---
    def should_continue(self, state: AgentState) -> str:
        """
        Phase 3, Step 6: Decides whether to end or continue refining.
        This is the "Act/Adjust" logic.
        """
        print("--- 🤔 DECISION POINT ---")
        if state["iteration_count"] >= state["max_iterations"]:
            print("🛑 Max iterations reached. Ending process.")
            return "end"

        evaluation = state["evaluation"]
        if evaluation.is_improvement_sufficient:
            print("🏆 Improvement is sufficient. Task complete.")
            return "end"
        else:
            print("ループ... Improvement not sufficient. Refining again.")  # "ループ" is Japanese for "loop"
            return "continue_refining"

    def get_graph(self) -> StateGraph:
        """Builds and returns the LangGraph StateGraph."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("decompose_goal", self.decompose_goal_node)
        workflow.add_node("strategize_revision", self.strategize_revision_node)
        workflow.add_node("generate_prompt", self.generate_prompt_node)
        workflow.add_node("evaluate_improvement", self.evaluate_improvement_node)

        # Set entry point
        workflow.set_entry_point("decompose_goal")

        # Add edges
        workflow.add_edge("decompose_goal", "strategize_revision")
        workflow.add_edge("strategize_revision", "generate_prompt")
        workflow.add_edge("generate_prompt", "evaluate_improvement")

        # Add the conditional edge for the self-correction loop
        workflow.add_conditional_edges(
            "evaluate_improvement",
            self.should_continue,
            {
                "continue_refining": "strategize_revision",
                "end": END,
            },
        )

        return workflow.compile()