from typing import List, TypedDict, Optional
from pydantic import BaseModel, Field


# --- Pydantic Models for Structured Output ---
# These models define the expected structure for each LLM tool's output.

class DecomposedGoal(BaseModel):
    """The structured output for the goal decomposition tool."""
    criteria: List[str] = Field(description="A list of specific, actionable criteria to improve the prompt.")


class RevisionPlan(BaseModel):
    """The structured output for the revision strategy tool."""
    plan: List[str] = Field(description="A step-by-step plan to revise the prompt.")


class GeneratedPrompt(BaseModel):
    """The structured output for the prompt generation tool."""
    new_prompt: str = Field(description="The full text of the newly generated prompt.")


class Evaluation(BaseModel):
    """The structured output for the self-correction/evaluation tool."""
    score: int = Field(description="An overall score from 1 to 10 for the improvement.")
    rationale: str = Field(description="The reasoning behind the score, highlighting successes and failures.")
    is_improvement_sufficient: bool = Field(description="Whether the prompt is now good enough (score >= 8).")


# --- Agent State ---
# This dictionary holds the agent's memory and passes data between nodes in the graph.

class AgentState(TypedDict):
    initial_prompt: str
    goal: str
    decomposed_goal: Optional[DecomposedGoal]
    revision_plan: Optional[RevisionPlan]
    current_prompt: str  # This will be updated with the latest generated prompt
    evaluation: Optional[Evaluation]
    iteration_count: int
    max_iterations: int