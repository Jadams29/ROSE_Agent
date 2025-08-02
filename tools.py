import json
from typing import Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from state import AgentState, DecomposedGoal, RevisionPlan, GeneratedPrompt, Evaluation


class ROSETools:
    """A class to encapsulate all the tools for the ROSE agent."""

    def __init__(self, llm_model_name="gemini-2.5-pro"):
        """Initializes the toolset with a specific Gemini model."""
        self.model_name = llm_model_name
        print(f"--- ROSE Tools initialized with model: {llm_model_name} ---")

    def _get_llm(self, temperature: float) -> ChatGoogleGenerativeAI:
        """Helper to get LLM with specific temperature."""
        return ChatGoogleGenerativeAI(model=self.model_name, temperature=temperature)

    def decompose_goal(self, state: AgentState) -> Dict:
        """
        Phase 1, Step 2: Decomposes the user's goal into actionable criteria.
        This is the "Orient" step.
        """
        print("--- 🧭 ORIENT: DECOMPOSING GOAL ---")

        prompt_template = PromptTemplate(
            template="""You are an expert prompt engineer and a meticulous project manager. Your task is to decompose a user's high-level goal for a prompt into a list of specific, actionable, and verifiable criteria. These criteria will serve as a checklist to guide the prompt's revision and to evaluate the final result.

        Analyze the provided User's Initial Prompt and their Goal. Based on them, generate a JSON object containing a single key "criteria" which is a list of strings. Each string in the list should be a distinct, clear, and actionable instruction.

        **User's Initial Prompt:**
        "{initial_prompt}"

        **User's Goal:**
        "{goal}"

        **Instructions:**
        1.  Focus on what needs to be *added, removed, or changed* in the prompt to meet the goal.
        2.  The criteria should be concrete. For example, instead of "make it more creative," a better criterion would be "Add a constraint that the story must be told from the perspective of an inanimate object."
        3.  Ensure the criteria directly address the user's goal.

        {format_instructions}
        """,
            input_variables=["initial_prompt", "goal"],
            partial_variables={
                "format_instructions": PydanticOutputParser(pydantic_object=DecomposedGoal).get_format_instructions()},
        )

        llm = self._get_llm(temperature=0.2)
        parser = PydanticOutputParser(pydantic_object=DecomposedGoal)

        formatted_prompt = prompt_template.format(initial_prompt=state["initial_prompt"], goal=state["goal"])
        response = llm.invoke(formatted_prompt)
        parsed_response = parser.parse(response.content)

        print(f"✅ Decomposed Goal into Criteria: {json.dumps(parsed_response.dict(), indent=2)}")

        return {"decomposed_goal": parsed_response, "current_prompt": state["initial_prompt"]}

    def strategize_revision(self, state: AgentState) -> Dict:
        """
        Phase 2, Step 3: Creates a plan to revise the prompt based on criteria.
        This is the "Plan" step.
        """
        print("--- ✍️ PLAN: STRATEGIZING REVISION ---")

        # Provide evaluation feedback if it exists from a previous loop
        feedback = state["evaluation"].rationale if state.get("evaluation") else "N/A"

        prompt_template = PromptTemplate(
            template="""You are a master prompt strategist. Your job is to create a clear, step-by-step plan to revise a prompt based on a set of improvement criteria. You are not writing the new prompt yet, only creating the plan to do so.

        **The Current Prompt:**
        "{current_prompt}"

        **Improvement Criteria Checklist:**
        {decomposed_goal}

        **Previous Evaluation Feedback (if any):**
        {evaluation_feedback}

        **Instructions:**
        1.  Review the current prompt and the criteria.
        2.  If there is previous feedback, prioritize addressing the points of failure.
        3.  Create a concise, step-by-step plan of action for the revision.

        {format_instructions}
        """,
            input_variables=["current_prompt", "decomposed_goal", "evaluation_feedback"],
            partial_variables={
                "format_instructions": PydanticOutputParser(pydantic_object=RevisionPlan).get_format_instructions()},
        )

        llm = self._get_llm(temperature=0.5)
        parser = PydanticOutputParser(pydantic_object=RevisionPlan)

        formatted_prompt = prompt_template.format(
            current_prompt=state["current_prompt"],
            decomposed_goal=state["decomposed_goal"].dict(),
            evaluation_feedback=feedback
        )
        response = llm.invoke(formatted_prompt)
        parsed_response = parser.parse(response.content)

        print(f"✅ Created Revision Plan: {json.dumps(parsed_response.dict(), indent=2)}")

        return {"revision_plan": parsed_response}

    def generate_prompt(self, state: AgentState) -> Dict:
        """
        Phase 2, Step 4: Executes the revision plan to generate a new prompt.
        This is the "Do" step.
        """
        print("--- ✍️ DO: GENERATING NEW PROMPT ---")

        prompt_template = PromptTemplate(
            template="""You are an expert AI prompt writer. Your task is to execute a revision plan to create a new, improved version of a prompt. Follow the plan precisely to generate the final text.

        **The Original Prompt:**
        "{current_prompt}"

        **The Revision Plan:**
        {revision_plan}

        **Instructions:**
        1.  Carefully implement each step in the revision plan.
        2.  The output should ONLY be the full text of the newly generated prompt inside a JSON object. Do not include any explanation or preamble.

        {format_instructions}
        """,
            input_variables=["current_prompt", "revision_plan"],
            partial_variables={
                "format_instructions": PydanticOutputParser(pydantic_object=GeneratedPrompt).get_format_instructions()},
        )

        llm = self._get_llm(temperature=0.7)
        parser = PydanticOutputParser(pydantic_object=GeneratedPrompt)

        formatted_prompt = prompt_template.format(
            current_prompt=state["current_prompt"],
            revision_plan=state["revision_plan"].dict()
        )
        response = llm.invoke(formatted_prompt)
        parsed_response = parser.parse(response.content)

        print(f"✅ Generated New Prompt Version.")

        return {"current_prompt": parsed_response.new_prompt, "iteration_count": state["iteration_count"] + 1}

    def evaluate_improvement(self, state: AgentState) -> Dict:
        """
        Phase 3, Step 5: Evaluates the new prompt against the criteria.
        This is the "Check" step for self-correction.
        """
        print("--- ✅ CHECK: EVALUATING IMPROVEMENT ---")

        prompt_template = PromptTemplate(
            template="""You are a meticulous Quality Assurance analyst for AI prompts. Your task is to evaluate a revised prompt against an original prompt based on a specific set of criteria. You must be objective and critical.

        **The Original Prompt:**
        "{initial_prompt}"

        **The Newly Revised Prompt:**
        "{new_prompt}"

        **Improvement Criteria Checklist:**
        {decomposed_goal}

        **Instructions:**
        1.  For each criterion in the checklist, assess if the new prompt successfully meets it.
        2.  Provide an overall score from 1 (no improvement) to 10 (perfectly met all criteria).
        3.  Write a brief, honest rationale for your score, explaining what was done well and what (if anything) is still missing or could be better.
        4.  Based on your analysis, determine if the improvement is sufficient to consider the task complete. The improvement is sufficient if the score is 8 or higher.

        {format_instructions}
        """,
            input_variables=["initial_prompt", "new_prompt", "decomposed_goal"],
            partial_variables={
                "format_instructions": PydanticOutputParser(pydantic_object=Evaluation).get_format_instructions()},
        )

        llm = self._get_llm(temperature=0.0)  # Low temp for objective evaluation
        parser = PydanticOutputParser(pydantic_object=Evaluation)

        formatted_prompt = prompt_template.format(
            initial_prompt=state["initial_prompt"],
            new_prompt=state["current_prompt"],
            decomposed_goal=state["decomposed_goal"].dict()
        )
        response = llm.invoke(formatted_prompt)
        parsed_response = parser.parse(response.content)

        print(f"✅ Evaluation Complete: {json.dumps(parsed_response.dict(), indent=2)}")

        return {"evaluation": parsed_response}