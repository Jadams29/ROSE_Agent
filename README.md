## Autonomous Agent Architecture: R.O.S.E.

I've designed a hybrid architecture named **R.O.S.E.**, which stands for **Reflective-Oriented Synthesis and Execution**. This model synthesizes the strongest aspects of OODA, ReAct, and PDCA to create a robust, self-correcting loop ideal for a creative and analytical task like prompt improvement.

* **From OODA (Observe, Orient, Decide, Act):** We take the critical **Orient** phase. Before acting, the agent must first deeply understand the user's *intent* by deconstructing their goal.
* **From ReAct (Reason, Act):** We adopt the core operational loop of using a tool based on a reasoned plan. The agent will **Plan** its changes and then **Act** to generate a new prompt.
* **From PDCA (Plan, Do, Check, Act):** We integrate the vital **Check** and **Act/Adjust** steps. This provides the explicit self-correction mechanism, forcing the agent to evaluate its own work.

### The R.O.S.E. Workflow

The agent operates in a three-phase cycle.

![ROSE Architecture Diagram](https://i.imgur.com/k28b05S.png)

#### Phase 1: Orient 🧭
This initial phase is about understanding the mission. The agent takes the user's raw inputs and transforms them into a structured, actionable mission plan.

* **Step 1: Ingest Inputs**
    The agent receives the two initial inputs from the user: `initial_prompt` and `goal`.

* **Step 2: Decompose Goal (Tool)**
    The agent uses an LLM to break down the user's abstract `goal` into a concrete list of measurable criteria. This transforms "make it better" into a clear checklist. This step requires the `GoalDecomposer` tool.

    **LLM Tool Prompt: `GoalDecomposer`**
    ```text
    You are an expert prompt engineer and a meticulous project manager. Your task is to decompose a user's high-level goal for a prompt into a list of specific, actionable, and verifiable criteria. These criteria will serve as a checklist to guide the prompt's revision and to evaluate the final result.

    Analyze the provided User's Initial Prompt and their Goal. Based on them, generate a JSON object containing a single key "criteria" which is a list of strings. Each string in the list should be a distinct, clear, and actionable instruction.

    **User's Initial Prompt:**
    "{initial_prompt}"

    **User's Goal:**
    "{goal}"

    **Instructions:**
    1.  Focus on what needs to be *added, removed, or changed* in the prompt to meet the goal.
    2.  The criteria should be concrete. For example, instead of "make it more creative," a better criterion would be "Add a constraint that the story must be told from the perspective of an inanimate object."
    3.  Ensure the criteria directly address the user's goal.

    **JSON Output format:**
    {{
      "criteria": [
        "Criterion 1...",
        "Criterion 2...",
        "Criterion 3..."
      ]
    }}
    ```

---
#### Phase 2: Plan & Do ✍️
This is the iterative refinement loop where the agent actively works on improving the prompt.

* **Step 3: Strategize Revision (Tool)**
    The agent reasons about *how* to implement the criteria, creating a step-by-step revision plan. This is the "Plan" part of the PDCA cycle and requires the `PromptStrategist` tool.

    **LLM Tool Prompt: `PromptStrategist`**
    ```text
    You are a master prompt strategist. Your job is to create a clear, step-by-step plan to revise a prompt based on a set of improvement criteria. You are not writing the new prompt yet, only creating the plan to do so.

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
    4.  Output a JSON object with a single key "plan" which is a list of strings.

    **JSON Output format:**
    {{
      "plan": [
        "Step 1: Change the persona to...",
        "Step 2: Add a section for defining the output format...",
        "Step 3: Incorporate the following constraint..."
      ]
    }}
    ```

* **Step 4: Generate New Prompt (Tool)**
    The agent executes the plan and writes a new version of the prompt. This is the "Do" part of the PDCA cycle and requires the `PromptGenerator` tool.

    **LLM Tool Prompt: `PromptGenerator`**
    ```text
    You are an expert AI prompt writer. Your task is to execute a revision plan to create a new, improved version of a prompt. Follow the plan precisely to generate the final text.

    **The Original Prompt:**
    "{current_prompt}"

    **The Revision Plan:**
    {revision_plan}

    **Instructions:**
    1.  Carefully implement each step in the revision plan.
    2.  The output should ONLY be the full text of the newly generated prompt inside a JSON object. Do not include any explanation or preamble.

    **JSON Output format:**
    {{
      "new_prompt": "The full text of the new and improved prompt goes here..."
    }}
    ```

---
#### Phase 3: Check ✅
This is the self-correction and evaluation phase where the agent inspects its own work.

* **Step 5: Evaluate Improvement (Tool)**
    The agent acts as an impartial judge, comparing the new prompt against the original using the decomposed criteria as a scorecard. This is the "Check" part of PDCA and requires the `PromptEvaluator` tool.

    **LLM Tool Prompt: `PromptEvaluator`**
    ```text
    You are a meticulous Quality Assurance analyst for AI prompts. Your task is to evaluate a revised prompt against an original prompt based on a specific set of criteria. You must be objective and critical.

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

    **JSON Output format:**
    {{
      "score": <integer from 1 to 10>,
      "rationale": "Your detailed reasoning here...",
      "is_improvement_sufficient": <true or false>
    }}
    ```
* **Step 6: Decision Point**
    This is the "Act/Adjust" part of PDCA. The agent reads the `is_improvement_sufficient` flag from the evaluation. If `True` or if a max iteration limit is reached, the process ends. If `False`, it loops back to Phase 2, using the evaluation feedback to inform the next attempt.

---
## Python Implementation

This script implements the R.O.S.E. architecture using Python, LangGraph, and Pydantic, with Google Gemini as the LLM.

```python
import os
import json
from typing import List, TypedDict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- Environment Setup ---
# Set your Google API key
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

# --- Pydantic Models for Structured Output ---
class DecomposedGoal(BaseModel):
    '''The structured output for the goal decomposition tool.'''
    criteria: List[str] = Field(description="A list of specific, actionable criteria to improve the prompt.")

class RevisionPlan(BaseModel):
    '''The structured output for the revision strategy tool.'''
    plan: List[str] = Field(description="A step-by-step plan to revise the prompt.")

class GeneratedPrompt(BaseModel):
    '''The structured output for the prompt generation tool.'''
    new_prompt: str = Field(description="The full text of the newly generated prompt.")

class Evaluation(BaseModel):
    '''The structured output for the self-correction/evaluation tool.'''
    score: int = Field(description="An overall score from 1 to 10 for the improvement.")
    rationale: str = Field(description="The reasoning behind the score, highlighting successes and failures.")
    is_improvement_sufficient: bool = Field(description="Whether the prompt is now good enough (score >= 8).")

# --- Agent State ---
class AgentState(TypedDict):
    initial_prompt: str
    goal: str
    decomposed_goal: Optional[DecomposedGoal]
    revision_plan: Optional[RevisionPlan]
    current_prompt: str
    evaluation: Optional[Evaluation]
    iteration_count: int
    max_iterations: int

# --- Agent Nodes (Tools) ---
def decompose_goal_node(state: AgentState) -> dict:
    '''Phase 1, Step 2: Decomposes the user's goal into actionable criteria.'''
    print("--- 🧭 ORIENT: DECOMPOSING GOAL ---")
    prompt_template = PromptTemplate(
        template='''You are an expert prompt engineer and a meticulous project manager. Your task is to decompose a user's high-level goal for a prompt into a list of specific, actionable, and verifiable criteria. These criteria will serve as a checklist to guide the prompt's revision and to evaluate the final result.
        **User's Initial Prompt:**\\n"{initial_prompt}"\\n\\n**User's Goal:**\\n"{goal}"\\n\\n**Instructions:**\\n1. Focus on what needs to be *added, removed, or changed* in the prompt to meet the goal.\\n2. The criteria should be concrete. For example, instead of "make it more creative," a better criterion would be "Add a constraint that the story must be told from the perspective of an inanimate object."\\n3. Ensure the criteria directly address the user's goal.\\n\\n{format_instructions}''',
        input_variables=["initial_prompt", "goal"],
        partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=DecomposedGoal).get_format_instructions()},
    )
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    parser = PydanticOutputParser(pydantic_object=DecomposedGoal)
    formatted_prompt = prompt_template.format(initial_prompt=state["initial_prompt"], goal=state["goal"])
    response = llm.invoke(formatted_prompt)
    parsed_response = parser.parse(response.content)
    print(f"✅ Decomposed Goal into Criteria: {json.dumps(parsed_response.dict(), indent=2)}")
    return {"decomposed_goal": parsed_response, "current_prompt": state["initial_prompt"]}

def strategize_revision_node(state: AgentState) -> dict:
    '''Phase 2, Step 3: Creates a plan to revise the prompt based on criteria.'''
    print("--- ✍️ PLAN: STRATEGIZING REVISION ---")
    feedback = state["evaluation"].rationale if state.get("evaluation") else "N/A"
    prompt_template = PromptTemplate(
        template='''You are a master prompt strategist. Your job is to create a clear, step-by-step plan to revise a prompt based on a set of improvement criteria. You are not writing the new prompt yet, only creating the plan to do so.
        **The Current Prompt:**\\n"{current_prompt}"\\n\\n**Improvement Criteria Checklist:**\\n{decomposed_goal}\\n\\n**Previous Evaluation Feedback (if any):**\\n{evaluation_feedback}\\n\\n**Instructions:**\\n1. Review the current prompt and the criteria.\\n2. If there is previous feedback, prioritize addressing the points of failure.\\n3. Create a concise, step-by-step plan of action for the revision.\\n\\n{format_instructions}''',
        input_variables=["current_prompt", "decomposed_goal", "evaluation_feedback"],
        partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=RevisionPlan).get_format_instructions()},
    )
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    parser = PydanticOutputParser(pydantic_object=RevisionPlan)
    formatted_prompt = prompt_template.format(current_prompt=state["current_prompt"], decomposed_goal=state["decomposed_goal"].dict(), evaluation_feedback=feedback)
    response = llm.invoke(formatted_prompt)
    parsed_response = parser.parse(response.content)
    print(f"✅ Created Revision Plan: {json.dumps(parsed_response.dict(), indent=2)}")
    return {"revision_plan": parsed_response}

def generate_prompt_node(state: AgentState) -> dict:
    '''Phase 2, Step 4: Executes the revision plan to generate a new prompt.'''
    print("--- ✍️ DO: GENERATING NEW PROMPT ---")
    prompt_template = PromptTemplate(
        template='''You are an expert AI prompt writer. Your task is to execute a revision plan to create a new, improved version of a prompt. Follow the plan precisely to generate the final text.
        **The Original Prompt:**\\n"{current_prompt}"\\n\\n**The Revision Plan:**\\n{revision_plan}\\n\\n**Instructions:**\\n1. Carefully implement each step in the revision plan.\\n2. The output should ONLY be the full text of the newly generated prompt inside a JSON object. Do not include any explanation or preamble.\\n\\n{format_instructions}''',
        input_variables=["current_prompt", "revision_plan"],
        partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=GeneratedPrompt).get_format_instructions()},
    )
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    parser = PydanticOutputParser(pydantic_object=GeneratedPrompt)
    formatted_prompt = prompt_template.format(current_prompt=state["current_prompt"], revision_plan=state["revision_plan"].dict())
    response = llm.invoke(formatted_prompt)
    parsed_response = parser.parse(response.content)
    print("✅ Generated New Prompt Version.")
    return {"current_prompt": parsed_response.new_prompt, "iteration_count": state["iteration_count"] + 1}

def evaluate_improvement_node(state: AgentState) -> dict:
    '''Phase 3, Step 5: Evaluates the new prompt against the criteria.'''
    print("--- ✅ CHECK: EVALUATING IMPROVEMENT ---")
    prompt_template = PromptTemplate(
        template='''You are a meticulous Quality Assurance analyst for AI prompts. Your task is to evaluate a revised prompt against an original prompt based on a specific set of criteria. You must be objective and critical.
        **The Original Prompt:**\\n"{initial_prompt}"\\n\\n**The Newly Revised Prompt:**\\n"{new_prompt}"\\n\\n**Improvement Criteria Checklist:**\\n{decomposed_goal}\\n\\n**Instructions:**\\n1. For each criterion in the checklist, assess if the new prompt successfully meets it.\\n2. Provide an overall score from 1 (no improvement) to 10 (perfectly met all criteria).\\n3. Write a brief, honest rationale for your score, explaining what was done well and what (if anything) is still missing or could be better.\\n4. Based on your analysis, determine if the improvement is sufficient to consider the task complete. The improvement is sufficient if the score is 8 or higher.\\n\\n{format_instructions}''',
        input_variables=["initial_prompt", "new_prompt", "decomposed_goal"],
        partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=Evaluation).get_format_instructions()},
    )
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)
    parser = PydanticOutputParser(pydantic_object=Evaluation)
    formatted_prompt = prompt_template.format(initial_prompt=state["initial_prompt"], new_prompt=state["current_prompt"], decomposed_goal=state["decomposed_goal"].dict())
    response = llm.invoke(formatted_prompt)
    parsed_response = parser.parse(response.content)
    print(f"✅ Evaluation Complete: {json.dumps(parsed_response.dict(), indent=2)}")
    return {"evaluation": parsed_response}

# --- Conditional Logic ---
def should_continue(state: AgentState) -> str:
    '''Phase 3, Step 6: Decides whether to end or continue refining.'''
    print("--- 🤔 DECISION POINT ---")
    if state["iteration_count"] >= state["max_iterations"]:
        print("🛑 Max iterations reached. Ending process.")
        return "end"
    evaluation = state["evaluation"]
    if evaluation.is_improvement_sufficient:
        print("🏆 Improvement is sufficient. Task complete.")
        return "end"
    else:
        print("🔄 Improvement not sufficient. Refining again.")
        return "continue_refining"

# --- Build the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("decompose_goal", decompose_goal_node)
workflow.add_node("strategize_revision", strategize_revision_node)
workflow.add_node("generate_prompt", generate_prompt_node)
workflow.add_node("evaluate_improvement", evaluate_improvement_node)
workflow.set_entry_point("decompose_goal")
workflow.add_edge("decompose_goal", "strategize_revision")
workflow.add_edge("strategize_revision", "generate_prompt")
workflow.add_edge("generate_prompt", "evaluate_improvement")
workflow.add_conditional_edges("evaluate_improvement", should_continue, {"continue_refining": "strategize_revision", "end": END})
app = workflow.compile()

# --- Run the Agent ---
if __name__ == "__main__":
    initial_prompt_input = "Write a story about a cat."
    goal_input = "Make the prompt more suitable for a magical realism author like Haruki Murakami. It should specify a mysterious tone and include an unusual, mundane object that becomes significant."
    inputs = {"initial_prompt": initial_prompt_input, "goal": goal_input, "iteration_count": 0, "max_iterations": 3}

    print("🚀 Starting Autonomous Prompt Improvement Agent...")
    print(f"Initial Prompt: '{inputs['initial_prompt']}'")
    print(f"Goal: '{inputs['goal']}'\\n")

    final_state = None
    for event in app.stream(inputs):
        if "__end__" not in event:
            node_name = list(event.keys())[0]
            node_output = event[node_name] 
            print("---")
            final_state = node_output

    print("\\n--- AGENT WORK COMPLETE ---")
    if final_state and final_state.get('evaluation'):
        final_prompt = final_state.get('current_prompt')
        final_evaluation = final_state.get('evaluation')
        print(f"\\n🌟 Final Improved Prompt:\\n{final_prompt}")
        print("\\n📊 Final Evaluation:")
        print(f"  - Score: {final_evaluation.score}/10")
        print(f"  - Rationale: {final_evaluation.rationale}")
    else:
        print("Agent did not complete the evaluation cycle.")
```
