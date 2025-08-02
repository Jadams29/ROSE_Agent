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


