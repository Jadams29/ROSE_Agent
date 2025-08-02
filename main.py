import os
from dotenv import load_dotenv

from agents import ROSEAgent

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Main Execution ---
if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    # Example User Inputs
    initial_prompt_input = "Write a story about a cat."
    goal_input = "Make the prompt more suitable for a magical realism author like Haruki Murakami. It should specify a mysterious tone and include an unusual, mundane object that becomes significant."

    # The inputs for the graph
    inputs = {
        "initial_prompt": initial_prompt_input,
        "goal": goal_input,
        "iteration_count": 0,
        "max_iterations": 3
    }

    print("🚀 Starting ROSE Autonomous Prompt Improvement Agent...")
    print(f"Initial Prompt: '{inputs['initial_prompt']}'")
    print(f"Goal: '{inputs['goal']}'\n")

    # Instantiate the agent and get the compiled graph
    rose_agent = ROSEAgent(llm_model_name="gemini-2.5-flash")
    app = rose_agent.get_graph()

    # The stream() method lets us see the output of each step
    final_state = None
    for event in app.stream(inputs):
        if "__end__" not in event:
            # The event key is the name of the node that just ran
            node_name = list(event.keys())[0]
            # The value is the dictionary returned by that node
            node_output = event[node_name]
            print("---")
            final_state = node_output

    print("\n--- ROSE AGENT WORK COMPLETE ---")
    if final_state and final_state.get('evaluation'):
        final_prompt = final_state.get('current_prompt')
        final_evaluation = final_state.get('evaluation')

        print(f"\n🌟 Final Improved Prompt:\n{final_prompt}")
        print("\n📊 Final Evaluation:")
        print(f"  - Score: {final_evaluation.score}/10")
        print(f"  - Rationale: {final_evaluation.rationale}")
    else:
        print("Agent did not complete the evaluation cycle.")