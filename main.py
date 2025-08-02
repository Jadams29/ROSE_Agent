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
    initial_prompt_input = "Write a financial analysis for a company."
    goal_input = ("The analysis should include the company's name, current price, and the company's financial statements such as balance sheet, cash flow, and income statements.")

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
    rose_agent = ROSEAgent(llm_model_name="gemini-2.5-pro")
    app = rose_agent.get_graph()

    print("--- Starting Agent Execution ---")
    
    # Run the agent and get the complete final state
    final_state = app.invoke(inputs)

    print("\n\n--- ROSE AGENT WORK COMPLETE ---")
    
    # Display debugging information
    print(f"\n--- Debugging Final State ---")
    if final_state:
        print(f"Final state keys: {list(final_state.keys())}")
        print(f"Iteration count: {final_state.get('iteration_count', 'N/A')}")
        if final_state.get('current_prompt'):
            print(f"Current prompt length: {len(final_state.get('current_prompt', ''))}")
        if final_state.get('evaluation'):
            print(f"Evaluation present: {type(final_state.get('evaluation'))}")
    else:
        print("final_state is None")

    # Extract and display final results
    if final_state and final_state.get('current_prompt') and final_state.get('evaluation'):
        final_prompt = final_state.get('current_prompt')
        final_evaluation = final_state.get('evaluation')

        print("\n\n=====================================")
        print("          FINAL RESULTS")
        print("=====================================")
        print("\n### Initial Prompt:")
        print(f'"{inputs["initial_prompt"]}"')
        print("\n### User Goal:")
        print(f'"{inputs["goal"]}"')
        print(f"\n### Final Improved Prompt:")
        print(f'"{final_prompt}"')
        print("\n### Final Evaluation:")
        print(f"  - Score: {final_evaluation.score}/10")
        print(f"  - Rationale: {final_evaluation.rationale}")
        print(f"  - Improvement Sufficient: {final_evaluation.is_improvement_sufficient}")
        print("\n=====================================")
    else:
        print("\n❌ ERROR: Agent did not complete successfully.")
        if not final_state:
            print("  - No final state returned")
        elif not final_state.get('current_prompt'):
            print("  - No final prompt generated")
        elif not final_state.get('evaluation'):
            print("  - No evaluation completed")
        else:
            print("  - Unknown issue occurred")