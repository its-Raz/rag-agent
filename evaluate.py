import requests
import json
import os
import urllib3
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
API_URL = "https://ted-talk-rag-agent-raz.vercel.app/api/prompt"
LLMSTUDIO_API_KEY = "sk-wTBXiguVpJZFjvcvs3OWoA"
LLM_MODEL = "RPRTHPB-gpt-5-mini"

# Test questions for each task type
TEST_QUESTIONS = {
    "precise_fact_retrieval": {
        "question": "Find a TED talk that discusses overcoming fear or anxiety. Provide the title and speaker.",
        "task_type": "Precise Fact Retrieval",
        "evaluation_criteria": """
        Evaluate if the response:
        1. Returns EXACTLY ONE talk (not multiple)
        2. Provides both title and speaker(s)
        3. The talk is relevant to overcoming fear or anxiety
        4. Answer is grounded in the provided context
        5. Does not include unnecessary information

        Score from 1-10 and explain your reasoning.
        """
    },
    "multi_result_listing": {
        "question": "Which TED talks focus on education or learning? Return a list of exactly 3 talk titles.",
        "task_type": "Multi-Result Topic Listing",
        "evaluation_criteria": """
        Evaluate if the response:
        1. Returns EXACTLY 3 distinct talk titles
        2. Each title refers to a DIFFERENT talk (not multiple chunks from same talk)
        3. All talks are relevant to education or learning
        4. Answer is grounded in the provided context
        5. Format is clear and follows instructions

        Score from 1-10 and explain your reasoning.
        """
    },
    "key_idea_summary": {
        "question": "Find a TED talk where the speaker talks about technology improving people's lives. Provide the title and a short summary of the key idea.",
        "task_type": "Key Idea Summary Extraction",
        "evaluation_criteria": """
        Evaluate if the response:
        1. Identifies ONE relevant talk
        2. Provides the title
        3. Includes a concise summary of the main idea
        4. Summary is grounded in the transcript chunks provided in context
        5. Does not include information not supported by context

        Score from 1-10 and explain your reasoning.
        """
    },
    "recommendation": {
        "question": "I'm looking for a TED talk about climate change and what individuals can do in their daily lives. Which talk would you recommend?",
        "task_type": "Recommendation with Evidence-Based Justification",
        "evaluation_criteria": """
        Evaluate if the response:
        1. Recommends EXACTLY ONE talk
        2. Provides title and speaker(s)
        3. Includes justification grounded in retrieved transcript evidence
        4. Does not use external knowledge (popularity, views, etc.)
        5. Recommendation is relevant to the question

        Score from 1-10 and explain your reasoning.
        """
    }
}


def get_llm_client():
    """Initialize LLM client for evaluation"""
    return ChatOpenAI(
        api_key=LLMSTUDIO_API_KEY,
        base_url="https://api.llmod.ai/v1",
        model=LLM_MODEL,
        temperature=1
    )


def call_api(question: str) -> Dict[str, Any]:
    """Make API call to TED Talk RAG endpoint"""
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={"question": question},
            verify=False,  # -k flag equivalent
            timeout=120  # 2 minute timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out after 120 seconds"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def validate_response_format(response: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate that response matches expected format"""
    issues = []

    if "error" in response:
        issues.append(f"API returned error: {response['error']}")
        return False, issues

    # Check for required top-level keys
    required_keys = ["response", "context", "Augmented_prompt"]
    for key in required_keys:
        if key not in response:
            issues.append(f"Missing required key: {key}")

    # Validate context structure
    if "context" in response:
        if not isinstance(response["context"], list):
            issues.append("'context' should be a list")
        else:
            for i, ctx in enumerate(response["context"]):
                required_ctx_keys = ["talk_id", "title", "chunk", "score"]
                for key in required_ctx_keys:
                    if key not in ctx:
                        issues.append(f"Context item {i} missing key: {key}")

    # Validate Augmented_prompt structure
    if "Augmented_prompt" in response:
        if not isinstance(response["Augmented_prompt"], dict):
            issues.append("'Augmented_prompt' should be a dictionary")
        else:
            required_prompt_keys = ["System", "User"]
            for key in required_prompt_keys:
                if key not in response["Augmented_prompt"]:
                    issues.append(f"Augmented_prompt missing key: {key}")

    return len(issues) == 0, issues


def evaluate_response(question: str, response: Dict[str, Any],
                      task_type: str, criteria: str, llm) -> Dict[str, Any]:
    """Use LLM to evaluate the quality of the response"""

    # Build context summary
    context = response.get('context', [])
    unique_titles = set([ctx.get('title', '') for ctx in context])

    evaluation_prompt = f"""You are evaluating a RAG system response for a TED Talk assistant.

Task Type: {task_type}

Original Question: {question}

System Response: {response.get('response', 'N/A')}

Retrieved Context Summary:
- Number of context chunks: {len(context)}
- Unique talks in context: {len(unique_titles)}

Evaluation Criteria:
{criteria}

Provide your evaluation in the following format:
Score: [1-10]
Reasoning: [Detailed explanation of why you gave this score]
Strengths: [What the response did well]
Weaknesses: [What could be improved]
"""

    try:
        messages = [
            ("system", "You are an expert evaluator of RAG system responses. Be fair but critical."),
            ("human", evaluation_prompt)
        ]
        eval_response = llm.invoke(messages)
        return {
            "evaluation": eval_response.content,
            "success": True,
            "error": None
        }
    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        print(f"    Error details: {error_msg}")
        return {
            "evaluation": "Evaluation could not be completed due to an error.",
            "success": False,
            "error": error_msg
        }


def save_results(results: Dict[str, Any], filename: str = "evaluation_results.json"):
    """Save evaluation results to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {filename}")
        return True
    except Exception as e:
        print(f"\n✗ Failed to save results: {str(e)}")
        return False


def save_raw_response(response: Dict[str, Any], task_name: str):
    """Save individual raw API response"""
    filename = f"response_{task_name}.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Raw response saved to {filename}")
    except Exception as e:
        print(f"  ✗ Failed to save raw response: {str(e)}")


def print_summary(results: Dict[str, Any]):
    """Print a summary of evaluation results"""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    for task_name, result in results.items():
        print(f"\n{task_name.upper().replace('_', ' ')}")
        print("-" * 80)
        print(f"Question: {result['question']}")
        print(f"\nFormat Valid: {'✓ YES' if result['format_valid'] else '✗ NO'}")

        if not result['format_valid']:
            print(f"Format Issues: {', '.join(result['format_issues'])}")

        if result['format_valid']:
            response_text = result.get('response', 'N/A')
            if len(response_text) > 200:
                print(f"\nResponse Preview: {response_text}...")
            else:
                print(f"\nResponse: {response_text}")

            if result['llm_evaluation']['success']:
                print(f"\nLLM Evaluation:")
                print(result['llm_evaluation']['evaluation'])
            else:
                print(f"\n✗ LLM Evaluation Failed: {result['llm_evaluation'].get('error', 'Unknown error')}")


def main():
    """Main evaluation pipeline"""
    print("Starting TED Talk RAG Evaluation...")
    print("=" * 80)

    # Check for API key
    if not LLMSTUDIO_API_KEY:
        print("\n✗ ERROR: LLMSTUDIO_API_KEY environment variable not set!")
        print("Please set it with: export LLMSTUDIO_API_KEY='your-key-here'")
        return

    # Initialize LLM for evaluation
    print("\n→ Initializing LLM client...")
    try:
        llm = get_llm_client()
        print("  ✓ LLM client initialized")
    except Exception as e:
        print(f"  ✗ Failed to initialize LLM: {str(e)}")
        return

    # Store all results
    all_results = {}

    # Process each test question
    for task_name, task_config in TEST_QUESTIONS.items():
        print(f"\n{'=' * 80}")
        print(f"Testing: {task_config['task_type']}")
        print(f"{'=' * 80}")
        print(f"Question: {task_config['question']}")

        # Call API
        print("\n→ Calling API...")
        api_response = call_api(task_config['question'])

        # Save raw response immediately
        save_raw_response(api_response, task_name)

        # Validate format
        print("→ Validating response format...")
        format_valid, format_issues = validate_response_format(api_response)

        if format_valid:
            print("  ✓ Format is valid")
        else:
            print("  ✗ Format issues found:")
            for issue in format_issues:
                print(f"    - {issue}")

        # Evaluate with LLM (only if format is valid)
        llm_evaluation = {"success": False, "evaluation": "Not evaluated", "error": "Format validation failed"}
        if format_valid:
            print("→ Evaluating response quality with LLM...")
            llm_evaluation = evaluate_response(
                task_config['question'],
                api_response,
                task_config['task_type'],
                task_config['evaluation_criteria'],
                llm
            )

            if llm_evaluation['success']:
                print("  ✓ Evaluation complete")
            else:
                print("  ✗ Evaluation failed")

        # Store results
        all_results[task_name] = {
            "task_type": task_config['task_type'],
            "question": task_config['question'],
            "response": api_response.get('response', 'N/A'),
            "format_valid": format_valid,
            "format_issues": format_issues,
            "llm_evaluation": llm_evaluation,
            "full_api_response": api_response
        }

    # Save combined results
    save_results(all_results)

    # Print summary
    print_summary(all_results)

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    print("\nFiles created:")
    print("  - evaluation_results.json (complete results)")
    print("  - response_precise_fact_retrieval.json")
    print("  - response_multi_result_listing.json")
    print("  - response_key_idea_summary.json")
    print("  - response_recommendation.json")


if __name__ == "__main__":
    main()