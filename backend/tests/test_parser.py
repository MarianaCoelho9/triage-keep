import re

def parse_agent_response(text: str) -> tuple[str, str]:
    """
    Parses the LLM response to extract thought and question using regex.
    Returns (thought, question).
    """
    thought_match = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL)
    question_match = re.search(r"<question>(.*?)</question>", text, re.DOTALL)
    
    thought = thought_match.group(1).strip() if thought_match else ""
    question = question_match.group(1).strip() if question_match else text.strip()
    
    return thought, question

def test_parser():
    test_cases = [
        {
            "name": "Standard output",
            "text": "<thought>I need to know the duration of the pain.</thought><question>How long have you had this pain?</question>",
            "expected": ("I need to know the duration of the pain.", "How long have you had this pain?")
        },
        {
            "name": "Missing thought",
            "text": "<question>Where is the pain located?</question>",
            "expected": ("", "Where is the pain located?")
        },
        {
            "name": "Missing tags (fallback)",
            "text": "What is your main concern today?",
            "expected": ("", "What is your main concern today?")
        },
        {
            "name": "Multiline and extra spaces",
            "text": "  <thought>\n  Checking red flags.\n  </thought>\n  <question>\n  Do you have difficulty breathing?\n  </question>  ",
            "expected": ("Checking red flags.", "Do you have difficulty breathing?")
        }
    ]

    for tc in test_cases:
        thought, question = parse_agent_response(tc["text"])
        print(f"Test case: {tc['name']}")
        print(f"  Thought: '{thought}'")
        print(f"  Question: '{question}'")
        assert (thought, question) == tc["expected"]
        print("  Result: PASS")

if __name__ == "__main__":
    test_parser()
