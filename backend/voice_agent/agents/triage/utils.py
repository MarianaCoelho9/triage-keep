"""Utility functions for parsing agent responses."""
import re


def parse_agent_response(text: str) -> tuple[str, str]:
    """
    Parses the LLM response to extract thought and question using regex.
    Returns (thought, question).
    """
    thought_match = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL)
    question_match = re.search(r"<question>(.*?)</question>", text, re.DOTALL)
    
    thought = thought_match.group(1).strip() if thought_match else ""
    if question_match:
        question = question_match.group(1).strip()
    else:
        # Some backends strip the closing tag via stop sequences.
        # Handle partial tag output like "<question> ..." gracefully.
        cleaned = re.sub(r"</question>\s*$", "", text.strip())
        cleaned = re.sub(r"^\s*<question>\s*", "", cleaned)
        question = cleaned.strip()
    
    return thought, question


def extract_json_from_text(text: str) -> str:
    """
    Extracts JSON string from text, processing markdown blocks and finding the first/last brace.
    """
    cleaned = text.strip()
    cleaned = cleaned.replace("<END_JSON>", "").strip()
    
    # Remove markdown code blocks
    if "```" in cleaned:
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
             return match.group(1)
        
    # Fallback: Find first { and last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end+1]
        
    return cleaned
