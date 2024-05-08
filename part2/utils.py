import re

# Define a function to extract an answer from a text response
def extract_answer(text_response: str, pattern: str = r"Answer:\s*(.*)") -> str:
    # Use regular expression to search for the pattern in the text response
    match = re.search(pattern, text_response)
    # If a match is found
    if match:
        # Extract the answer text and strip any leading or trailing whitespace
        answer_text = match.group(1).strip()
        # Return the answer text
        return answer_text
    else:
        # If no match is found, return a default message
        return "Answer not found."
