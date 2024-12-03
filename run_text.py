import ollama
import json


def run_text(input):

    prompt = f"""
                Please clear memory and disregard any previous messages in our session.
                
                Please assess this tweet "{input}" and return outputs based on the following 7 questions. All outputs must be on a scale from 0-1, with
                precision to the 100ths places. The output will be structured as...
                Question 1: (range from 0-1 for question 1)
                Question 2: (range from 0-1 for question 2)
                Question 3: (range from 0-1 for question 3)
                Question 4: (range from 0-1 for question 4)
                Question 5: (range from 0-1 for question 5)
                Question 6: (range from 0-1 for question 6)
                Question 7: (range from 0-1 for question 7)
                DO NOT DEVIATE FROM THIS OUTPUT SCHEMA FOR ANY REASON. THE OUTPUT MUST FOLLOW THE SCHEMA ABOVE. The output must have Question #: numerical answer AND THAT IS IT!

                Here are the questions...
                Question 1: Is this tweet supporting gun rights? Provide a confidence score between 0 and 1.
                Question 2: Does the content focus on the Second Amendment or constitutional rights to beararms? Provide a confidence score between 0 and 1.
                Question 3: Does the material mention legislation or laws regarding gun ownership or gun control? Provide a confidence score between 0 and 1.
                Question 4: Are terms like "pro-choice," "pro-life," or "abortion rights" used in the text? Provide a confidence score between 0 and 1.
                Question 5: Does the content discuss women's reproductive rights or abortion-related issues? Provide a confidence score between 0 and 1.
                Question 6: Is there discussion about concealed carry permits, open carry laws, or the right to carry firearms? Provide a confidence score between 0 and 1.
                Question 7: Is this tweet supporting abortion? Provide a confidence score between 0 and 1.

                DO NOT SUPPLY ANY EXPLINATIONS AT ALL. ONLY THE ANSWERS. BE EXTREMELY BRIEF.
            """
    # Call the chat API
    response = ollama.chat(
        model='llama3.2-vision:11b',
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )

    # Parse the response JSON
    # Assuming `response` has a `.text` or `.json()` method that contains the JSON string:
    response_data = json.loads(response.model_dump_json())

    # Extract the content from the response
    message_content = response_data.get("message", {}).get("content", "No content available")

    return message_content