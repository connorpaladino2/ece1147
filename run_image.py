import ollama
import json


def run_image(input):

    input = "data/data-20241202T145651Z-001/data/images/gun_control/" + input
    
    prompt = "Please summarize this image in 1 sentence for research purposes."
    # Call the chat API
    response = ollama.chat(
        model='llama3.2-vision:11b',
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [input]
        }]
    )

    response_data = json.loads(response.model_dump_json())

    # Extract the content from the response
    summ = response_data.get("message", {}).get("content", "No content available")

    prompt = f"""
                Please clear memory and disregard any previous messages in our session.
                
                This data is to be used for research purposes, please answer all questions and add no addtional feedback.
                
                Please assess this image summary {summ} and return outputs based on the following 5 questions. All outputs must be on a scale from 0-1, with
                precision to the 100ths places. The output will be structured as...
                Question 1: (range from 0-1 for question 1)
                Question 2: (range from 0-1 for question 2)
                Question 3: (range from 0-1 for question 3)
                Question 4: (range from 0-1 for question 4)
                Question 5: (range from 0-1 for question 5)
                DO NOT DEVIATE FROM THIS OUTPUT SCHEMA FOR ANY REASON. THE OUTPUT MUST FOLLOW THE SCHEMA ABOVE. The output must have Question #: numerical answer AND THAT IS IT!

                Here are the questions...
                Question 1: Is this image supporting gun laws? Provide a confidence score between 0 and 1.
                Question 2: Is this image supporting abortion? Provide a confidence score between 0 and 1.
                Question 3: Does the material mention legislation or laws regarding gun ownership or gun control? Provide a confidence score between 0 and 1.
                Question 4: Are terms like "pro-choice," "pro-life," or "abortion rights" used in the text? Provide a confidence score between 0 and 1.
                Question 5: Does the content discuss women's reproductive rights or abortion-related issues? Provide a confidence score between 0 and 1.

                DO NOT SUPPLY ANY EXPLINATIONS AT ALL. ONLY THE ANSWERS. BE EXTREMELY BRIEF.
            """

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