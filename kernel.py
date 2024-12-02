import ollama
import json

# Call the chat API
response = ollama.chat(
    model='llama3.2-vision:11b',
    messages=[{
        'role': 'user',
        'content': 'What is your favorite color.'
    }]
)

# Parse the response JSON
# Assuming `response` has a `.text` or `.json()` method that contains the JSON string:
response_data = json.loads(response.model_dump_json())

# Extract the content from the response
message_content = response_data.get("message", {}).get("content", "No content available")

print(message_content)

#test