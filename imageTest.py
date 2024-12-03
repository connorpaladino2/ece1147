import ollama
import json

response = ollama.chat(
    model='llama3.2-vision:11b',
    messages=[{
        'role': 'user',
        'content': 'Can you tell me if this image supports abortion or not.',
        'images': ['data/data-20241202T145651Z-001/data/images/abortion/1018237459270914049.jpg']
    }]
)

print(response)

# Parse the response JSON
# Assuming `response` has a `.text` or `.json()` method that contains the JSON string:
response_data = json.loads(response.model_dump_json())

# Extract the content from the response
message_content = response_data.get("message", {}).get("content", "No content available")

print(message_content)