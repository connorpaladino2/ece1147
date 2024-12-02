import ollama

response = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'Can you tell me if this image supports abortion or not.',
        'images': ['data/data-20241202T145651Z-001/data/images/abortion/1018237459270914049.jpg']
    }]
)

print(response)