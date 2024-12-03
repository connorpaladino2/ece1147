import os
import json
import ollama

image_folder = 'data/data-20241202T145651Z-001/data/images/abortion'

images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')]
results = []

for image in images:
    try:
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': '''In a maximum of 20 words, can you determine if the image supports abortion or not. 
                              Then provide a number at the end of your response, 1 representing pro abortion, and 0 being anti abortion''',
                'images': [image]
            }]
        )
        
        if 'message' in response and 'content' in response['message']:
            content = response['message']['content']
        else:
            content = 'Model did not return a valid response'
        
        print(f"Image: {image}, Response: {content}")
        results.append({'image': image, 'response': content})
    except Exception as e:
        print(f"Failed to process {image}: {e}")
        results.append({'image': image, 'error': str(e)})



#save results to a JSON file
output_file = 'results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")
