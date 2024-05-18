import PIL.Image
import google.generativeai as genai
from IPython.display import Markdown
import textwrap

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def process(image, prompt=None):
    img = image
    api_key = "AIzaSyAd05f5BRfPz5IKFWldyNSdoP2HrVQB74Q" 
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro-vision')
    questions=[]
    if prompt is None:
        with open('fields.txt') as f:
            questions = f.readlines()
        questions = [x.strip() for x in questions]
        #print(questions)
        prompt = ""
        for i in questions:
            prompt = prompt + i + "\n"
        prompt = prompt + "Extract relevant information from the image and return key-value pairs in this format fied==value"
    else :
        prompt = prompt
    response = model.generate_content([prompt,img])
    print(response.text)
    return response.text
    

