from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt):
    # Tokenize input
    inputs = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt")

    # Generate a response with stricter parameters to prevent repetition
    response = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=50,  # Restrict the maximum length of the response
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,  # Avoid repeating n-grams
        num_return_sequences=1,  # Only return 1 response
        temperature=0.6,  # Lower temperature for more focused responses
        top_p=0.9,  # Nucleus sampling (for more natural responses)
        do_sample=True,  # Sampling to avoid deterministic responses
        repetition_penalty=1.5  # Penalty for repeated words/phrases
    )

    return tokenizer.decode(response[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True)

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# API route to handle the chatbot conversation
@app.route('/chat', methods=['POST'])
def chat():
    # Get the data from the frontend
    data = request.get_json()
    user_message = data.get('message')
    print(f"User message received: {user_message}")  # Debugging purpose

    # Generate a response using the GPT model
    bot_response = generate_response(user_message)
    print(f"Bot response: {bot_response}")  # Debugging purpose

    # Return the response as JSON
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
