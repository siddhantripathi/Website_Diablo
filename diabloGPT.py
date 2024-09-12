from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

def chat_with_bot():
    print("Start chatting with the bot (type 'quit' to stop)!")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Exit the loop if the user types 'quit'
        if user_input.lower() == 'quit':
            break
        
        # Generate and print the response
        bot_response = generate_response(user_input)
        print(f"Bot: {bot_response}")

if __name__ == "__main__":
    chat_with_bot()
