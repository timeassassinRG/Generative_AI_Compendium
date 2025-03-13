from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the larger model for better responses (use "facebook/blenderbot-3B" if possible)
model_name = "facebook/blenderbot-3B"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Maintain conversation history
conversation_history = []

def chat_with_bot():
    global conversation_history  # Keep track of context

    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # Append user input to conversation history
        conversation_history.append(f"User: {user_input}")

        # Create input text with conversation history (keep only the last few messages)
        context = " ".join(conversation_history[-3:])  # Keep the last 3 exchanges

        # Tokenize input and generate response with improved decoding strategy
        inputs = tokenizer(context, return_tensors="pt", truncation=True)
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        # Decode and print response
        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print("Chatbot:", bot_response)

        # Append bot response to history
        conversation_history.append(f"Bot: {bot_response}")

# Start chat
chat_with_bot()
