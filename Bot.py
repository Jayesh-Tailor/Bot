import os
import time
import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load API Key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
# print(f"DEBUG: Loaded API Key -> {repr(API_KEY)}")
MODEL_NAME = 'gemini-2.5-flash'

if not API_KEY:
    print("❌ Error: GEMINI_API_KEY environment variable not set.")
    exit()

client = genai.Client(api_key=API_KEY)

# 2. Initialize Chat Session
chat = client.chats.create(
    model=MODEL_NAME,
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful AI assistant. Answer questions concisely and accurately.",
        temperature=0.7,
    )
)

def send_message_with_retry(chat_session, user_message, max_retries=3):
    """
    Sends a message with automatic retries for 503 and 429 errors.
    """
    for attempt in range(max_retries):
        try:
            # Check if user is asking about time/date to attach context dynamically
            time_keywords = ["time", "date", "day", "today", "now"]
            
            if any(keyword in user_message.lower() for keyword in time_keywords):
                current_time = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
                
                # Pass time context using config without altering user's raw message history
                config = types.GenerateContentConfig(
                    system_instruction=f"You are a helpful AI assistant. The current system time is {current_time}."
                )
                response = chat_session.send_message(user_message, config=config)
            else:
                response = chat_session.send_message(user_message)

            return response.text
            
        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "429" in error_str:
                wait_time = (2 ** attempt) + 5
                print(f"⚠️ Server busy (503/429). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return f"❌ An unrecoverable error occurred: {e}"
    
    return "❌ Server is currently too busy. Please try again in a minute."     

# --- Main Loop ---
print("🤖 AI Chatbot (Connected and Ready)")
print("Type 'bye' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue

    print('\n')
    
    if user_input.lower() in ["bye", "exit", "quit"]:
        print("🤖 AI Chatbot: Goodbye!")
        break

    response_text = send_message_with_retry(chat, user_input)
    print("🤖 AI Chatbot:", response_text, "\n")