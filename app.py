# For Web App 

import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)

chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful AI assistant.",
        temperature=0.7,
    ),
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    try:
        response = chat.send_message(message)
        return jsonify({"response": response.text})
    except Exception as error:
        error_text = str(error)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            return jsonify({
                "error": "Gemini API quota is temporarily exhausted. Please try again later or check your API plan and billing."
            }), 429
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run()