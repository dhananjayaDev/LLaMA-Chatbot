from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from huggingface_hub import InferenceClient
import pymongo
import os
import re
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_url_path="/static")
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET") or "default_dev_secret_key"

# Hugging Face Token (load from env)
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    token=HF_TOKEN
)

# MongoDB config
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME") or "test"

def connect_to_mongo():
    client = pymongo.MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]

# Enhanced system prompt
system_prompt = {
    "role": "system",
    "content": (
        "You are Mangalam's intelligent wedding assistant. Mangalam is a Sri Lankan wedding service platform. "
        "You have direct access to Mangalam’s vendor database and can respond with real-time wedding service details. "
        "If users ask about DJs, decorators, photography, cakes, catering, venues, or prices — always provide accurate info from the database. "
        "NEVER say you don’t have access. Never mention contacting the website. "
        "If vendors are in the MongoDB collection, give them clearly and helpfully. "
        "Avoid general language model phrases like 'I am just an AI'. Stay helpful, confident, and business-specific."
    )
}

def generate_response(messages):
    prompt = f"[INST] {system_prompt['content']} [/INST]"
    for m in messages:
        if m["role"] == "user":
            prompt += f"[INST] {m['content']} [/INST]"
        elif m["role"] == "assistant":
            prompt += f"{m['content']}</s>"

    response = client.text_generation(
        prompt,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    return response.strip()

def is_unrelated(q):
    greetings = ["hi", "hello", "hey"]
    if any(greet in q.lower() for greet in greetings):
        return "greeting"
    keywords = ["wedding", "vendor", "price", "venue", "service", "photography", "mangalam", "decorator", "dj", "cake"]
    return "unrelated" if not any(k in q.lower() for k in keywords) else "related"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if "chat_history" not in session:
        session["chat_history"] = []

    check = is_unrelated(question)
    if check == "greeting":
        return jsonify({"answer": "🫠 Hello! I’m your Mangalam wedding assistant. How can I help with your big day? 💍"})
    elif check == "unrelated":
        return jsonify({"answer": "🫠 I specialize in wedding-related questions on Mangalam! Try asking about vendors, prices, or services 🌺"})

    # Price query
    price_match = re.search(r"\d+", question)
    if any(x in question.lower() for x in ["under", "below", "less than"]) and price_match:
        price = int(price_match.group(0))
        db = connect_to_mongo()
        vendors = db["vendors"].find({"price": {"$lte": price}}).limit(5)
        results = list(vendors)

        if not results:
            return jsonify({"answer": f"Sorry, no services found under {price} LKR."})

        response = f"<strong>Here are services under {price} LKR:</strong><br>"
        for v in results:
            response += f"<br><strong>{v['service_name']}</strong> ({v['service_type']})<br>"
            response += f"{v['description']}<br>💰 {v['price']} LKR<br>📦 Includes: {v['what_we_provide']}<hr>"
        return jsonify({"answer": response})

    # Specific type search (e.g., DJs)
    if "dj" in question.lower():
        db = connect_to_mongo()
        djs = db["vendors"].find({"service_type": {"$regex": "dj", "$options": "i"}}).limit(5)
        results = list(djs)
        if not results:
            return jsonify({"answer": "No DJ services found currently."})

        response = "<strong>DJ Services:</strong><br>"
        for v in results:
            response += f"<br><strong>{v['service_name']}</strong><br>{v['description']}<br>💰 {v['price']} LKR<br>📦 Includes: {v['what_we_provide']}<hr>"
        return jsonify({"answer": response})

    # Mistral AI fallback with memory
    messages = []
    for turn in session["chat_history"][-3:]:
        messages.append({"role": "user", "content": turn["q"]})
        messages.append({"role": "assistant", "content": turn["a"]})
    messages.append({"role": "user", "content": question})

    try:
        answer = generate_response(messages)
    except Exception as e:
        answer = f"Error: {str(e)}"

    session["chat_history"].append({"q": question, "a": answer})
    session.modified = True

    return jsonify({"answer": answer})

@app.route("/reset", methods=["POST"])
def reset():
    session["chat_history"] = []
    return jsonify({"message": "Chat history cleared!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)