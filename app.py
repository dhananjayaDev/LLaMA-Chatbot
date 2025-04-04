# from flask import Flask, render_template, request, jsonify, session
# from flask_cors import CORS
# from huggingface_hub import InferenceClient
# import pymongo
# import os
# import re
# from dotenv import load_dotenv
# import logging
#
# # Load environment variables
# load_dotenv()
#
# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# app = Flask(__name__, static_url_path="/static")
# CORS(app)  # Note: In production, restrict origins, e.g., CORS(app, origins=["yourdomain.com"])
#
# # Flask configuration
# app.secret_key = os.getenv("FLASK_SECRET")
# if not app.secret_key:
#     logger.error("FLASK_SECRET not set in environment variables. Using default (insecure for production).")
#     app.secret_key = "default_dev_secret_key"
#
# # Hugging Face Token
# HF_TOKEN = os.getenv("HF_TOKEN")
# if not HF_TOKEN:
#     logger.error("HF_TOKEN not set in environment variables. The application will not work without it.")
#     raise ValueError("HF_TOKEN is required")
#
# client = InferenceClient(
#     model="mistralai/Mistral-7B-Instruct-v0.1",
#     token=HF_TOKEN
# )
#
# # MongoDB configuration
# MONGO_URI = os.getenv("MONGO_URI")
# if not MONGO_URI:
#     logger.error("MONGO_URI not set in environment variables. The application will not work without it.")
#     raise ValueError("MONGO_URI is required")
#
# MONGO_DB_NAME = os.getenv("MONGO_DB_NAME") or "test"
#
# # Singleton MongoDB connection
# _mongo_client = None
# _mongo_db = None
#
# def connect_to_mongo():
#     global _mongo_client, _mongo_db
#     if _mongo_client is None:
#         try:
#             _mongo_client = pymongo.MongoClient(MONGO_URI)
#             _mongo_db = _mongo_client[MONGO_DB_NAME]
#             logger.info("Connected to MongoDB successfully.")
#         except Exception as e:
#             logger.error(f"Failed to connect to MongoDB: {str(e)}")
#             raise
#     return _mongo_db
#
# # Enhanced system prompt
# system_prompt = {
#     "role": "system",
#     "content": (
#         "You are Mangalam's intelligent wedding assistant. Mangalam is a Sri Lankan wedding service platform. "
#         "You have direct access to Mangalam’s vendor database and can respond with real-time wedding service details. "
#         "If users ask about DJs, decorators, photography, cakes, catering, venues, or prices — always provide accurate info from the database. "
#         "NEVER say you don’t have access. Never mention contacting the website. "
#         "If vendors are in the MongoDB collection, give them clearly and helpfully. "
#         "Avoid general language model phrases like 'I am just an AI'. Stay helpful, confident, and business-specific."
#     )
# }
#
# def generate_response(messages):
#     try:
#         prompt = f"[INST] {system_prompt['content']} [/INST]"
#         for m in messages:
#             if m["role"] == "user":
#                 prompt += f"[INST] {m['content']} [/INST]"
#             elif m["role"] == "assistant":
#                 prompt += f"{m['content']}</s>"
#
#         response = client.text_generation(
#             prompt,
#             max_new_tokens=300,
#             temperature=0.7,
#             top_p=0.9,
#             repetition_penalty=1.1,
#         )
#         return response.strip()
#     except Exception as e:
#         logger.error(f"Error generating response from Hugging Face: {str(e)}")
#         return "Sorry, I encountered an issue while processing your request. Please try again later."
#
# def is_unrelated(q):
#     greetings = ["hi", "hello", "hey"]
#     if any(greet in q.lower() for greet in greetings):
#         return "greeting"
#     keywords = ["wedding", "vendor", "price", "venue", "service", "photography", "mangalam", "decorator", "dj", "cake"]
#     return "unrelated" if not any(k in q.lower() for k in keywords) else "related"
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
# @app.route("/ask", methods=["POST"])
# def ask():
#     # Validate request
#     if not request.is_json:
#         return jsonify({"answer": "Invalid request format. Please send JSON data."}), 400
#
#     data = request.get_json()
#     question = data.get("question", "").strip()
#
#     if not question:
#         return jsonify({"answer": "Please provide a question."}), 400
#
#     # Initialize chat history if not present
#     if "chat_history" not in session:
#         session["chat_history"] = []
#
#     # Limit chat history to last 10 interactions to prevent excessive memory usage
#     if len(session["chat_history"]) > 10:
#         session["chat_history"] = session["chat_history"][-10:]
#         session.modified = True
#
#     # Check for unrelated questions or greetings
#     check = is_unrelated(question)
#     if check == "greeting":
#         return jsonify({"answer": "🫠 Hello! I’m your Mangalam wedding assistant. How can I help with your big day? 💍"})
#     elif check == "unrelated":
#         return jsonify({"answer": "🫠 I specialize in wedding-related questions on Mangalam! Try asking about vendors, prices, or services 🌺"})
#
#     # Price query
#     price_match = re.search(r"\d+", question)
#     if any(x in question.lower() for x in ["under", "below", "less than"]) and price_match:
#         price = int(price_match.group(0))
#         try:
#             db = connect_to_mongo()
#             vendors = db["vendors"].find({"price": {"$lte": price}}).limit(5)
#             results = list(vendors)
#
#             if not results:
#                 return jsonify({"answer": f"Sorry, no services found under {price} LKR."})
#
#             response = f"<strong>Here are services under {price} LKR:</strong><br>"
#             for v in results:
#                 response += f"<br><strong>{v['service_name']}</strong> ({v['service_type']})<br>"
#                 response += f"{v['description']}<br>💰 {v['price']} LKR<br>📦 Includes: {v['what_we_provide']}<hr>"
#             return jsonify({"answer": response})
#         except Exception as e:
#             logger.error(f"Error querying MongoDB for price: {str(e)}")
#             return jsonify({"answer": "Sorry, I couldn’t fetch the vendor details right now. Please try again later."})
#
#     # Specific type search (e.g., DJs)
#     if "dj" in question.lower():
#         try:
#             db = connect_to_mongo()
#             djs = db["vendors"].find({"service_type": {"$regex": "dj", "$options": "i"}}).limit(5)
#             results = list(djs)
#             if not results:
#                 return jsonify({"answer": "No DJ services found currently."})
#
#             response = "<strong>DJ Services:</strong><br>"
#             for v in results:
#                 response += f"<br><strong>{v['service_name']}</strong><br>{v['description']}<br>💰 {v['price']} LKR<br>📦 Includes: {v['what_we_provide']}<hr>"
#             return jsonify({"answer": response})
#         except Exception as e:
#             logger.error(f"Error querying MongoDB for DJs: {str(e)}")
#             return jsonify({"answer": "Sorry, I couldn’t fetch the DJ services right now. Please try again later."})
#
#     # Mistral AI fallback with memory
#     messages = []
#     for turn in session["chat_history"][-3:]:
#         messages.append({"role": "user", "content": turn["q"]})
#         messages.append({"role": "assistant", "content": turn["a"]})
#     messages.append({"role": "user", "content": question})
#
#     answer = generate_response(messages)
#
#     session["chat_history"].append({"q": question, "a": answer})
#     session.modified = True
#
#     return jsonify({"answer": answer})
#
# @app.route("/reset", methods=["POST"])
# def reset():
#     session["chat_history"] = []
#     return jsonify({"message": "Chat history cleared!"})
#
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5003, debug=True)


from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from huggingface_hub import InferenceClient
import pymongo
import os
import re
import joblib
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path="/static")
CORS(app)

app.secret_key = os.getenv("FLASK_SECRET") or "default_dev_secret_key"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is required")

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    token=HF_TOKEN
)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI is required")

MONGO_DB_NAME = os.getenv("MONGO_DB_NAME") or "test"

_mongo_client = None
_mongo_db = None

def connect_to_mongo():
    global _mongo_client, _mongo_db
    if _mongo_client is None:
        _mongo_client = pymongo.MongoClient(MONGO_URI)
        _mongo_db = _mongo_client[MONGO_DB_NAME]
    return _mongo_db

# Load classifier model
model = joblib.load("business_question_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def classify_question(q):
    greetings = ["hi", "hello", "hey"]
    if any(greet in q.lower() for greet in greetings):
        return "greeting"
    vec = vectorizer.transform([q])
    prediction = model.predict(vec)[0]
    return "related" if prediction == 1 else "unrelated"

# OLD KEYWORD FUNCTION (Commented)
# def is_unrelated(q):
#     greetings = ["hi", "hello", "hey"]
#     if any(greet in q.lower() for greet in greetings):
#         return "greeting"
#     keywords = ["wedding", "vendor", "price", "venue", "service", "photography", "mangalam", "decorator", "dj", "cake"]
#     return "unrelated" if not any(k in q.lower() for k in keywords) else "related"

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    if not request.is_json:
        return jsonify({"answer": "Invalid request format. Please send JSON data."}), 400

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please provide a question."}), 400

    if "chat_history" not in session:
        session["chat_history"] = []

    if len(session["chat_history"]) > 10:
        session["chat_history"] = session["chat_history"][-10:]
        session.modified = True

    # Use ML-based classification
    check = classify_question(question)
    if check == "greeting":
        return jsonify({"answer": "🫐 Hello! I’m your Mangalam wedding assistant. How can I help with your big day? 💍"})
    elif check == "unrelated":
        return jsonify({"answer": "🫐 I specialize in wedding-related questions on Mangalam! Try asking about vendors, prices, or services 🌺"})

    price_match = re.search(r"\d+", question)
    if any(x in question.lower() for x in ["under", "below", "less than"]) and price_match:
        price = int(price_match.group(0))
        try:
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
        except Exception as e:
            return jsonify({"answer": "Sorry, I couldn’t fetch the vendor details right now. Please try again later."})

    if "dj" in question.lower():
        try:
            db = connect_to_mongo()
            djs = db["vendors"].find({"service_type": {"$regex": "dj", "$options": "i"}}).limit(5)
            results = list(djs)
            if not results:
                return jsonify({"answer": "No DJ services found currently."})

            response = "<strong>DJ Services:</strong><br>"
            for v in results:
                response += f"<br><strong>{v['service_name']}</strong><br>{v['description']}<br>💰 {v['price']} LKR<br>📦 Includes: {v['what_we_provide']}<hr>"
            return jsonify({"answer": response})
        except Exception as e:
            return jsonify({"answer": "Sorry, I couldn’t fetch the DJ services right now. Please try again later."})

    messages = []
    for turn in session["chat_history"][-3:]:
        messages.append({"role": "user", "content": turn["q"]})
        messages.append({"role": "assistant", "content": turn["a"]})
    messages.append({"role": "user", "content": question})

    answer = generate_response(messages)
    session["chat_history"].append({"q": question, "a": answer})
    session.modified = True

    return jsonify({"answer": answer})

@app.route("/reset", methods=["POST"])
def reset():
    session["chat_history"] = []
    return jsonify({"message": "Chat history cleared!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)