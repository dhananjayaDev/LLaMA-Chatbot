# Mangalam Wedding Chatbot

A Flask-based intelligent chatbot integrated with Hugging Face language models and MongoDB for providing real-time wedding service assistance. This project is tailored for the Mangalam platform to assist users in discovering wedding vendors, services, and prices using natural language queries.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Switching Language Models](#switching-language-models)
- [Deployment Recommendations](#deployment-recommendations)
- [License](#license)

---

## Overview

This chatbot system is designed to enhance the user experience on the Mangalam wedding platform by enabling users to inquire about wedding services in a conversational manner. It integrates a language model hosted on Hugging Face and retrieves contextual vendor data from a MongoDB database.

---

## Features

- Flask-based web server and user interface
- Chatbot interface with contextual memory
- Vendor and service filtering by type or price
- Secure environment variable configuration using `.env`
- Model switching capability with Hugging Face Inference API

---

## Technology Stack

| Component        | Technology                          |
|------------------|--------------------------------------|
| Backend          | Python (Flask)                       |
| Frontend         | HTML/CSS (Jinja templates)           |
| NLP Model        | Hugging Face Inference API           |
| Database         | MongoDB Atlas                        |
| Configuration    | dotenv                               |

---

## Project Structure

```
chatbot00/
├── app.py                    # Main Flask application
├── inspect_mongo.py         # MongoDB inspection utility
├── templates/
│   └── index.html           # Chatbot UI template
├── static/
│   └── style.css            # Frontend styling
├── .env                     # Environment variables (excluded from Git)
├── .gitignore               # Git ignore list
├── requirements.txt         # Python dependencies
```

---

## Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/dhananjayaDev/LLaMA-Chatbot.git
cd LLaMA-Chatbot
```

2. **Set Up Python Virtual Environment**

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Create `.env` File**

Create a file named `.env` in the project root with the following content:

```env
HF_TOKEN=your_huggingface_token
FLASK_SECRET=your_flask_secret_key
MONGO_URI=your_mongodb_connection_uri
MONGO_DB_NAME=test
```

> Note: Ensure `.env` is excluded via `.gitignore`.

---

## Running the Application

```bash
python app.py
```

The server will start at: [http://localhost:5003](http://localhost:5003)

---

## Switching Language Models

To use a different model such as LLaMA or Falcon:

1. Go to [huggingface.co](https://huggingface.co/models)
2. Choose a model that supports text generation
3. Update `app.py`:

```python
client = InferenceClient(
    model="meta-llama/Llama-2-7b-chat-hf",
    token=os.getenv("HF_TOKEN")
)
```

Ensure you have access to the selected model and that it is suitable for chat tasks.

---

## Deployment Recommendations

This app can be deployed using platforms such as:

- Render.com
- Railway.app
- Heroku

To deploy:
- Include a `Procfile` with the line: `web: python app.py`
- Set your environment variables in the hosting platform securely

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

Developed by Dhananjaya | GitHub: [@dhananjayaDev](https://github.com/dhananjayaDev)

For questions, please raise an issue in the repository.

