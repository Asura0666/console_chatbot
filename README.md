# Console Chatbot using LangChain & Gemini

A **console-based chatbot application** built with **Python** and **LangChain**, powered by **Google Gemini models**. This chatbot runs directly in the terminal, providing intelligent responses based on user input.

Developed as part of a **remote internship at NTT DATA**.

---

## Tech Stack

- **Python 3.x**
- **LangChain**
- **Google Gemini API**
- **python-dotenv**
- **Git** (for version control)

---

## Project Structure

```
console_chatbot/
├── main.py              # Main application script
├── requirements.txt     # Python dependencies
├── .env                # Environment variables (API key)
├── README.md           # Project documentation
└── .gitignore          # Git ignore rules
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Asura0666/console_chatbot.git
cd console_chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

**Activate the environment:**

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

- **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
touch .env  # Linux/macOS
# or create manually on Windows
```

Add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

> ⚠️ **Important:** Never commit your `.env` file or expose your API key publicly. Ensure `.env` is listed in `.gitignore`.

---

## Running the Chatbot

```bash
python main.py
```

or

```bash
python3 main.py
```

The chatbot will start in your terminal. Type your messages and press Enter to receive responses.

---

## Features

- **Terminal-based interface** – Simple, lightweight, and fast
-  **Powered by Google Gemini** – Advanced language model capabilities
-  **Secure configuration** – API keys stored securely in environment variables
-  **Conversation memory** – Maintains context within the session
-  **Error handling** – Graceful handling of API issues and user interrupts

---

## Example Usage

```bash
$ python main.py
=========================================
Gemini Chatbot Initialized!
Type 'exit' or 'quit' to quit
=========================================

You: Hello, how are you?
Bot: Hello! I'm doing well, thank you for asking. How can I assist you today?

You: Can you explain quantum computing?
Bot: Quantum computing is a type of computation that uses quantum bits...
```

---

##  Author

**Dhiraj Lande**  
Remote Intern – NTT DATA  
[GitHub Profile](https://github.com/Asura0666)

---

## Mentorship & Guidance

### Project Mentor
**Mr. Ramesh Pothikara**  
Principal Client Service Transition Specialist  
NTT Global Data Centers & Cloud Infrastructure India Pvt. Ltd.

### Internship Supervisor
**Mr. Anish Khot**  
Human Resources  
NTT Global Data Centers & Cloud Infrastructure India Pvt. Ltd.

---

## License

This project is created for educational purposes as part of an internship with NTT DATA.

---

## Useful Links

- [LangChain Documentation](https://python.langchain.com/)
- [Google AI Studio](https://makersuite.google.com/app/apikey)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
