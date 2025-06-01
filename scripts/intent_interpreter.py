import ollama
from .config import Config

class IntentInterpreter:
    def __init__(self):
        self.intent_model = Config.INTENT_MODEL
        self.chat_model = Config.CHAT_MODEL
    
    def interpret(self, user_input: str) -> str:
        """Interpret if user wants code execution or general conversation"""
        
        prompt = f"""Analyze this user request and decide if they want code execution or general conversation. You are an agentic agent, so if the user requests something that you can solve by executing python code then do it.

User request: "{user_input}"

Respond with ONLY one word:
- "code" if they want code written/executed/demonstrated
- "no" if they want general conversation/explanation/information

Examples:
- "Write a function to sort numbers" → code
- "Create a snake game" → code  
- "Calculate the factorial of 5" → code
- "Show me how to use pandas" → code
- "im on a windows machine tell me what files are on my desktop" → code
- "i have a folder named images in my directory search it and tell me how many files are inside" → code
- "create a connect four game in the terminal" → code
- "What is Python?" → no
- "Hello how are you?" → no
- "import random" → no
- "import subprocess" → no
- "Explain machine learning" → no
- "How does subprocess work?" → no
- "What is the tempfile module?" → no

Response:"""

        try:
            response = ollama.chat(
                model=self.intent_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response['message']['content'].strip().lower()
            
            if "code" in response_text:
                return "code"
            else:
                return "no"
                
        except Exception as e:
            print(f"Error in interpretation: {e}")
            return "no"
    
    def handle_conversation(self, user_input: str) -> str:
        """Handle general conversation"""
        
        try:
            response = ollama.chat(
                model=self.chat_model,
                messages=[{"role": "user", "content": user_input}]
            )
            return response['message']['content']
        except Exception as e:
            return f"Sorry, I encountered an error: {e}"