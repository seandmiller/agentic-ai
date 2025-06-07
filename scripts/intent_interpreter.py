import ollama
from .config import Config

class IntentInterpreter:
    def __init__(self):
        self.model = Config.INTENT_MODEL
    
    def interpret(self, user_input: str) -> str:
        prompt = f"""Does this request need code execution?

Request: "{user_input}"

Respond with only: "code" or "chat"

Response:"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = response['message']['content'].strip().lower()
            return "code" if "code" in result else "chat"
                
        except Exception:
            return "chat"
    
    def handle_conversation(self, user_input: str) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": user_input}]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: {e}"