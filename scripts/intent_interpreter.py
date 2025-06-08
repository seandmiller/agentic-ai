import ollama
from .config import Config

class IntentInterpreter:
    def __init__(self):
        self.model = Config.INTENT_MODEL
    
    def interpret(self, user_input: str) -> str:
        """Determine if request needs code execution or conversation"""
        prompt = f"""Does this request need code execution?
        Example: 
        -tell me what the current weather is in Foster city California -> Code
        -How does the sun generate light -> Chat
                       

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
    
    def determine_execution_strategy(self, user_request: str) -> str:
        """Determine if request needs sequential steps or unified approach"""
        
        prompt = f"""Analyze this request and determine the best execution strategy:

Request: "{user_request}"

Sequential approach is needed when:
- Multiple distinct steps that pass data between them
- "Search X and then Y" patterns  
- "Get/fetch/find X and then create/analyze/process Y"
- External data gathering followed by processing

Unified approach is needed when:
- Single cohesive task (games, apps, tools)
- Creative projects without external dependencies
- Mathematical/computational problems
- File processing or data analysis of provided data

Respond with only: "sequential" or "unified"

Strategy:"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = response['message']['content'].strip().lower()
            return "sequential" if "sequential" in result else "unified"
                
        except Exception:
            # Default to unified for simplicity
            return "unified"
    
    def handle_conversation(self, user_input: str) -> str:
        """Handle regular conversation requests"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": user_input}]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: {e}"