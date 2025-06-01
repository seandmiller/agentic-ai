from scripts.intent_interpreter import IntentInterpreter
from scripts.code_executor import CodeExecutor
from scripts.config import Config

class AIAgent:
    def __init__(self):
        self.interpreter = IntentInterpreter()
        self.executor = CodeExecutor()
    
    def process(self, user_input: str) -> dict:
        """Main method: interpret intent and execute code if needed"""
        
        print(f"🤖 Processing: '{user_input}'")
        
        # Step 1: Interpret user intent
        intent = self.interpreter.interpret(user_input)
        print(f"💭 Intent: {intent}")
        
        if intent == "code":
            # Step 2: Generate and execute code
            result = self.executor.generate_and_execute(user_input)
            result['intent'] = 'code'
            return result
        else:
            # Step 3: Handle general conversation
            response = self.interpreter.handle_conversation(user_input)
            return {
                'intent': 'chat',
                'success': True,
                'response': response,
                'code': '',
                'output': '',
                'error': ''
            }


def main():
    """Interactive chat with the AI agent"""
    agent = AIAgent()
    
    print("🤖 AI Agent with DeepSeek-R1 Ready!")
    print("💬 Type your requests below")
    print("🔧 Configuration commands:")
    print("   - 'config' to show current settings")
    print("   - 'fast' for fast mode (deepseek-r1:1.5b)")
    print("   - 'balanced' for balanced mode (deepseek-r1:7b)")
    print("   - 'powerful' for powerful mode (deepseek-r1)")
    print("   - 'quit' to exit")
    print("="*60)
    
    # Show initial config
    Config.print_config()
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            # Check for config commands
            elif user_input.lower() == 'config':
                Config.print_config()
                continue
            elif user_input.lower() == 'fast':
                Config.set_performance_mode("fast")
                agent = AIAgent()  # Reinitialize with new config
                continue
            elif user_input.lower() == 'balanced':
                Config.set_performance_mode("balanced")
                agent = AIAgent()  # Reinitialize with new config
                continue
            elif user_input.lower() == 'powerful':
                Config.set_performance_mode("quality")
                agent = AIAgent()  # Reinitialize with new config
                continue
            
            if not user_input:
                continue
            
            # Process the request
            result = agent.process(user_input)
            
            # Display results based on intent
            if result['intent'] == 'code':
                print(f"✅ Success: {result['success']}")
                if result['output']:
                    print(f"📤 Output:\n{result['output']}")
                if result['error']:
                    print(f"❌ Error:\n{result['error']}")
            else:
                print(f"🤖 AI: {result['response']}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()