import ollama
import json
from typing import List, Dict
from dataclasses import dataclass
from scripts.code_executor import CodeExecutor
from scripts.intent_interpreter import IntentInterpreter
from scripts.config import Config
from scripts.tools import JSONExtractor

@dataclass
class AgenticTask:
    description: str
    goal: str
    context_variables: List[str] = None

class AgenticExecutor:
    
    def __init__(self):
        self.model_name = getattr(Config, 'INTENT_MODEL', Config.CODE_MODEL)
        self.code_executor = CodeExecutor()
        self.intent_interpreter = IntentInterpreter()
        self.context = {}
    
    def execute(self, user_request: str) -> dict:
        print(f"🤖 Processing: {user_request}")
        
        # First: Determine if this needs code execution or just conversation
        intent = self.intent_interpreter.interpret(user_request)
        print(f"🧠 Intent: {intent}")
        
        if intent == "chat":
            # Handle as regular conversation
            response = self.intent_interpreter.handle_conversation(user_request)
            print(f"💬 Chat response provided")
            return {'success': True, 'message': response, 'type': 'chat'}
        
        # It's a code request - determine execution strategy
        execution_strategy = self.intent_interpreter.determine_execution_strategy(user_request)
        print(f"📋 Strategy: {execution_strategy}")
        
        if execution_strategy == "unified":
            return self._execute_unified(user_request)
        else:
            return self._execute_sequential(user_request)
    

    def _execute_unified(self, user_request: str) -> dict:
        """Execute as one comprehensive solution"""
        print(f"⚡ Unified execution")
        
        unified_prompt = self._create_unified_prompt(user_request)
        result = self.code_executor.generate_and_execute(unified_prompt)
        
        if result['success']:
            print(f"✅ Unified execution completed")
            self._extract_context(result['output'])
            return {'success': True, 'message': "Completed unified execution", 'context': self.context, 'type': 'code'}
        else:
            return {'success': False, 'error': f"Unified execution failed: {result['error']}", 'type': 'code'}
    
    def _execute_sequential(self, user_request: str) -> dict:
        """Execute as sequential tasks with data passing"""
        print(f"⚡ Sequential execution")
        
        tasks = self._break_down_request(user_request)
        
        # Fallback to unified if task breakdown fails
        if not tasks:
            print("🔄 Task breakdown failed, falling back to unified approach")
            return self._execute_unified(user_request)
        
        print(f"📋 Sequential tasks: {len(tasks)}")
        
        for i, task in enumerate(tasks, 1):
            print(f"\n🔗 Step {i}: {task.description}")
            
            result = self._execute_task(task)
            
            if not result['success']:
                return {'success': False, 'error': f"Step {i} failed: {result['error']}", 'type': 'code'}
            
            print(f"✅ Step {i} completed")
        
        return {'success': True, 'message': f"Completed {len(tasks)} sequential steps", 'context': self.context, 'type': 'code'}
    
    def _create_unified_prompt(self, user_request: str) -> str:
        """Create a comprehensive prompt for unified execution"""
        
        # Include any existing context
        context_code = ""
        if self.context:
            for var, value in self.context.items():
                if isinstance(value, str):
                    context_code += f'{var} = """{value}"""\n'
                else:
                    context_code += f'{var} = {repr(value)}\n'
        
        return f"""{context_code}
# Complete Request: {user_request}

# Instructions:
# - This is a unified task - handle the COMPLETE request in one comprehensive solution
# - Write a single, cohesive program that accomplishes everything requested
# - Think through all requirements and implement them together
# - Include all necessary imports at the top
# - Store any important final results as variables and print them as:
#   print(f"RESULT: variable_name = {{variable_name}}")
# - Write clean, working Python code that fully satisfies the request

# Your comprehensive solution:"""

    def _break_down_request(self, user_request: str) -> List[AgenticTask]:
        """Break down request into sequential tasks with data passing"""
        prompt = f'''This request needs sequential execution with data passing between steps.
Break it down into ordered tasks where later tasks use results from earlier ones.

Request: "{user_request}"

Return ONLY the JSON array:

[
  {{
    "description": "What this step does", 
    "goal": "Specific Python code goal for this step",
    "context_variables": ["data_from_previous_steps"]
  }}
]'''

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        
        print(response['message']['content'])
        tasks_json = JSONExtractor.extract_json(response['message']['content'])
        
        if not tasks_json:
            print(f"⚠️ Failed to extract JSON from model response: {response['message']['content'][:200]}...")
            # Return empty list to trigger fallback
            return []
        
        return [AgenticTask(
            description=task['description'],
            goal=task['goal'],
            context_variables=task.get('context_variables', [])
        ) for task in tasks_json]
    
    def _execute_task(self, task: AgenticTask) -> dict:
        """Execute individual task with context from previous steps"""
        code_prompt = self._build_task_prompt(task)
        result = self.code_executor.generate_and_execute(code_prompt)
        
        if result['success']:
            self._extract_context(result['output'])
        
        return result
    
    def _build_task_prompt(self, task: AgenticTask) -> str:
        """Build prompt for individual task with context"""
        context_code = ""
        
        if task.context_variables:
            for var in task.context_variables:
                if var in self.context:
                    value = self.context[var]
                    if isinstance(value, str):
                        context_code += f'{var} = """{value}"""\n'
                    else:
                        context_code += f'{var} = {repr(value)}\n'
        
        return f"""{context_code}
# Step Goal: {task.goal}

# Instructions:
# - Focus ONLY on this specific step
# - Use any provided context variables from previous steps
# - Store important results for next steps using:
#   print(f"RESULT: variable_name = {{variable_name}}")
# - Keep this step focused and don't try to do everything

# Your code for this step:"""
    
    def _extract_context(self, output: str):
        """Extract context variables from execution output"""
        for line in output.split('\n'):
            if 'RESULT:' in line and '=' in line:
                try:
                    var_part = line.split('RESULT:', 1)[1].strip()
                    var_name = var_part.split('=', 1)[0].strip()
                    var_value = var_part.split('=', 1)[1].strip()
                    self.context[var_name] = var_value
                    print(f"📥 Stored: {var_name}")
                except:
                    pass

def main():
    executor = AgenticExecutor()
    
    print("🚀 AI Agent Ready - Chat or Code")
    print("Examples:")
    print("  💬 Chat: 'What's the weather like?'")
    print("  🔗 Sequential: 'Search web for Tesla stock and make a poem about it'")
    print("  ⚡ Unified: 'Create a snake game in Python'")
    print()
    
    while True:
        request = input("\n> ")
        if request.lower() in ['quit', 'exit']:
            break
            
        result = executor.execute(request)
        
        if result['success']:
            if result.get('type') == 'chat':
                print(f"💬 {result['message']}")
            else:
                print(f"🎉 {result['message']}")
        else:
            print(f"❌ {result['error']}")

if __name__ == "__main__":
    main()