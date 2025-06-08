import ollama
import json
from typing import List, Dict
from dataclasses import dataclass
from scripts.code_executor import CodeExecutor
from scripts.config import Config

@dataclass
class AgenticTask:
    description: str
    goal: str
    context_variables: List[str] = None

class AgenticExecutor:
    
    def __init__(self):
        self.model_name = getattr(Config, 'INTENT_MODEL', Config.CODE_MODEL)
        self.code_executor = CodeExecutor()
        self.context = {}
    
    def execute(self, user_request: str) -> dict:
        print(f"🤖 Executing: {user_request}")
        
        tasks = self._break_down_request(user_request)
        print(f"📋 Tasks: {tasks}")
        
        
        for i, task in enumerate(tasks, 1):
            print(f"\n⚡ Task {i}: {task.description}")
            
            result = self._execute_task(task)
            
            if not result['success']:
                return {'success': False, 'error': f"Task {i} failed: {result['error']}"}
            
            print(f"✅ Task {i} completed")
        
        return {'success': True, 'message': f"Completed {len(tasks)} tasks", 'context': self.context}
    
    def _break_down_request(self, user_request: str) -> List[AgenticTask]:
        prompt = f'''Break down this request into sequential tasks. Return ONLY the JSON array, nothing else:

"{user_request}"

[
  {{
    "description": "Brief task description", 
    "goal": "Specific Python code goal",
    "context_variables": ["var1", "var2"]
  }}
]'''

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        
        tasks_json = json.loads(response['message']['content'])
        
        return [AgenticTask(
            description=task['description'],
            goal=task['goal'],
            context_variables=task.get('context_variables', [])
        ) for task in tasks_json]
    
    def _execute_task(self, task: AgenticTask) -> dict:
        code_prompt = self._build_prompt(task)
        result = self.code_executor.generate_and_execute(code_prompt)
        
        if result['success']:
            self._extract_context(result['output'])
        
        return result
    
    def _build_prompt(self, task: AgenticTask) -> str:
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
# Task: {task.goal}

# Important: Store results in variables and print them as:
# print(f"RESULT: variable_name = {{variable_name}}")

# Your code here:"""
    
    def _extract_context(self, output: str):
        for line in output.split('\n'):
            if 'RESULT:' in line and '=' in line:
                try:
                    var_part = line.split('RESULT:', 1)[1].strip()
                    var_name = var_part.split('=', 1)[0].strip()
                    var_value = var_part.split('=', 1)[1].strip()
                    self.context[var_name] = var_value
                    print(f"📥 {var_name}")
                except:
                    pass

def main():
    executor = AgenticExecutor()
    
    while True:
        request = input("\n> ")
        if request.lower() in ['quit', 'exit']:
            break
            
        result = executor.execute(request)
        
        if result['success']:
            print(f"🎉 {result['message']}")
        else:
            print(f"❌ {result['error']}")

if __name__ == "__main__":
    main()