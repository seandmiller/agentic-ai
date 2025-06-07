import ollama
import json
import re
from typing import List, Dict, Any, Optional
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
        self.execution_log = []
    
    def execute(self, user_request: str) -> dict:
        print(f"🤖 Starting agentic execution: {user_request}")
        
        self._reset_state()
        
        tasks = self._break_down_request(user_request)
        print(tasks)
        if not tasks:
            return {'success': False, 'error': 'Failed to break down request into tasks'}
        
        print(f"📋 Identified {len(tasks)} tasks to execute")
        
        for i, task in enumerate(tasks, 1):
            print(f"\n⚡ Task {i}/{len(tasks)}: {task.description}")
            
            result = self._execute_task(task, i)
            
            if not result['success']:
                return {
                    'success': False,
                    'error': f"Task {i} failed: {result['error']}",
                    'completed_tasks': i-1,
                    'execution_log': self.execution_log
                }
            
            print(f"✅ Task {i} completed successfully")
        
        return {
            'success': True,
            'message': f"Successfully completed all {len(tasks)} tasks",
            'execution_log': self.execution_log,
            'final_context': self.context
        }
    
    def _break_down_request(self, user_request: str) -> List[AgenticTask]:
        prompt = f"""Break down this user request into separate, sequential tasks that can be accomplished with Python code.

USER REQUEST: "{user_request}"

Each task should be a clear, actionable step that can be coded. Think about:
- What needs to be searched/fetched from the web
- What data needs to be processed or transformed  
- What content needs to be generated
- What files need to be created/read
- How results from one task feed into the next

Return as JSON array in this format:
[
  {{
    "description": "Brief description of what this task does",
    "goal": "Specific goal that can be accomplished with Python code",
    "context_variables": ["variable1", "variable2"] // variables from previous tasks this needs
  }}
]

Example for "search tesla news and create poem":
[
  {{
    "description": "Search web for Tesla stock news",
    "goal": "Use web scraping/API to get latest Tesla stock news and store in 'tesla_news' variable",
    "context_variables": []
  }},
  {{
    "description": "Create poem about Tesla news", 
    "goal": "Generate a creative poem based on the Tesla news data and store in 'poem_text' variable",
    "context_variables": ["tesla_news"]
  }},
  {{
    "description": "Save poem to desktop file",
    "goal": "Write the poem to a .txt file on the desktop",
    "context_variables": ["poem_text"]
  }}
]

Return ONLY the JSON array:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            print(response['message']['content'])
            tasks_json = self._extract_json(response['message']['content'])
            if not tasks_json:
                return []
            
            tasks = []
            for task_data in tasks_json:
                task = AgenticTask(
                    description=task_data['description'],
                    goal=task_data['goal'],
                    context_variables=task_data.get('context_variables', [])
                )
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            print(f"Error breaking down request: {e}")
            return []
    
    def _execute_task(self, task: AgenticTask, task_number: int) -> dict:
        code_prompt = self._build_code_prompt(task)
        
        print(f"🔧 Generating code for: {task.goal}")
        
        result = self.code_executor.generate_and_execute(code_prompt)
        
        if result['success']:
            output = result['output']
            self._extract_variables_from_output(output, task_number)
            
            self.execution_log.append({
                'task_number': task_number,
                'description': task.description,
                'goal': task.goal,
                'code': result['code'],
                'output': output,
                'success': True
            })
        else:
            self.execution_log.append({
                'task_number': task_number,
                'description': task.description,
                'goal': task.goal,
                'code': result.get('code', ''),
                'error': result['error'],
                'success': False
            })
        
        return result
    
    def _build_code_prompt(self, task: AgenticTask) -> str:
        context_info = ""
        if task.context_variables:
            context_info = "AVAILABLE CONTEXT VARIABLES:\n"
            for var in task.context_variables:
                if var in self.context:
                    value_preview = str(self.context[var])[:200]
                    if len(str(self.context[var])) > 200:
                        value_preview += "..."
                    context_info += f"- {var}: {value_preview}\n"
                else:
                    context_info += f"- {var}: (not available)\n"
            context_info += "\n"
        
        prompt = f"""{context_info}TASK: {task.goal}

Write Python code to accomplish this task. The code should:
1. Complete the specific goal described
2. Store any important results in clearly named variables
3. Print the variable name and its value at the end so results can be captured
4. Use appropriate libraries (requests for web, os for files, etc.)
5. Handle errors gracefully

Example output format:
```
# Your code here
result_variable = "some result"
print(f"RESULT_VAR: result_variable = {{result_variable}}")
```

Make sure to print variables that might be needed for subsequent tasks."""

        return prompt
    
    def _extract_variables_from_output(self, output: str, task_number: int):
        lines = output.split('\n')
        
        for line in lines:
            if 'RESULT_VAR:' in line:
                try:
                    var_part = line.split('RESULT_VAR:', 1)[1].strip()
                    if '=' in var_part:
                        var_name = var_part.split('=', 1)[0].strip()
                        var_value = var_part.split('=', 1)[1].strip()
                        self.context[var_name] = var_value
                        print(f"📥 Captured variable: {var_name}")
                except:
                    pass
            
            elif '=' in line and any(keyword in line.lower() for keyword in 
                                   ['news', 'data', 'result', 'content', 'text', 'poem', 'summary']):
                try:
                    var_name = line.split('=')[0].strip()
                    if var_name.isidentifier():
                        var_value = '='.join(line.split('=')[1:]).strip()
                        if var_value.startswith('"') or var_value.startswith("'"):
                            self.context[var_name] = var_value.strip('"\'')
                        else:
                            self.context[var_name] = var_value
                        print(f"📥 Auto-captured variable: {var_name}")
                except:
                    pass
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and '=' not in line:
                if len(line) > 50 and any(keyword in task.goal.lower() for keyword in 
                                        ['search', 'find', 'get', 'fetch', 'create', 'generate']):
                    var_name = f"task_{task_number}_output"
                    self.context[var_name] = line
                    print(f"📥 Captured output as: {var_name}")
                    break
    
    def _extract_json(self, text: str) -> Optional[List[Dict]]:
        if not text.strip():
            return None
        
        # Remove common AI response prefixes
        text = re.sub(r'^(Sure!|Here\'s|Here is|Here are).*?[\n:]', '', text, flags=re.IGNORECASE)
        
        # Strategy 1: Try to extract from code blocks
        code_block_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'`\s*(.*?)\s*`'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match.strip())
                    if isinstance(result, list):
                        return result
                except:
                    continue
        
        # Strategy 2: Find JSON arrays with better regex
        json_patterns = [
            r'\[\s*\{.*?\}\s*\]',  # Complete array with objects
            r'\[[\s\S]*?\]',       # Any array content
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Clean up the match
                    cleaned = match.strip()
                    
                    # Balance brackets if needed
                    open_brackets = cleaned.count('[')
                    close_brackets = cleaned.count(']')
                    if open_brackets > close_brackets:
                        cleaned += ']' * (open_brackets - close_brackets)
                    
                    result = json.loads(cleaned)
                    if isinstance(result, list) and len(result) > 0:
                        return result
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
                    continue
                except Exception as e:
                    print(f"Other error: {e}")
                    continue
        
        # Strategy 3: Try to parse the entire text
        try:
            text_cleaned = text.strip()
            if text_cleaned.startswith('[') and text_cleaned.endswith(']'):
                return json.loads(text_cleaned)
        except:
            pass
        
        # Strategy 4: Last resort - try to fix common JSON issues
        try:
            # Fix common issues
            fixed_text = text.strip()
            
            # Remove trailing commas
            fixed_text = re.sub(r',\s*}', '}', fixed_text)
            fixed_text = re.sub(r',\s*]', ']', fixed_text)
            
            # Fix unquoted keys
            fixed_text = re.sub(r'(\w+):', r'"\1":', fixed_text)
            
            # Try to find and extract just the array part
            start_idx = fixed_text.find('[')
            if start_idx != -1:
                bracket_count = 0
                end_idx = start_idx
                
                for i in range(start_idx, len(fixed_text)):
                    if fixed_text[i] == '[':
                        bracket_count += 1
                    elif fixed_text[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i
                            break
                
                if bracket_count == 0:
                    json_str = fixed_text[start_idx:end_idx + 1]
                    result = json.loads(json_str)
                    if isinstance(result, list):
                        return result
        except Exception as e:
            print(f"Final parsing attempt failed: {e}")
        
        print(f"❌ Could not extract JSON from response. Text preview: {text[:200]}...")
        return None
    def _reset_state(self):
        self.context.clear()
        self.execution_log.clear()
    
    def get_context_summary(self) -> dict:
        return {
            'variables': list(self.context.keys()),
            'context_data': {k: str(v)[:100] + "..." if len(str(v)) > 100 else str(v) 
                           for k, v in self.context.items()},
            'total_tasks': len(self.execution_log)
        }

if __name__ == "__main__":
    executor = AgenticExecutor()
    
    test_requests = [
        "search the web for latest tesla stock news and create a poem about it, then save it to my desktop",
        "find current AI news, summarize the key points, and save summary to a file",
        "get weather information for San Francisco and write a short story about it"
    ]
    
    for request in test_requests:
        print(f"\n{'='*60}")
        print(f"Testing: {request}")
        print('='*60)
        
        result = executor.execute(request)
        
        if result['success']:
            print(f"\n🎉 SUCCESS: {result['message']}")
            summary = executor.get_context_summary()
            print(f"📊 Variables created: {', '.join(summary['variables'])}")
        else:
            print(f"\n❌ FAILED: {result['error']}")
        
        print(f"\nExecution log: {len(executor.execution_log)} tasks completed")