import ollama
import json
from typing import List, Dict
from dataclasses import dataclass
from scripts.code_executor import CodeExecutor
from scripts.intent_interpreter import IntentInterpreter
from scripts.config import Config
from scripts.tools import JSONExtractor, CodeExtractor

@dataclass
class AgenticTask:
    description: str
    goal: str
    context_variables: List[str] = None

@dataclass
class TaskResult:
    task: AgenticTask
    success: bool
    code: str = ""
    error: str = ""

class CodeMerger:
    """Handles merging multiple code blocks into a cohesive final product using LLM"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def merge_code_blocks(self, task_results: List[TaskResult], original_request: str) -> str:
        """Merge successful task results into one cohesive program using LLM"""
        successful_results = [r for r in task_results if r.success]
        
        if not successful_results:
            return ""
        
        if len(successful_results) == 1:
            # Only one successful task, return its code cleaned
            return CodeExtractor.extract_code(successful_results[0].code) or ""
        
        # Prepare the merging prompt
        merge_prompt = self._create_merge_prompt(successful_results, original_request)
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": merge_prompt}]
            )
            
            merged_code = response['message']['content'].strip()
            
            # Clean the merged code using CodeExtractor to ensure clean executable Python code
            cleaned_merged_code = CodeExtractor.extract_code(merged_code)
            
            return cleaned_merged_code or ""
            
        except Exception as e:
            print(f"⚠️ LLM merge failed: {e}")
            return ""
    
    def _create_merge_prompt(self, successful_results: List[TaskResult], original_request: str) -> str:
        """Create a prompt for LLM to merge the code blocks"""
        
        code_blocks = []
        for i, result in enumerate(successful_results, 1):
            # Clean the code using CodeExtractor before including in prompt
            cleaned_code = CodeExtractor.extract_code(result.code)
            if cleaned_code:
                code_blocks.append(f"""
# Task {i}: {result.task.description}
# Goal: {result.task.goal}
```python
{cleaned_code}
```
""")
        
        prompt = f"""You need to merge multiple Python code blocks into one cohesive, working program.

Original Request: "{original_request}"

Here are the individual code blocks from sequential tasks:
{"".join(code_blocks)}

Instructions for merging:
1. Combine all code into ONE cohesive Python program that fulfills the original request
2. Remove duplicate imports and organize them at the top
3. Merge functions and classes intelligently - avoid duplicates
4. Organize the main execution logic in a logical flow
5. Ensure all variables and data flow between the original tasks is preserved
6. Add comments to explain the merged sections if helpful
7. Make sure the final program is complete and executable
8. Return ONLY the merged Python code, no explanations

Merged Python code:"""

        return prompt

class AgenticExecutor:
    
    def __init__(self):
        self.model_name = getattr(Config, 'INTENT_MODEL', Config.CODE_MODEL)
        self.code_executor = CodeExecutor()
        self.intent_interpreter = IntentInterpreter()
        self.code_merger = CodeMerger(self.model_name)  # Pass model name to merger
        self.context = {}
        self.task_results = []  # Store all task results for merging
    
    def execute(self, user_request: str) -> dict:
        print(f"🤖 Processing: {user_request}")
        
        # Reset task results for new request
        self.task_results = []
        
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
            return {'success': True, 'message': "Completed unified execution", 
                   'context': self.context, 'type': 'code', 'final_code': result.get('code', '')}
        else:
            return {'success': False, 'error': f"Unified execution failed: {result['error']}", 'type': 'code'}
    
    def _execute_sequential(self, user_request: str) -> dict:
        """Generate code for sequential tasks, merge, then execute final product"""
        print(f"⚡ Sequential execution")
        
        tasks = self._break_down_request(user_request)
        
        # Fallback to unified if task breakdown fails
        if not tasks:
            print("🔄 Task breakdown failed, falling back to unified approach")
            return self._execute_unified(user_request)
        
        print(f"📋 Sequential tasks: {len(tasks)}")
        
        # Generate code for each task (don't execute individually)
        for i, task in enumerate(tasks, 1):
            print(f"\n🔗 Step {i}: {task.description}")
            
            result = self._generate_task_code(task)
            
            # Store task result
            task_result = TaskResult(
                task=task,
                success=result['success'],
                code=result.get('code', ''),
                error=result.get('error', '')
            )
            self.task_results.append(task_result)
            
            if not result['success']:
                print(f"⚠️ Step {i} code generation failed, but continuing with other tasks")
                continue
            
            print(f"✅ Step {i} code generated")
        
        # Merge and execute the final product
        return self._finalize_sequential_execution(user_request)
    
    def _finalize_sequential_execution(self, user_request: str) -> dict:
        """Merge all successful tasks and execute final product"""
        successful_tasks = [r for r in self.task_results if r.success]
        
        if not successful_tasks:
            return {'success': False, 'error': 'No tasks generated code successfully', 'type': 'code'}
        
        print(f"\n🔄 Merging {len(successful_tasks)} successful tasks using LLM...")
        
        # Merge code into final product using LLM
        final_code = self.code_merger.merge_code_blocks(self.task_results, user_request)
        
        if not final_code:
            return {'success': False, 'error': 'Failed to merge code', 'type': 'code'}
        
        print("🧩 Code merged successfully")
        print("🚀 Executing final merged code...")
        
        # Execute the final merged code
        execution_result = self.code_executor._execute_code(final_code)
        
        if execution_result['success']:
            print("✅ Final code executed successfully!")
            # Extract context from final execution
            self._extract_context(execution_result['output'])
            
            return {
                'success': True, 
                'message': f"Generated {len(successful_tasks)} code blocks, merged and executed successfully",
                'context': self.context,
                'type': 'code',
                'final_code': final_code,
                'output': execution_result['output']
            }
        else:
            print(f"❌ Final code execution failed: {execution_result['error']}")
            return {
                'success': False,
                'error': f"Final code execution failed: {execution_result['error']}",
                'type': 'code',
                'final_code': final_code
            }
    
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
ONLY return json data NO explanations NO other texts or strings in output JUST JSON DATA WE ARE on A windows System
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
    
    def _generate_task_code(self, task: AgenticTask) -> dict:
        """Generate code for individual task (don't execute)"""
        code_prompt = self._build_task_prompt(task)
        
        try:
            code = self.code_executor._generate_code(code_prompt)
            if not code:
                return {
                    'success': False,
                    'error': 'Failed to generate code for task',
                    'code': ''
                }
            
            return {
                'success': True,
                'code': code
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Code generation error: {str(e)}',
                'code': ''
            }
    
    def _build_task_prompt(self, task: AgenticTask) -> str:
        """Build prompt for individual task code generation"""
        
        return f"""# Step Goal: {task.goal}

# Instructions:
# - Focus ONLY on this specific step
# - This code will be merged with other tasks later
# - Write clean, focused Python code for this specific goal
# - Don't worry about execution - just generate good code
# - Include necessary imports for this step

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
                
                # Show execution output if available
                if result.get('output'):
                    print(f"\n📤 Output:\n{result['output']}")
                
                # Show final merged code if available
                if 'final_code' in result:
                    print(f"\n📄 Final code available")
                    show_code = input("Show final code? (y/n): ")
                    if show_code.lower() == 'y':
                        print(f"\n{'='*50}")
                        print(result['final_code'])
                        print(f"{'='*50}")
        else:
            print(f"❌ {result['error']}")

if __name__ == "__main__":
    main()
