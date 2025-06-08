import ollama
import subprocess
import tempfile
import os
from .config import Config

class CodeExecutor:
    
    def __init__(self):
        self.model_name = Config.CODE_MODEL
        self.timeout = Config.TIMEOUT
    
    def generate_and_execute(self, user_request: str) -> dict:
        print("⚡ Generating code...")
        
        # Generate the Python code
        code = self._generate_code(user_request)
        if not code:
            return {
                'success': False,
                'output': '',
                'error': 'Failed to generate code',
                'code': ''
            }
        
        if Config.SHOW_GENERATED_CODE:
            print(f"📝 Generated code:\n{code}")
        
        # Execute the code
        return self._execute_code(code)
    
    def _generate_code(self, user_request: str) -> str:
        prompt = f"""Write Python code to accomplish this task: {user_request}

IMPORTANT RULES:
- Write ONLY executable Python code
- NO explanations before or after the code
- NO markdown code blocks
- Start immediately with import statements or code
- Include all necessary imports
- Make the code complete and runnable
- Add print statements to show results

Code:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._clean_code(response['message']['content'].strip())
            
        except Exception as e:
            print(f"Error generating code: {e}")
            return ""
    
    def _clean_code(self, raw_code: str) -> str:
        if not raw_code:
            return ""
        
        # Remove code blocks if present
        import re
        
        # Try to extract from ```python blocks
        python_block = re.search(r'```python\s*(.*?)\s*```', raw_code, re.DOTALL)
        if python_block:
            return python_block.group(1).strip()
        
        # Try to extract from ``` blocks
        code_block = re.search(r'```\s*(.*?)\s*```', raw_code, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        
        # Remove obvious explanation lines
        lines = raw_code.split('\n')
        filtered_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip explanation lines
            if any(stripped.lower().startswith(phrase) for phrase in 
                  ['sure', 'here', 'let me', 'i will', 'this code', 'the code']):
                continue
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines).strip()
    
    def _execute_code(self, code: str) -> dict:
        # Write the code to a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute the Python file
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            success = result.returncode == 0
            
            return {
                'success': success,
                'output': result.stdout,
                'error': result.stderr,
                'code': code
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': f'Code execution timed out after {self.timeout} seconds',
                'code': code
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'Execution error: {str(e)}',
                'code': code
            }
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass