import ollama
import subprocess
import tempfile
import os
from .config import Config
from .tools import CodeExtractor

class CodeExecutor:
    
    def __init__(self):
        self.model_name = Config.CODE_MODEL
        self.timeout = Config.TIMEOUT
    
    def generate_and_execute(self, user_request: str) -> dict:
        print("⚡ Generating code...")
        
   
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
        prompt = f"""{user_request}

Write Python code. Return ONLY executable Python code. No markdown, no explanations, no code blocks.

Start immediately with imports or code:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw_response = response['message']['content']
            
            # Use CodeExtractor to clean the response
            cleaned_code = CodeExtractor.extract_code(raw_response)
            
            if not cleaned_code:
                print(f"⚠️ No code extracted from AI response: {raw_response[:200]}...")
                return ""
            
            return cleaned_code
            
        except Exception as e:
            print(f"Error generating code: {e}")
            return ""
    
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