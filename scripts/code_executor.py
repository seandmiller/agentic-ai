import ollama
import subprocess
import tempfile
import os
import sys
from .config import Config

class CodeExecutor:
    def __init__(self):
        self.model_name = Config.CODE_MODEL
        self.timeout = Config.TIMEOUT
        self.max_retries = Config.MAX_RETRIES
    
    def generate_and_execute(self, user_request: str) -> dict:
        """Generate Python code from user request and execute it with error recovery"""
        
        if Config.SHOW_EXECUTION_STEPS:
            print("⚡ Generating code...")
        
        # Step 1: Generate initial code
        generated_code = self._generate_code(user_request)
        if not generated_code:
            return {
                'success': False,
                'output': '',
                'error': 'Failed to generate code',
                'code': '',
                'timeout': False
            }
        
        if Config.SHOW_GENERATED_CODE:
            print(f"📝 Code generated:\n{generated_code}")
        
        if Config.SHOW_EXECUTION_STEPS:
            print("🔥 Executing...")
        
        # Step 2: Try to execute the code
        result = self._execute_code(generated_code)
        result['code'] = generated_code
        
        # Step 3: If execution failed and auto-fix is enabled, try to fix it
        retry_count = 0
        while (Config.ENABLE_AUTO_FIX and not result['success'] and 
               retry_count < self.max_retries and not result['timeout']):
            retry_count += 1
            print(f"❌ Execution failed! Attempting to fix... (Retry {retry_count}/{self.max_retries})")
            
            # Generate corrected code based on the error
            fixed_code = self._fix_code_with_error(user_request, generated_code, result['error'])
            
            if not fixed_code or fixed_code == generated_code:
                print("🤷 AI couldn't generate a fix, stopping retries")
                break
            
            if Config.SHOW_GENERATED_CODE:
                print(f"🔧 Fixed code generated:\n{fixed_code}")
            
            if Config.SHOW_EXECUTION_STEPS:
                print("🔥 Re-executing...")
            
            # Try executing the fixed code
            new_result = self._execute_code(fixed_code)
            new_result['code'] = fixed_code
            
            if new_result['success']:
                print(f"✅ Fixed successfully on retry {retry_count}!")
                return new_result
            else:
                print(f"❌ Fix attempt {retry_count} failed")
                result = new_result  # Update result for next iteration
                generated_code = fixed_code  # Update code for next iteration
        
        return result
    
    def _generate_code(self, user_request: str) -> str:
        """Generate Python code based on user request"""
        
        prompt = f"""Generate Python code to solve: "{user_request}"

CRITICAL RULES:
- Return ONLY executable Python code
- NO explanations, NO markdown, NO comments outside code
- Start directly with Python statements
- Include print() to show results
- Use standard library imports as needed
AGAIN do not generate any comments  Your output is going directly in a python file to be executed
Request: {user_request}

Python code:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            code = response['message']['content'].strip()
            
            # Aggressive cleaning to extract only code
            lines = code.split('\n')
            clean_lines = []
            in_code_block = False
            
            for line in lines:
                # Skip explanatory text before code
                if line.strip().startswith('```python'):
                    in_code_block = True
                    continue
                elif line.strip() == '```':
                    if in_code_block:
                        break  # End of code block
                    continue
                elif line.strip().startswith('```'):
                    continue
                
                # If we haven't found a code block, look for actual Python code
                if not in_code_block:
                    # Skip lines that look like explanations
                    if (line.strip().startswith(('Sure!', 'Here', 'This', 'Note:', 'Please')) or
                        'python program' in line.lower() or
                        len(line.strip()) > 100):  # Very long lines are likely explanations
                        continue
                    # If line looks like Python code, start collecting
                    if (line.strip() and 
                        (line.strip()[0].isalpha() or line.strip().startswith(('#', 'def ', 'import ', 'from ', 'print', 'for ', 'if ', 'while ')))):
                        in_code_block = True
                
                if in_code_block:
                    clean_lines.append(line)
            
            clean_code = '\n'.join(clean_lines).strip()
            
            # Final check - if still contains explanatory text, try to extract just the essentials
            if 'Sure!' in clean_code or 'Here is' in clean_code:
                # Try to find and extract just function definitions and executable statements
                final_lines = []
                for line in clean_code.split('\n'):
                    line = line.strip()
                    if (line and not line.startswith('#') and 
                        (line.startswith(('def ', 'import ', 'from ', 'print', 'for ', 'if ', 'while ', 'return')) or
                         '=' in line or line.endswith(':'))):
                        final_lines.append(line)
                clean_code = '\n'.join(final_lines)
            
            return clean_code if clean_code else ""
            
        except Exception as e:
            print(f"Error generating code: {e}")
            return ""
    
    def _fix_code_with_error(self, original_request: str, failed_code: str, error_message: str) -> str:
        """Generate corrected code based on the error"""
        
        prompt = f"""The following Python code failed to execute. Fix the error and return corrected code.

ORIGINAL REQUEST: "{original_request}"

FAILED CODE:
```python
{failed_code}
```

ERROR MESSAGE:
{error_message}

INSTRUCTIONS:
- Analyze the error and fix the issue
- Return ONLY the corrected Python code
- Include ALL necessary imports
- Keep the same functionality as requested
- NO explanations, just working code

CORRECTED CODE:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            code = response['message']['content'].strip()
            
            # Clean up the code same as before
            lines = code.split('\n')
            clean_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith('```python'):
                    in_code_block = True
                    continue
                elif line.strip() == '```':
                    if in_code_block:
                        break
                    continue
                elif line.strip().startswith('```'):
                    continue
                
                if not in_code_block:
                    if (line.strip().startswith(('Sure!', 'Here', 'This', 'Note:', 'Please')) or
                        'python program' in line.lower() or
                        len(line.strip()) > 100):
                        continue
                    if (line.strip() and 
                        (line.strip()[0].isalpha() or line.strip().startswith(('#', 'def ', 'import ', 'from ', 'print', 'for ', 'if ', 'while ')))):
                        in_code_block = True
                
                if in_code_block:
                    clean_lines.append(line)
            
            clean_code = '\n'.join(clean_lines).strip()
            
            if 'Sure!' in clean_code or 'Here is' in clean_code:
                final_lines = []
                for line in clean_code.split('\n'):
                    line = line.strip()
                    if (line and not line.startswith('#') and 
                        (line.startswith(('def ', 'import ', 'from ', 'print', 'for ', 'if ', 'while ', 'return')) or
                         '=' in line or line.endswith(':'))):
                        final_lines.append(line)
                clean_code = '\n'.join(final_lines)
            
            return clean_code if clean_code else ""
            
        except Exception as e:
            print(f"Error generating fix: {e}")
            return ""
    
    def _execute_code(self, code: str) -> dict:
        """Execute Python code safely"""
        
        # Create restricted Python code with safety measures
        safe_code = f'''
import sys

# Restrict memory usage (Windows compatible)
{f"""try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, ({Config.MEMORY_LIMIT_MB * 1024 * 1024}, {Config.MEMORY_LIMIT_MB * 1024 * 1024}))
except ImportError:
    # Windows doesn't have resource module, skip memory limiting
    pass
except:
    pass""" if Config.ENABLE_MEMORY_LIMIT else "# Memory limiting disabled"}

# Execute generated code
try:
{self._indent_code(code, 4)}
except Exception as e:
    print(f"EXECUTION_ERROR: {{e}}")
'''
        
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(safe_code)
            temp_file = f.name
        
        try:
            # Execute the code
            process = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return {
                'success': process.returncode == 0,
                'output': process.stdout,
                'error': process.stderr,
                'timeout': False
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': f'Code execution timed out after {self.timeout} seconds',
                'timeout': True
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e),
                'timeout': False
            }
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Add indentation to code"""
        lines = code.split('\n')
        indented_lines = [' ' * spaces + line for line in lines]
        return '\n'.join(indented_lines)