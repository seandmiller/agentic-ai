import ollama
import subprocess
import tempfile
import os
import sys
import hashlib
from .config import Config

class CodeExecutor:
    
    def __init__(self):
        self.model_name = Config.CODE_MODEL
        self.timeout = Config.TIMEOUT
        self.max_fix_depth = getattr(Config, 'MAX_FIX_DEPTH', 5)
        self.fix_history = []
        self.code_hashes = set()
    
    def generate_and_execute(self, user_request: str) -> dict:
        print("⚡ Generating initial code...")
        self._reset_state()
        
        initial_code = self._generate_code(user_request)
        if not initial_code:
            return self._create_error_result('Failed to generate initial code')
        
        if Config.SHOW_GENERATED_CODE:
            print(f"📝 Initial code:\n{initial_code}")
        
        return self._execute_with_recursive_fixes(user_request, initial_code, depth=0)
    
    def _execute_with_recursive_fixes(self, original_request: str, code: str, depth: int = 0) -> dict:
        indent = "  " * depth
        print(f"{indent}🔥 Executing (depth {depth})...")
        
        result = self._execute_code(code)
        result.update({'code': code, 'fix_attempts': depth})
        
        if result['success']:
            self._print_success_message(indent, depth)
            return result
        
        if not self._should_attempt_fix(depth, result):
            return self._handle_max_depth_reached(result, depth, indent)
        
        print(f"{indent}❌ Execution failed! Attempting recursive fix #{depth + 1}...")
        
        fixed_code = self._generate_recursive_fix(original_request, code, result['error'], depth)
        
        if not self._is_valid_fix(fixed_code, code, indent, depth):
            return self._handle_invalid_fix(result, depth)
        
        self._record_fix_attempt(depth, code, fixed_code, result['error'])
        
        if Config.SHOW_GENERATED_CODE:
            print(f"{indent}🔧 Fixed code (attempt {depth + 1}):\n{fixed_code}")
        
        return self._execute_with_recursive_fixes(original_request, fixed_code, depth + 1)
    
    def _generate_recursive_fix(self, original_request: str, failed_code: str, error_message: str, depth: int) -> str:
        fix_context = self._build_fix_context()
        
        prompt = f"""Fix this Python code that failed to execute. This is recursive fix attempt #{depth + 1}.

ORIGINAL REQUEST: "{original_request}"

CURRENT FAILED CODE:
```python
{failed_code}
```

ERROR MESSAGE:
{error_message}

PREVIOUS FIX ATTEMPTS:
{fix_context}

INSTRUCTIONS FOR RECURSIVE FIXING:
1. Analyze the specific error carefully
2. Learn from previous fix attempts (if any) 
3. Consider what didn't work before
4. Generate a DIFFERENT approach if previous fixes failed
5. Return ONLY the corrected Python code
6. Make sure ALL imports are included
7. Ensure the code addresses the original request

CORRECTED CODE:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return self._clean_code(response['message']['content'].strip())
        except Exception as e:
            print(f"Error generating recursive fix: {e}")
            return ""
    
    def _build_fix_context(self) -> str:
        if not self.fix_history:
            return "No previous attempts."
        
        recent_attempts = self.fix_history[-3:]
        context_parts = []
        
        for attempt in recent_attempts:
            error_preview = attempt['error'][:200] + "..." if len(attempt['error']) > 200 else attempt['error']
            context_parts.append(f"Attempt {attempt['depth'] + 1}: Error was: {error_preview}")
        
        return "\n".join(context_parts)
    
    def _is_duplicate_fix(self, new_code: str) -> bool:
        code_hash = hashlib.md5(new_code.strip().encode()).hexdigest()
        if code_hash in self.code_hashes:
            return True
        self.code_hashes.add(code_hash)
        return False
    
    def _generate_code(self, user_request: str) -> str:
        prompt = f"""Generate Python code to solve: "{user_request}"

RULES:
- Return ONLY executable Python code
- NO explanations, NO markdown, NO comments
- Start directly with Python statements
- Include print() to show results
- Use standard library imports as needed

Python code:"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return self._clean_code(response['message']['content'].strip())
        except Exception as e:
            print(f"Error generating initial code: {e}")
            return ""
    
    def _clean_code(self, raw_code: str) -> str:
        if not raw_code:
            return ""
        
        lines = raw_code.split('\n')
        clean_lines = []
        in_code_block = False
        
        code_indicators = ('import ', 'from ', 'def ', 'class ', 'print', 'if ', 'for ', 'while ', 'try')
        skip_phrases = ('Sure!', 'Here', 'This', 'Note:', 'Let me', 'I will')
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('```python'):
                in_code_block = True
                continue
            elif stripped == '```' and in_code_block:
                break
            elif stripped.startswith('```'):
                continue
            
            if not in_code_block:
                if any(stripped.startswith(phrase) for phrase in skip_phrases):
                    continue
                if 'python' in stripped.lower() and len(stripped) > 50:
                    continue
                if stripped and (stripped[0].isalpha() or stripped.startswith(code_indicators)):
                    in_code_block = True
            
            if in_code_block and line:
                clean_lines.append(line)
        
        cleaned = '\n'.join(clean_lines).strip()
        
        if any(phrase in cleaned for phrase in skip_phrases):
            essential_lines = [
                line for line in cleaned.split('\n')
                if line.strip() and not line.strip().startswith('#') and (
                    line.strip().startswith(code_indicators) or
                    '=' in line or line.endswith(':') or line.startswith('    ')
                )
            ]
            if essential_lines:
                cleaned = '\n'.join(essential_lines)
        
        return cleaned
    
    def _execute_code(self, code: str) -> dict:
        safe_code = f'''import sys
import os

try:
{self._indent_code(code, 4)}
except Exception as e:
    print(f"EXECUTION_ERROR: {{type(e).__name__}}: {{e}}")
    sys.exit(1)
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(safe_code)
            temp_file = f.name
        
        try:
            process = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.getcwd()
            )
            
            success = process.returncode == 0
            error_output = process.stderr
            
            if "EXECUTION_ERROR:" in process.stdout:
                success = False
                error_lines = [line for line in process.stdout.split('\n') if 'EXECUTION_ERROR:' in line]
                if error_lines and not error_output:
                    error_output = error_lines[0].replace('EXECUTION_ERROR: ', '')
            
            return {
                'success': success,
                'output': process.stdout,
                'error': error_output,
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
                'error': f'Subprocess execution error: {str(e)}',
                'timeout': False
            }
        finally:
            self._safe_cleanup(temp_file)
    
    def _indent_code(self, code: str, spaces: int) -> str:
        return '\n'.join(' ' * spaces + line for line in code.split('\n'))
    
    def _reset_state(self):
        self.fix_history.clear()
        self.code_hashes.clear()
    
    def _create_error_result(self, error_msg: str) -> dict:
        return {
            'success': False,
            'output': '',
            'error': error_msg,
            'code': '',
            'fix_attempts': 0
        }
    
    def _print_success_message(self, indent: str, depth: int):
        if depth > 0:
            print(f"{indent}✅ Fixed successfully after {depth} recursive fix(es)!")
        else:
            print(f"{indent}✅ Code executed successfully!")
    
    def _should_attempt_fix(self, depth: int, result: dict) -> bool:
        return depth < self.max_fix_depth and not result.get('timeout', False)
    
    def _handle_max_depth_reached(self, result: dict, depth: int, indent: str) -> dict:
        if depth >= self.max_fix_depth:
            print(f"{indent}🛑 Reached maximum fix depth ({self.max_fix_depth})")
            result['error'] += f"\n\nReached maximum recursive fix depth ({self.max_fix_depth})"
        elif result.get('timeout'):
            print(f"{indent}⏰ Code execution timed out")
        return result
    
    def _is_valid_fix(self, fixed_code: str, original_code: str, indent: str, depth: int) -> bool:
        if not fixed_code or fixed_code == original_code:
            print(f"{indent}🤷 No meaningful fix generated, stopping recursion")
            return False
        
        if self._is_duplicate_fix(fixed_code):
            print(f"{indent}🔄 Detected duplicate fix attempt, stopping recursion")
            return False
        
        return True
    
    def _handle_invalid_fix(self, result: dict, depth: int) -> dict:
        result['error'] += f"\n\nRecursive fixing stopped at depth {depth} - no improvement found"
        return result
    
    def _record_fix_attempt(self, depth: int, original_code: str, fixed_code: str, error: str):
        self.fix_history.append({
            'depth': depth,
            'original_code': original_code,
            'fixed_code': fixed_code,
            'error': error
        })
        
        if len(self.fix_history) > 10:
            self.fix_history.pop(0)
    
    def _safe_cleanup(self, temp_file: str):
        if os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except OSError:
                pass
    
    def get_fix_summary(self) -> dict:
        return {
            'total_attempts': len(self.fix_history),
            'fix_history': self.fix_history,
            'success_depth': len(self.fix_history)
        }