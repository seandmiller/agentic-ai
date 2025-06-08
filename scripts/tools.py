import json
import re
from typing import Any, Optional, List, Dict

class JSONExtractor:
    """Tool to extract and clean JSON from messy LLM outputs"""
    
    @staticmethod
    def extract_json(raw_text: str) -> Optional[Any]:
        """
        Extract JSON from LLM output that may contain:
        - Markdown code blocks
        - Extra explanatory text
        - Malformed JSON
        - Various quote types
        """
        if not raw_text or not raw_text.strip():
            return None
        
        # Try multiple extraction strategies
        strategies = [
            JSONExtractor._extract_from_code_blocks,
            JSONExtractor._extract_from_brackets,
            JSONExtractor._extract_and_fix_json,
            JSONExtractor._parse_direct
        ]
        
        for strategy in strategies:
            try:
                result = strategy(raw_text)
                if result is not None:
                    return result
            except:
                continue
        
        return None
    
    @staticmethod
    def _extract_from_code_blocks(text: str) -> Optional[Any]:
        """Extract JSON from ```json or ``` code blocks"""
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'`(.*?)`'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                cleaned = match.strip()
                if cleaned and (cleaned.startswith('{') or cleaned.startswith('[')):
                    try:
                        return json.loads(cleaned)
                    except:
                        continue
        return None
    
    @staticmethod
    def _extract_from_brackets(text: str) -> Optional[Any]:
        """Extract JSON by finding bracket pairs"""
        # Find JSON objects
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start_idx = text.find(start_char)
            if start_idx == -1:
                continue
            
            bracket_count = 0
            end_idx = start_idx
            
            for i in range(start_idx, len(text)):
                if text[i] == start_char:
                    bracket_count += 1
                elif text[i] == end_char:
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i
                        break
            
            if bracket_count == 0:
                json_str = text[start_idx:end_idx + 1]
                try:
                    return json.loads(json_str)
                except:
                    continue
        
        return None
    
    @staticmethod
    def _extract_and_fix_json(text: str) -> Optional[Any]:
        """Extract and fix common JSON formatting issues"""
        # Remove common prefixes
        prefixes_to_remove = [
            r'^.*?(?:here\s+is|here\'s|the\s+json|json\s+output|result).*?:?\s*',
            r'^.*?```json\s*',
            r'^.*?```\s*',
            r'^[^{\[]*'
        ]
        
        cleaned = text.strip()
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
            cleaned = cleaned.strip()
            if cleaned.startswith(('{', '[')):
                break
        
        # Remove common suffixes
        suffixes_to_remove = [
            r'\s*```.*$',
            r'\s*[^}\]]*$'
        ]
        
        for suffix in suffixes_to_remove:
            cleaned = re.sub(suffix, '', cleaned, flags=re.MULTILINE)
        
        # Fix common JSON issues
        fixes = [
            (r',\s*}', '}'),  # Remove trailing commas before }
            (r',\s*]', ']'),  # Remove trailing commas before ]
            (r'(\w+):', r'"\1":'),  # Quote unquoted keys
            (r"'([^']*)':", r'"\1":'),  # Convert single quotes to double
            (r":\s*'([^']*)'", r': "\1"'),  # Convert single quoted values
        ]
        
        for pattern, replacement in fixes:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        try:
            return json.loads(cleaned)
        except:
            return None
    
    @staticmethod
    def _parse_direct(text: str) -> Optional[Any]:
        """Try to parse text directly as JSON"""
        try:
            return json.loads(text.strip())
        except:
            return None

class CodeExtractor:
    """Tool to extract clean code from LLM outputs"""
    
    @staticmethod
    def extract_code(raw_text: str) -> Optional[str]:
        """
        Extract Python code from LLM output that may contain:
        - Markdown code blocks
        - Explanatory text
        - Multiple code snippets
        """
        if not raw_text or not raw_text.strip():
            return None
        
        strategies = [
            CodeExtractor._extract_from_python_blocks,
            CodeExtractor._extract_from_code_blocks,
            CodeExtractor._extract_code_lines,
            CodeExtractor._return_as_is
        ]
        
        for strategy in strategies:
            result = strategy(raw_text)
            if result and result.strip():
                return result.strip()
        
        return None
    
    @staticmethod
    def _extract_from_python_blocks(text: str) -> Optional[str]:
        """Extract from ```python blocks"""
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        return None
    
    @staticmethod
    def _extract_from_code_blocks(text: str) -> Optional[str]:
        """Extract from ``` blocks"""
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            cleaned = match.strip()
            if CodeExtractor._looks_like_python(cleaned):
                return cleaned
        
        return None
    
    @staticmethod
    def _extract_code_lines(text: str) -> Optional[str]:
        """Extract lines that look like Python code"""
        lines = text.split('\n')
        code_lines = []
        
        for line in lines:
            stripped = line.strip()
            if CodeExtractor._looks_like_python_line(stripped):
                code_lines.append(line)
            elif stripped.startswith(('Sure', 'Here', 'Let me', 'This')):
                continue  # Skip explanation lines
            elif code_lines and not stripped:
                code_lines.append(line)  # Keep empty lines within code
        
        if code_lines:
            return '\n'.join(code_lines)
        return None
    
    @staticmethod
    def _return_as_is(text: str) -> str:
        """Return text as-is if it looks like code"""
        if CodeExtractor._looks_like_python(text):
            return text
        return ""
    
    @staticmethod
    def _looks_like_python(text: str) -> bool:
        """Check if text looks like Python code"""
        python_indicators = [
            'import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ',
            'try:', 'except:', 'with ', 'print(', '=', 'return '
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in python_indicators)
    
    @staticmethod
    def _looks_like_python_line(line: str) -> bool:
        """Check if a single line looks like Python"""
        if not line.strip():
            return False
        
        # Skip obvious explanation lines
        explanation_starts = [
            'sure', 'here', 'let me', 'this', 'the code', 'now', 'first', 'next'
        ]
        
        line_lower = line.lower().strip()
        if any(line_lower.startswith(start) for start in explanation_starts):
            return False
        
        # Check for Python patterns
        python_patterns = [
            r'^\s*(import|from)\s+',
            r'^\s*(def|class|if|for|while|try|with)\s+',
            r'^\s*\w+\s*=',
            r'^\s*(print|return|break|continue)',
            r'^\s*#',  # Comments
            r'^\s*$'   # Empty lines
        ]
        
        return any(re.match(pattern, line) for pattern in python_patterns)

class OutputCleaner:
    """General purpose LLM output cleaner"""
    
    @staticmethod
    def clean_variable_output(text: str) -> str:
        """Clean variable output from LLM execution results"""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip error messages and warnings
            if any(keyword in line.lower() for keyword in 
                  ['error', 'warning', 'traceback', 'exception']):
                continue
            
            # Clean up the line
            cleaned = line.strip()
            if cleaned:
                cleaned_lines.append(cleaned)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def extract_result_variables(text: str) -> Dict[str, str]:
        """Extract RESULT_VAR outputs from code execution"""
        variables = {}
        
        if not text:
            return variables
        
        lines = text.split('\n')
        
        for line in lines:
            if 'RESULT:' in line and '=' in line:
                try:
                    # Extract: "RESULT: var_name = value"
                    parts = line.split('RESULT:', 1)[1].strip()
                    if '=' in parts:
                        var_name = parts.split('=', 1)[0].strip()
                        var_value = parts.split('=', 1)[1].strip()
                        
                        # Clean the variable name and value
                        var_name = var_name.replace('"', '').replace("'", '')
                        var_value = var_value.replace('"', '').replace("'", '')
                        
                        variables[var_name] = var_value
                except:
                    continue
        
        return variables