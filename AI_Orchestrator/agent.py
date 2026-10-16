import os
from dotenv import load_dotenv
import json
import re
from ollama import chat
from tool_implementation import ToolImplementations
from tools import *

load_dotenv()

class AIDataOrchestrator:
    def __init__(self):
        self.model = "mistral"
        self.tools_impl = ToolImplementations()
    
    def extract_first_tool_call(self, text: str) -> dict:
        """Extract ONLY the first valid tool call JSON"""
        start = text.find('{')
        if start == -1:
            return None
        
        depth = 0
        in_string = False
        escape = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    
                    if depth == 0:
                        json_str = text[start:i+1]
                        try:
                            obj = json.loads(json_str)
                            if "tool" in obj and "input" in obj:
                                return obj
                        except:
                            pass
                        return None
        
        return None
    
    def fix_tool_input(self, tool_name: str, tool_input: dict, collected_data: dict) -> dict:
        """Fix common mistakes in tool input"""
        
        # Fix parse_csv: "file" → "file_path"
        if tool_name == "parse_csv":
            if "file" in tool_input and "file_path" not in tool_input:
                tool_input["file_path"] = tool_input.pop("file")
            if "file_path" not in tool_input:
                tool_input["file_path"] = "data/sample.csv"
            if "delimiter" not in tool_input:
                tool_input["delimiter"] = ","
        
        # Fix filter_data: use actual data from collected_data
        elif tool_name == "filter_data":
            if collected_data and "rows" in collected_data:
                tool_input["data"] = collected_data["rows"]
            if "column" not in tool_input:
                tool_input["column"] = "region"
            if "operator" not in tool_input:
                tool_input["operator"] = "=="
            if "value" not in tool_input:
                tool_input["value"] = "North"
        
        # Fix calculate_statistics: use actual data
        elif tool_name == "calculate_statistics":
            if collected_data:
                if "filtered_rows" in collected_data:
                    tool_input["data"] = collected_data["filtered_rows"]
                elif "rows" in collected_data:
                    tool_input["data"] = collected_data["rows"]
            if "columns" not in tool_input:
                tool_input["columns"] = ["sales"]
            if "metrics" not in tool_input:
                tool_input["metrics"] = ["mean", "sum", "min", "max"]
        
        # Fix send_email: provide all required fields
        elif tool_name == "send_email":
            if "recipient" not in tool_input:
                tool_input["recipient"] = "user@example.com"
            if "subject" not in tool_input:
                tool_input["subject"] = "Data Summary"
            if "body" not in tool_input:
                tool_input["body"] = "Summary of data analysis"
            if "preview" not in tool_input:
                tool_input["preview"] = True
        
        return tool_input
    
    def orchestrate(self, user_request: str) -> str:
        print(f"\n{'='*60}")
        print(f"USER REQUEST: {user_request}")
        print(f"{'='*60}\n")
        
        # VERY EXPLICIT system prompt with exact JSON examples
        system_prompt = """You are a tool-calling agent. Output ONLY ONE JSON tool call per message.

CRITICAL: Use EXACT format shown below. NO explanations, NO text before/after JSON.

TOOL EXAMPLES:

1. parse_csv:
{{"tool": "parse_csv", "input": {{"file_path": "data/sample.csv", "delimiter": ","}}}}

2. filter_data:
{{"tool": "filter_data", "input": {{"data": [], "column": "region", "operator": "==", "value": "North"}}}}

3. calculate_statistics:
{{"tool": "calculate_statistics", "input": {{"data": [], "columns": ["sales"], "metrics": ["mean", "sum", "min", "max", "count", "median", "std"]}}

4. send_email:
{{"tool": "send_email", "input": {{"recipient": "user@example.com", "subject": "Report", "body": "Summary here", "preview": true}}}}

When done, respond with exactly: DONE: <summary>

Task: {user_request}
""".format(user_request=user_request)
        
        messages = [{"role": "user", "content": system_prompt}]
        max_iterations = 10
        iteration = 0
        collected_data = None
        
        while iteration < max_iterations:
            iteration += 1
            print(f"[Iteration {iteration}]")
            
            response = chat(
                model=self.model,
                messages=messages,
                stream=False
            )
            
            response_text = response['message']['content'].strip()
            
            # Check for completion
            if response_text.startswith("DONE:"):
                summary = response_text.replace("DONE:", "").strip()
                print(f"\n✅ COMPLETED\n{summary}\n")
                return summary
            
            # Extract tool call
            tool_call = self.extract_first_tool_call(response_text)
            
            if not tool_call:
                print(f"⚠️  Invalid JSON format\n")
                messages.append({"role": "assistant", "content": response_text[:50]})
                messages.append({
                    "role": "user",
                    "content": "Invalid format. Copy-paste an EXACT example from above. Do NOT modify the format."
                })
                continue
            
            tool_name = tool_call["tool"]
            tool_input = tool_call.get("input", {})
            
            # FIX common mistakes
            tool_input = self.fix_tool_input(tool_name, tool_input, collected_data)
            
            print(f"🔧 {tool_name}")
            
            # Execute tool
            try:
                result = self.execute_tool(tool_name, tool_input)
                result_dict = json.loads(result)
                
                # Check for errors
                if "error" in result_dict:
                    print(f"   ❌ {result_dict['error'][:100]}\n")
                    messages.append({"role": "assistant", "content": json.dumps(tool_call)})
                    messages.append({
                        "role": "user",
                        "content": f"Error: {result_dict['error']}. Use different input or say DONE:"
                    })
                    continue
                
                # Show success
                if tool_name == "parse_csv":
                    row_count = result_dict.get('row_count', 0)
                    columns = result_dict.get('columns', [])
                    print(f"   ✅ Parsed {row_count} rows | Columns: {columns}\n")
                    collected_data = result_dict
                
                elif tool_name == "filter_data":
                    matched = result_dict.get('matched_count', 0)
                    total = result_dict.get('original_count', 0)
                    print(f"   ✅ Filtered: {matched}/{total} rows matched\n")
                    collected_data = result_dict
                
                elif tool_name == "calculate_statistics":
                    stats = result_dict.get("statistics", {})
                    print(f"   ✅ Statistics:\n")
                    for col, metrics in stats.items():
                        for metric, value in metrics.items():
                            if isinstance(value, float):
                                print(f"      {col} {metric}: {value:.2f}")
                            else:
                                print(f"      {col} {metric}: {value}")
                    print()
                
                elif tool_name == "send_email":
                    print(f"   ✅ {result_dict.get('message', 'Email sent')}\n")
                
                # Add to conversation
                messages.append({"role": "assistant", "content": json.dumps(tool_call)})
                messages.append({
                    "role": "user",
                    "content": f"Success! Result: {json.dumps(result_dict)}\n\nContinue with next step or respond with: DONE: <summary>"
                })
            
            except Exception as e:
                error_msg = str(e)[:100]
                print(f"   ❌ {error_msg}\n")
                messages.append({"role": "assistant", "content": json.dumps(tool_call)})
                messages.append({
                    "role": "user",
                    "content": f"Error: {error_msg}. Say DONE: or continue"
                })
        
        return "Max iterations reached"
    
    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute tool"""
        try:
            if tool_name == "parse_csv":
                result = self.tools_impl.parse_csv(CSVParseRequest(**tool_input))
            elif tool_name == "filter_data":
                result = self.tools_impl.filter_data(DataFilterRequest(**tool_input))
            elif tool_name == "calculate_statistics":
                result = self.tools_impl.calculate_statistics(StatisticsRequest(**tool_input))
            elif tool_name == "send_email":
                result = self.tools_impl.send_email(EmailRequest(**tool_input))
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
            
            return result.model_dump_json()
        except Exception as e:
            return json.dumps({"error": str(e)})