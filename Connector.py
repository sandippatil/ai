import asyncio
import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
import base64
import httpx
from dotenv import load_dotenv
import json

load_dotenv()

def getLlm():
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    return llm

def get_system_prompt():
    return SystemMessage(
        content="""
You are a Jira assistant.

IMPORTANT RULES:
- The user is already authenticated via Jira API token.
- NEVER ask for email, username, or accountId.
- To find the user's issues, ALWAYS use:
  assignee = currentUser()
"""
    )

def ask_human_approval(tool_name, tool_args):
    """Ask for human approval before executing MCP server calls"""
    print(f"\n{'='*60}")
    print(f"🔧 MCP Tool Call Request:")
    print(f"Tool: {tool_name}")
    print(f"Arguments: {json.dumps(tool_args, indent=2)}")
    print(f"{'='*60}")
    
    while True:
        response = input("Do you want to ALLOW or REJECT this call? (allow/reject): ").strip().lower()
        if response in ['allow', 'a', 'yes', 'y']:
            print("✅ Call ALLOWED")
            return True
        elif response in ['reject', 'r', 'no', 'n']:
            print("❌ Call REJECTED")
            return False
        else:
            print("Please enter 'allow' or 'reject'")

class HumanInTheLoopMCPClient:
    """MCP Client wrapper that asks for human approval before tool calls"""
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.tools = []
    
    async def get_tools(self):
        original_tools = await self.mcp_client.get_tools()
        self.tools = []
        
        for tool in original_tools:
            # Wrap each tool to require human approval
            wrapped_tool = self._wrap_tool(tool)
            self.tools.append(wrapped_tool)
        
        return self.tools
    
    def _wrap_tool(self, original_tool):
        """Wrap a tool to require human approval before execution"""
        async def wrapped_tool(*args, **kwargs):
            # Extract tool name and arguments for approval
            tool_name = original_tool.name
            
            # Handle different calling patterns
            if args and not kwargs:
                # Single argument case
                tool_args = args[0] if isinstance(args[0], dict) else {"input": str(args[0])}
                input_to_pass = args[0]
            elif kwargs and not args:
                # Keyword arguments case
                tool_args = kwargs
                input_to_pass = kwargs
            elif args and kwargs:
                # Mixed case - combine them
                tool_args = {"args": str(args), **kwargs}
                input_to_pass = kwargs if kwargs else args[0]
            else:
                # No arguments
                tool_args = {}
                input_to_pass = {}
            
            # Ask for human approval
            if ask_human_approval(tool_name, tool_args):
                return await original_tool.ainvoke(input_to_pass)
            else:
                return "Tool call was rejected by user."
        
        # Create a new tool-like object
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel
        import inspect
        
        # Get the original tool's schema
        if hasattr(original_tool, 'args_schema') and original_tool.args_schema:
            args_schema = original_tool.args_schema
        else:
            # Create a simple schema if none exists
            args_schema = BaseModel
        
        wrapped_tool_obj = StructuredTool(
            name=original_tool.name,
            description=original_tool.description,
            func=wrapped_tool,
            args_schema=args_schema,
            coroutine=wrapped_tool
        )
        
        return wrapped_tool_obj

async def run_agent():
    # Connect to MCP server
    mcp_client = MultiServerMCPClient(
            {
                "atlassian": {
                    "transport": "stdio",
                    "command": "docker",
                    "args": [
                        "run", "--rm", "-i",
                        "-e", "JIRA_URL=https://sandip-vpatil.atlassian.net",
                        "-e", "JIRA_USERNAME=sandippatil8797@gmail.com",
                        "-e", f"JIRA_API_TOKEN={os.environ['JIRA_API_TOKEN']}",
                        "ghcr.io/sooperset/mcp-atlassian:latest"
                    ]
                }
            }
        )

    # Wrap with human-in-the-loop
    human_loop_client = HumanInTheLoopMCPClient(mcp_client)
    mcp_tools = await human_loop_client.get_tools()

    llm = getLlm()
    system_prompt = get_system_prompt()

    # Create LangChain agent with MCP tools
    agent = create_agent(llm, tools=mcp_tools)

    print("🤖 Jira Assistant is ready! Type 'quit' or 'exit' to stop.")
    print("🔧 All MCP server calls will require your approval.")
    print("-" * 60)
    
    # Continuous communication loop
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            # Check for exit conditions
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Invoke agent with system prompt and user message
            result = await agent.ainvoke(
                {
                    "messages": [
                        system_prompt,
                        HumanMessage(content=user_input),
                    ]
                }
            )
            
            # Display the response
            messages = result.get("messages", []) if isinstance(result, dict) else []
            if messages:
                print(f"\n🤖 Assistant: {messages[-1].content}")
            else:
                print(f"\n🤖 Assistant: {result}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")
    
    # Cleanup
    await mcp_client.close()

if __name__ == "__main__":
    asyncio.run(run_agent())
