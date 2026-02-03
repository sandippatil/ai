import asyncio
import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

def getLlm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    return llm


async def run_agent():
    atlassian_mcp_token = os.environ.get("ATLASSIAN_MCP_TOKEN")
    atlassian_mcp_token = "ATATT3xFfGF02x_PIKUawEao13UA3D7AO40dqvrC3rIVJsdaoX_8th7p_i-o1hMT7ctRDsahshPkbpAEPTvENmCvbDhyxvMA7Y4vnHCz6sg2UAKr_Q6lyQf9j7tLP91Q7UaLmv6Hdjdy5SibvDxmhXnfThoEcN-hT-xGGwusdQKYmRNDuXQmL0c=38B21175"
    '''
    if not atlassian_mcp_token:
        raise RuntimeError(
            "Missing ATLASSIAN_MCP_TOKEN environment variable required to connect to Atlassian MCP."
        )
    '''
    # Connect to MCP server
    mcp_client = MultiServerMCPClient(
        {
            "atlassian": {
                "transport": "http",
                "url": "https://mcp.atlassian.com/v1/mcp",
                "headers": {"Authorization": f"Bearer {atlassian_mcp_token}"},
            }
        }
    )

    mcp_tools = await mcp_client.get_tools()

    llm = getLlm()

    # Create LangChain agent with MCP tools
    agent = create_agent(llm, tools=mcp_tools)

    # Ask something that uses the Atlassian MCP tools
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Get the 5 most recent Jira issues from DEMO project"),
            ]
        }
    )
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if messages:
        print(messages[-1].content)
    else:
        print(result)

if __name__ == "__main__":
    asyncio.run(run_agent())

'''

app = FastAPI()
class AgentRequest(BaseModel):
    question: str

@app.post("/agent")
def agent(req: AgentRequest):
    return run_agent(req.question)

'''