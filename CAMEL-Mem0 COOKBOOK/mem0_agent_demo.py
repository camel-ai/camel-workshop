import os
import sys

from rich import print as rprint

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType

from mem0_tools import Mem0CloudToolkit


def main():
    """Main function to run the Mem0 cloud agent demo."""
    if not os.getenv("MEM0_API_KEY"):
        rprint("[bold red]ERROR: MEM0_API_KEY environment variable not set.[/bold red]")
        sys.exit(1)
    if not os.getenv("GOOGLE_API_KEY"):
        rprint("[bold red]ERROR: GOOGLE_API_KEY environment variable not set.[/bold red]")
        sys.exit(1)

    # Use predefined IDs for easier testing
    agent_id = "demo-agent"
    user_id = "demo-user"
    rprint(f"[dim]Using agent_id: {agent_id}, user_id: {user_id}[/dim]")

    # Initialize the Mem0 cloud toolkit
    toolkit = Mem0CloudToolkit(agent_id=agent_id, user_id=user_id)
    tools = toolkit.get_tools()

    model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        model_config_dict={"temperature": 0.0, "max_tokens": 4096},
    )

    agent_name = "Memory Master"
    system_message_content = (
        "You are a helpful memory assistant that manages memories using Mem0 cloud storage. "
        "When you use a tool to help the user, always explain what you did in a conversational way. "
        "For example, if you add a memory, say something like 'I've stored that information for you!' "
        "If you retrieve memories, say 'Here's what I found in your memories:' followed by the results. "
        "If you search, say 'I searched your memories and found:' followed by the results. "
        "Always be friendly and helpful, and explain what memory operation you performed."
    )
    sys_msg = BaseMessage.make_assistant_message(
        role_name=agent_name,
        content=system_message_content,
    )

    agent = ChatAgent(sys_msg, model=model, tools=tools)

    rprint(
        "[bold green]🧠 Mem0 Cloud Agent is ready! How can I help you with your"
        " memories?[/bold green]"
    )

    while True:
        user_query = input("> ")
        if user_query.lower() in ["exit", "quit"]:
            break

        user_msg = BaseMessage.make_user_message(
            role_name="User", content=user_query
        )
        response = agent.step(user_msg)

        # Show the agent's conversational response first
        rprint(f"[green]🤖 Agent:[/green] {response.msg.content}")
        
        # Then show the raw tool outputs for debugging
        if response.info.get('tool_calls'):
            rprint("\n[dim]--- Raw Tool Output (for debugging) ---[/dim]")
            for tool_call in response.info['tool_calls']:
                rprint(f"[yellow]Tool '{tool_call.tool_name}':[/yellow]")
                rprint(f"[cyan]Raw Result:[/cyan] {tool_call.result}")
            rprint("[dim]--- End Raw Output ---[/dim]\n")


if __name__ == "__main__":
    main()
