# Mem0 Cloud Memory Toolkit with CAMEL AI

**Persistent Memory Management for AI Agents using Mem0 Cloud Storage**

You can also check this cookbook in Colab. [CAMEL Homepage](https://www.camel-ai.org/) | [Join Discord](https://discord.gg/CNcNpquyDc) | ⭐ [Star us on GitHub](https://github.com/camel-ai/camel)

This notebook demonstrates how to build an AI agent with persistent memory capabilities using CAMEL AI and Mem0's cloud storage. In this notebook, you'll explore:

- **CAMEL AI**: A powerful multi-agent framework enabling sophisticated AI-driven tasks with memory management
- **Mem0 Cloud**: Cloud-based persistent memory storage with semantic search capabilities  
- **Memory Toolkit**: A custom toolkit for adding, retrieving, searching, and managing memories
- **Interactive Agent**: A conversational agent that remembers past interactions across sessions

This setup provides a practical foundation for building AI agents that can maintain context and learn from previous conversations.

## 📦 Installation

First, install the required packages:

```bash
!pip install "camel-ai>=0.2.16"
!pip install "mem0ai"
!pip install "rich"
```

## 📥 Download the Toolkit

You'll need the custom Mem0 toolkit. Download or create the `mem0_tools.py` file:

```python
# Download the toolkit file (replace with your actual download method)
# For this demo, we assume you have the mem0_tools.py file in your working directory

# Verify the file exists
import os
if os.path.exists('mem0_tools.py'):
    print("✅ Mem0 toolkit found!")
else:
    print("❌ Please ensure mem0_tools.py is in your working directory")
```

The toolkit provides these key capabilities:
- **Add Memory**: Store information with optional metadata
- **Retrieve Memories**: Get all stored memories 
- **Search Memories**: Semantic search with vector matching
- **Delete Memories**: Clear stored memories

## 🔑 Setting Up API Keys

You'll need API keys for both Mem0 and your chosen LLM provider (Gemini in this example):

```python
import os
from getpass import getpass

# Set up Mem0 API key
mem0_api_key = getpass('Enter your Mem0 API key: ')
os.environ["MEM0_API_KEY"] = mem0_api_key

# Set up Gemini API key
gemini_api_key = getpass('Enter your Gemini API key: ')
os.environ["GEMINI_API_KEY"] = gemini_api_key
```

You can obtain:
- **Mem0 API Key**: Sign up at [mem0.ai](https://mem0.ai)
- **Gemini API Key**: Get it from [Google AI Studio](https://aistudio.google.com/app/apikey)

Alternatively, if running on Colab, you can save your API keys as Colab Secrets:

```python
# import os
# from google.colab import userdata

# os.environ["MEM0_API_KEY"] = userdata.get("MEM0_API_KEY")
# os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
```

## 🛠️ Import the Mem0 Cloud Toolkit

Now let's import our custom toolkit that interfaces with Mem0's cloud storage:

```python
# Import the custom Mem0 toolkit
from mem0_tools import Mem0CloudToolkit

# The toolkit provides 4 main methods:
# - add_memory(content, metadata=None): Store new information
# - retrieve_memories(): Get all stored memories  
# - search_memories(query): Semantic search for relevant memories
# - delete_memories(): Clear all stored memories

print("✅ Mem0CloudToolkit imported successfully!")
```

## 🤖 Building the Memory-Enabled Agent

Now let's create an interactive agent that uses our Mem0 toolkit:

```python
import os
from rich import print as rprint
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType

# Set up agent and user IDs
agent_id = "demo-agent"
user_id = "demo-user"
rprint(f"[dim]Using agent_id: {agent_id}, user_id: {user_id}[/dim]")

# Initialize the Mem0 toolkit and get tools
toolkit = Mem0CloudToolkit(agent_id=agent_id, user_id=user_id)
tools = toolkit.get_tools()

# Create the language model
model = ModelFactory.create(
    model_platform=ModelPlatformType.GEMINI,
    model_type="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    model_config_dict={"temperature": 0.0, "max_tokens": 4096},
)

# Configure the agent system message
system_message = (
    "You are a helpful memory assistant that manages memories using Mem0 cloud storage. "
    "When you use a tool, always explain what you did conversationally. "
    "For example: 'I've stored that information!' or 'Here's what I found in your memories:' "
    "Always be friendly and explain the memory operation you performed."
)

sys_msg = BaseMessage.make_assistant_message(
    role_name="Memory Master",
    content=system_message,
)

# Create the agent
agent = ChatAgent(sys_msg, model=model, tools=tools)

rprint("[bold green]🧠 Mem0 Cloud Agent is ready![/bold green]")
```

## 💬 Quick Demo

Let's test our memory-enabled agent with some example interactions:

```python
# Example conversations to demonstrate memory capabilities
demo_messages = [
    "Hi! My name is Alice and I'm a software engineer working on AI projects.",
    "I love Python programming and have been learning about machine learning recently.",
    "What do you remember about me?",
    "Can you search for information about my profession?",
]

for user_input in demo_messages:
    print(f"\n> {user_input}")
    
    user_msg = BaseMessage.make_user_message(role_name="User", content=user_input)
    response = agent.step(user_msg)
    
    # Show the agent's response
    rprint(f"[green]🤖 Agent:[/green] {response.msg.content}")
```

## 🎯 Interactive Mode

For a full interactive experience with your memory-enabled agent:

```python
# Start interactive chat mode
rprint("[bold green]🧠 Mem0 Cloud Agent is ready![/bold green]")
rprint("[dim]Type 'exit' or 'quit' to end the conversation.[/dim]\n")

while True:
    try:
        user_query = input("> ")
        if user_query.lower() in ["exit", "quit"]:
            rprint("[yellow]Goodbye! Your memories are safely stored in Mem0 cloud.[/yellow]")
            break

        user_msg = BaseMessage.make_user_message(role_name="User", content=user_query)
        response = agent.step(user_msg)
        
        # Show the agent's response
        rprint(f"[green]🤖 Agent:[/green] {response.msg.content}")
        
    except KeyboardInterrupt:
        rprint("\n[yellow]Goodbye![/yellow]")
        break
```

## 🌟 Key Features

This memory-enabled agent demonstrates several powerful capabilities:

### 🧠 **Persistent Memory**
- Memories are stored in Mem0's cloud infrastructure
- Information persists across different chat sessions
- Automatic memory organization by agent and user IDs

### 🔍 **Semantic Search**
- Find relevant memories using natural language queries
- Vector-based semantic matching for better results
- Advanced filtering and reranking for improved accuracy

### 🛠️ **Memory Management Tools**
- **Add Memory**: Store new information with optional metadata
- **Retrieve Memories**: Get all stored memories for the user/agent
- **Search Memories**: Find specific information using semantic search
- **Delete Memories**: Clear all memories when needed

### 🎯 **Example Use Cases**
- **Personal Assistant**: Remember user preferences, appointments, and important information
- **Customer Support**: Maintain context across multiple support sessions
- **Learning Companion**: Track learning progress and past topics discussed
- **Project Management**: Remember project details, deadlines, and team information

## 🎮 Try It Yourself!

Here are some example prompts to test with your agent:

```python
# Example interactions you can try:
example_prompts = [
    "Remember that I prefer meetings in the morning",
    "My favorite programming language is Python",
    "I'm working on a project called 'SmartHome Assistant'",
    "What do you remember about my work preferences?",
    "Search for information about programming languages",
    "Tell me about my current projects",
]

for prompt in example_prompts:
    print(f"Try: {prompt}")
```

## 🌟 Highlights

This notebook has guided you through building a sophisticated AI agent with persistent memory capabilities. Key technologies utilized include:

- **CAMEL AI**: A powerful multi-agent framework enabling memory-enhanced AI interactions
- **Mem0 Cloud**: Cloud-based vector storage with semantic search capabilities
- **Custom Toolkits**: Extensible toolkit architecture for adding new capabilities
- **Rich UI**: Beautiful terminal output with the Rich library

That's everything! Got questions about 🐫 CAMEL-AI? [Join us on Discord!](https://discord.gg/CNcNpquyDc) Whether you want to share feedback, explore the latest in multi-agent systems, get support, or connect with others on exciting projects, we'd love to have you in the community! 🤝 

Check out some of our other work:
- 🐫 [Creating Your First CAMEL Agent](https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim) (free Colab)
- [Graph RAG Cookbook](https://colab.research.google.com/drive/1LhI-7a5JUhLXxAfNHzSfvV8uOPl-FLhp) (free Colab)
- 🧑‍⚖️ [Create A Hackathon Judge Committee with Workforce](https://colab.research.google.com/drive/1XpWMxJZ7Kv6RCLaE9RjZH8jdQsyPLzEJ) (free Colab)
- 🔥 [3 Ways to Ingest Data from Websites with Firecrawl & CAMEL](https://colab.research.google.com/drive/1eYTlHkgj9bGbI1d0vYpK0vMSdIlJNRq3) (free Colab)

Thanks from everyone at 🐫 CAMEL-AI!

[CAMEL Homepage](https://www.camel-ai.org/) | [Join Discord](https://discord.gg/CNcNpquyDc) | ⭐ [Star us on GitHub](https://github.com/camel-ai/camel)