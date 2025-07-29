---
title: CAMEL Cookbook - Object Detection with ACI.dev MCP Tools
description: Learn how to build an object detection agent using CAMEL AI and ACI.dev's MCP protocol for seamless ML tasks.
---
---

You can also check this cookbook in [Google Colab]().  
[CAMEL Homepage](https://www.camel-ai.org/) | [Join Discord](https://discord.gg/EXAMPLE)

⭐ Star us on [GitHub](https://github.com/camel-ai/camel), join our [Discord](https://discord.gg/EXAMPLE), or follow us on [X](https://x.com/camelaiorg)

This cookbook shows how to build a powerful object detection agent using CAMEL AI connected to ACI.dev's MCP tools. We'll create an agent that analyzes images, detects objects like cars or trees, and explains results in natural language—all without writing complex ML code.

**Key Learnings:**
- Why agents need tools to be truly useful.
- How MCP enables dynamic, aware tool usage for tasks like object detection.
- Setting up CAMEL with ACI.dev for real-time image analysis.
- Building and running your own object detection agent.
- Handling outputs with summaries, tables, and visualized results.

This setup uses CAMEL's `MCPToolkit` to connect to ACI.dev's MCP servers, powering object detection via Replicate's ML models.

## 📦 Installation

Install the required packages:
```
pip install "camel-ai[all]==0.2.62" python-dotenv rich
```

## 🔑 Setting Up API Keys

You'll need:
- **ACI.dev API Key**: Sign up at [ACI.dev](https://aci.dev) and grab it from Project Settings.
- **Google Gemini API Key**: Get it from [Google's API Console](https://console.developers.google.com/).
- **Linked Account Owner ID**: Provided when you connect apps in ACI.dev (e.g., Replicate for object detection).

Create a `.env` file:
```
ACI_API_KEY=your_aci_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
LINKED_ACCOUNT_OWNER_ID=your_account_id_here
```

## 🚀 Introduction

This cookbook demonstrates how to build a powerful **object detection agent** that transforms simple text requests into sophisticated image analysis. Instead of writing complex ML code, we'll use **CAMEL AI** connected to **ACI.dev's MCP tools** to create an agent that can analyze images, detect objects, and explain results—all through natural conversation.

### Why This Matters

**Traditional AI Limitations:**
- **Just Chatbots**: Without tools, AI agents are limited to text-based responses
- **No Real Actions**: They can describe object detection but can't actually analyze images  
- **Static Integration**: Manual coding required for each new capability
- **Technical Barriers**: Complex ML setup prevents non-technical users from accessing AI power

**Our Solution with MCP + ACI.dev:**
- **Dynamic Tool Discovery**: Agents automatically find and use the right tools
- **Natural Language Control**: Ask "Find cars in this image" and get structured results
- **Seamless Integration**: 600+ tools through a single MCP connection
- **Democratized AI**: Anyone can perform advanced ML tasks via chat

### How MCP Transforms AI Agents

**MCP (Model Context Protocol)** acts as a universal bridge between AI agents and external tools:

- **Client-Server Architecture**: Your CAMEL agent (client) connects to ACI.dev's MCP server
- **Intelligent Tool Selection**: Agents discover available tools and pick the right one automatically  
- **Real-Time Execution**: From "Find trees in this photo" to structured detection results in seconds
- **Adaptive Behavior**: No hardcoded logic—agents adapt to new tasks dynamically

### What We're Building

**Core Components:**
- **`create_config.py`**: Sets up MCP connection to ACI.dev servers
- **`object_agent.py`**: Main agent with Gemini 2.5 Flash reasoning and natural language interface
- **Auto-generated `config.json`**: Configuration linking to Replicate's ML models via ACI.dev

**Agent Capabilities:**
- **Image Analysis**: Process any image URL for object detection
- **Smart Detection**: Identify cars, trees, people, and hundreds of other objects
- **Structured Output**: Natural language summary + markdown tables + bounding box visualizations
- **Conversational Interface**: Simple CLI that feels like chatting with an AI expert

## 🛠️ Building the Object Detection Agent

We'll build this in two parts: a configuration script and the main agent. Let's walk through each component and understand how they work together.

### Step 1: MCP Configuration (`create_config.py`)

```python
import os
import json
from dotenv import load_dotenv

def create_config():
    """Create MCP config with proper environment variable substitution"""
    load_dotenv()

    aci_api_key = os.getenv("ACI_API_KEY")
    if not aci_api_key:
        raise ValueError("ACI_API_KEY environment variable is required")

    config = {
        "mcpServers": {
            "aci_apps": {
                "command": "aci-mcp",
                "args": [
                    "apps-server",
                    "--apps=REPLICATE",
                    "--linked-account-owner-id",
                    "your_linked_account_owner_id_here",  # Replace with yours
                ],
                "env": {"ACI_API_KEY": aci_api_key},
            }
        }
    }

    # Write to config.json
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✓ Config created successfully")
    return config

if __name__ == "__main__":
    create_config()
```

**How it works:** This script creates the bridge between your CAMEL agent and ACI.dev's MCP server. 
1. It loads your API key from the `.env` file
2. Configures the MCP server to use Replicate's ML models for object detection 
3. Generates a `config.json` file that CAMEL's MCPToolkit reads to establish the connection. 

Run this once, and your agent will have access to powerful object detection capabilities through ACI.dev's unified server.

### Step 2: The Main Agent (`object_agent.py`)

Now let's build the intelligent agent that can understand natural language and perform object detection:

#### **Imports and Environment Setup**

```python
import asyncio
import os
from dotenv import load_dotenv
from rich import print as rprint

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.toolkits import MCPToolkit
from camel.types import ModelPlatformType
from create_config import create_config

load_dotenv()
```

**Essential Imports:**
- **`asyncio`**: Enables asynchronous programming for MCP connections
- **`rich`**: Beautiful terminal output with colors and formatting
- **CAMEL Components**: Core framework for building AI agents
- **`create_config`**: Our configuration script for MCP setup

#### **Agent Initialization and Connection**

```python
async def main():
    """
    Main function to run the Object Detection Agent.
    """
    create_config()
    mcp_toolkit = MCPToolkit(config_path="config.json")
    await mcp_toolkit.connect()
```

**MCP Connection Setup:**
- **`create_config()`**: Generates the MCP configuration file
- **`MCPToolkit`**: CAMEL's interface to MCP servers
- **`await mcp_toolkit.connect()`**: Establishes async connection to ACI.dev

#### **Agent Persona and System Prompt**

```python
    agent_name = "ObjectDetectionAgent"
    system_message_content = (
        "You are a specialized Object Detection Agent. Your primary function is to use the `REPLICATE.run` tool for object detection and present the findings in a user-friendly format. "
        "The user will provide a text prompt containing an image URL and a query. You must extract the `image` URL and the `query` object(s). "
        "Immediately call the `REPLICATE.run` tool. The `input` must be a dictionary with two keys: `image` (the URL) and `query` (a string of the object(s)). "
        "Do not ask for clarification; make a reasonable inference if the query is ambiguous. "
        "After receiving the tool's output, format your response as follows: "
        "- **Natural Language Summary:** Start with a detailed friendly, insightful analysis of the detection results in plain English. "
        "- **Markdown Table:** Create a markdown table with columns: 'Object', 'Confidence Score', and 'Bounding Box Coordinates'. "
        "- **Result Image:** If the tool provides a URL for an image with bounding boxes, display it using markdown: `![Detected Objects](URL_HERE)`. "
        "Whenever I give you a link, trigger the tool call, extract its outputs and links, and present me in a proper markdown format with detailed analysis from the tool call in natural language."
    )
```

**Intelligent Agent Behavior:**
- **Autonomous Tool Usage**: Agent automatically calls `REPLICATE.run` when given image URLs
- **Smart Input Parsing**: Extracts image URLs and detection queries from natural language
- **Structured Output**: Produces consistent, formatted responses with summaries, tables, and visualizations
- **No Confirmation Needed**: Makes reasonable inferences instead of asking for clarification

#### **Model and Tool Configuration**

```python
    tools = mcp_toolkit.get_tools()

    model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        model_config_dict={"temperature": 0.0, "max_tokens": 4096},
    )

    sys_msg = BaseMessage.make_assistant_message(
        role_name=agent_name,
        content=system_message_content,
    )

    agent = ChatAgent(model=model, system_message=sys_msg, tools=tools, memory=None)
```

**Agent Configuration:**
- **Tool Discovery**: `get_tools()` automatically loads all available MCP tools
- **Gemini 2.5 Flash**: Fast, capable model for reasoning and tool orchestration
- **Zero Temperature**: Deterministic outputs for consistent behavior
- **High Token Limit**: 4096 tokens for detailed analysis and formatting
- **Stateless Memory**: Fresh context for each interaction

#### **Interactive CLI Loop**

```python
    rprint(
        f"[bold green]Started {agent_name}. Enter your request or type 'exit' to quit.[/bold green]"
    )
    rprint("[dim]Example: 'Find cars in this image: https://example.com/image.jpg'[/dim]")
    
    try:
        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["exit", "quit", "bye"]:
                rprint("[bold yellow]Exiting...[/bold yellow]")
                break

            if not user_input:
                continue

            user_message = BaseMessage.make_user_message(
                role_name="User", content=user_input
            )

            response = await agent.astep(user_message)

            if response and hasattr(response, "msgs") and response.msgs:
                for msg in response.msgs:
                    rprint(f"[bold blue]{agent_name}:[/bold blue]\n{msg.content}")

    except KeyboardInterrupt:
        rprint(f"[bold yellow]\nExiting due to KeyboardInterrupt...[/bold yellow]")
    finally:
        await mcp_toolkit.disconnect()
        rprint("[bold red]MCP Toolkit disconnected. Program ended.[/bold red]")


if __name__ == "__main__":
    asyncio.run(main()) 
```

**User Experience Features:**
- **Rich Terminal Output**: Colorful, formatted console messages
- **Graceful Exit**: Multiple exit commands and keyboard interrupt handling
- **Example Prompts**: Shows users how to interact with the agent
- **Async Processing**: `agent.astep()` handles tool calls and response generation
- **Clean Shutdown**: Properly disconnects from MCP server on exit



## 🏃 Running the Toolkit

1. Ensure `.env` is set up.
2. Run `python object_agent.py`.
3. At the prompt, enter a query like: "find cars and tree in this image and tell me what you see in there - https://media.istockphoto.com/id/159406920/photo/aerial-view-of-cars-in-traffic.jpg?s=612x612&w=0&k=20&c=2BeXnk1EppE_mfWgYFqoXidmge0uAhSk9gl8bbtHyF8="

The agent connects to ACI's MCP server, runs detection via Replicate, and responds.

## ✅ Checking Expected Output

Here's a sample run:

```
Started ObjectDetectionAgent. Enter your request or type 'exit' to quit.
Example: 'Find cars in this image: https://example.com/image.jpg'

You: find cars and tree in this image and tell me what you see in there - https://media.istockphoto.com/id/159406920/photo/aerial-view-of-cars-in-traffic.jpg?s=612x612&w=0&k=20&c=2BeXnk1EppE_mfWgYFqoXidmge0uAhSk9gl8bbtHyF8=
ObjectDetectionAgent:
Here's what I found in the image:

**Natural Language Summary:**
The object detection model successfully identified multiple instances of "cars" and a few "trees" in the provided aerial image of traffic. The model shows high confidence 
in detecting the cars, with scores ranging from approximately 0.25 to 0.57, indicating a strong presence of vehicles. The trees were detected with slightly lower 
confidence, around 0.25 to 0.37, and appear to be located towards the edges of the image. The bounding boxes clearly outline the detected objects, providing a visual 
representation of their positions within the traffic scene.

**Markdown Table:**

| Object | Confidence Score | Bounding Box Coordinates (x_min, y_min, x_max, y_max) |
|---|---|---|
| cars | 0.57 | [1, 38, 608, 405] |
| tree | 0.37 | [0, 18, 68, 140] |
| cars | 0.32 | [222, 300, 322, 406] |
| cars | 0.33 | [94, 286, 188, 376] |
| cars | 0.34 | [347, 265, 446, 354] |
| cars | 0.30 | [241, 212, 311, 281] |
| cars | 0.31 | [137, 205, 206, 270] |
| cars | 0.30 | [335, 192, 406, 254] |
| cars | 0.30 | [476, 253, 599, 349] |
| cars | 0.29 | [2, 226, 91, 295] |
| cars | 0.29 | [428, 193, 505, 255] |
| cars | 0.25 | [386, 353, 505, 407] |
| cars | 0.27 | [247, 160, 299, 208] |
| tree | 0.25 | [18, 0, 71, 68] |
| cars | 0.25 | [411, 150, 468, 198] |
| cars | 0.27 | [53, 172, 121, 225] |
| cars | 0.26 | [570, 97, 607, 129] |
| cars | 0.26 | [0, 299, 59, 395] |
| cars | 0.25 | [89, 134, 138, 172] |
| cars | 0.26 | [157, 146, 211, 198] |

**Result Image:**
![Detected Objects](https://replicate.delivery/xezq/PtiJITIVM6KVBtdXBNAT6X0o0pwkbYS3PH95QYCf2vFeeZfTB/result.png)
```

## 🎯 Conclusion

You've now built an object detection agent that turns complex ML into simple chats. MCP via ACI.dev makes this possible, abstracting heavy lifting so agents can focus on natural interactions. Extend it for inventory tracking, security, or more— the possibilities are endless.

That’s everything! Got questions about 🐫 CAMEL-AI? Join us on [Discord](https://discord.gg/EXAMPLE)! Whether you want to share feedback, explore the latest in multi-agent systems, get support, or connect with others on exciting projects, we’d love to have you in the community! 🤝

Check out some of our other work:
- 🐫 [Creating Your First CAMEL Agent](free Colab)
- [Graph RAG Cookbook](free Colab)
- 🧑‍⚖️ [Create A Hackathon Judge Committee with Workforce](free Colab)
- 🔥 [3 ways to ingest data from websites with Firecrawl & CAMEL](free Colab)
- 🦥 [Agentic SFT Data Generation with CAMEL and Mistral Models, Fine-Tuned with Unsloth](free Colab)

Thanks from everyone at 🐫 CAMEL-AI
