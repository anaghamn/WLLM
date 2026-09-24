"""Expt 2 - AI Weather Dashboard (MCP server + MCP client + Streamlit UI, in one file).
Shows the LLM's thought process: its reasoning, the decision to call the tool, the raw data, and the final answer.

The same file plays two roles:
  * `python lab6exp2.py --server`  -> MCP server exposing get_current_weather (live data from wttr.in)
  * `streamlit run lab6exp2.py`     -> Streamlit UI, which launches itself as the MCP server

Setup:  pip install mcp groq streamlit httpx
Run:    streamlit run lab6exp2.py
"""
import asyncio
import json
import os
import sys

MODEL = "openai/gpt-oss-120b"
THIS_FILE = os.path.abspath(__file__)


# ----------------------------------------------------------------------------
# MCP SERVER
# ----------------------------------------------------------------------------
def run_server():
    import httpx
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("weather-server")

    @mcp.tool()
    def get_current_weather(location: str) -> str:
        """Get the current weather and 3-day forecast for a city. Pass city and country, e.g. "Tokyo, Japan"."""
        try:
            r = httpx.get(f"https://wttr.in/{location}", params={"format": "j1"}, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return f"Error fetching weather for {location}: {e}"

        now = data["current_condition"][0]
        area = data["nearest_area"][0]
        place = f"{area['areaName'][0]['value']}, {area['country'][0]['value']}"

        lines = [
            f"Location: {place}",
            f"Condition: {now['weatherDesc'][0]['value']}",
            f"Temperature: {now['temp_C']} C (feels like {now['FeelsLikeC']} C)",
            f"Humidity: {now['humidity']}%",
            f"Wind: {now['windspeedKmph']} km/h {now['winddir16Point']}",
            f"Observed at: {now.get('localObsDateTime', now.get('observation_time', 'n/a'))}",
            "Forecast:",
        ]
        for day in data["weather"]:
            desc = day["hourly"][4]["weatherDesc"][0]["value"]  # around midday
            lines.append(f"  {day['date']}: {day['mintempC']}-{day['maxtempC']} C, {desc}")
        return "\n".join(lines)

    mcp.run()  # stdio transport


if "--server" in sys.argv:
    run_server()
    sys.exit(0)


# ----------------------------------------------------------------------------
# MCP CLIENT + STREAMLIT UI
# ----------------------------------------------------------------------------
import streamlit as st
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SYSTEM = (
    "You are a weather assistant. Use get_current_weather to fetch live data, "
    "then give a short, friendly summary with the key numbers."
)


async def ask(client, question, status):
    params = StdioServerParameters(command=sys.executable, args=[THIS_FILE, "--server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            tools = [
                {"type": "function", "function": {
                    "name": t.name, "description": t.description, "parameters": t.input_schema,
                }}
                for t in mcp_tools
            ]
            status.write(f"🔌 Connected to MCP server. Tools: `{[t.name for t in mcp_tools]}`")

            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ]
            while True:
                status.write("🧠 Sending question and available tools to the LLM...")
                response = client.chat.completions.create(
                    model=MODEL, messages=messages, tools=tools, tool_choice="auto",
                )
                msg = response.choices[0].message
                if getattr(msg, "reasoning", None):
                    status.markdown(f"💭 **LLM reasoning:** {msg.reasoning}")

                if not msg.tool_calls:
                    status.write("✅ LLM has enough data and is writing the final answer.")
                    return msg.content

                if msg.content:
                    status.markdown(f"💭 **LLM:** {msg.content}")
                messages.append({
                    "role": "assistant", "content": msg.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ],
                })
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    status.markdown(f"🛠️ **LLM decided to call:** `{tc.function.name}({args})`")
                    result = await session.call_tool(tc.function.name, args)
                    output = "".join(c.text for c in result.content if c.type == "text")
                    status.markdown("📦 **Data returned by server:**")
                    status.code(output)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})


st.set_page_config(page_title="Weather Dashboard", page_icon="🌦️")
st.title("🌦️ AI Weather Dashboard")
st.caption(f"Streamlit MCP client ↔ weather MCP server (wttr.in) ↔ Groq `{MODEL}`")

with st.sidebar:
    api_key = st.text_input("GROQ API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    st.caption("Get one at https://console.groq.com/keys")

question = st.chat_input("Ask about the weather, e.g. What's the weather in Tokyo?")
if question and not api_key:
    st.error("Please enter your GROQ API key in the sidebar.")
elif question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.status("LLM thought process", expanded=True) as status:
            try:
                answer = asyncio.run(ask(Groq(api_key=api_key), question, status))
                status.update(label="LLM thought process (done)", state="complete")
            except Exception as e:
                status.update(label="Error", state="error")
                answer = None
                st.error(f"Error: {e}")
        if answer:
            st.markdown(answer)
