"""Expt 1 - Personal Assistant with Memory (MCP server + MCP client + Streamlit UI, in one file).

The same file plays two roles:
  * `python lab6exp1.py --server`  -> MCP server exposing save_note / search_notes / delete_note
  * `streamlit run lab6exp1.py`     -> Streamlit chat UI, which launches itself as the MCP server

Setup:  pip install mcp groq streamlit
Run:    streamlit run lab6exp1.py
"""
import asyncio
import json
import os
import sys
from datetime import datetime

MODEL = "openai/gpt-oss-120b"
THIS_FILE = os.path.abspath(__file__)
NOTES_FILE = os.path.join(os.path.dirname(THIS_FILE), "lab6exp1_notes.json")


# ----------------------------------------------------------------------------
# MCP SERVER
# ----------------------------------------------------------------------------
def load_notes():
    if not os.path.exists(NOTES_FILE):
        return []
    with open(NOTES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def write_notes(notes):
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)


def run_server():
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("notes-server")

    @mcp.tool()
    def save_note(content: str, tags: list[str] = []) -> str:
        """Save a new note with optional tags (e.g. ["project", "deadline"])."""
        notes = load_notes()
        note = {
            "id": max([n["id"] for n in notes], default=0) + 1,
            "content": content,
            "tags": [t.lower() for t in tags],
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        notes.append(note)
        write_notes(notes)
        return f"Saved note #{note['id']}: {content} (tags: {', '.join(note['tags']) or 'none'})"

    @mcp.tool()
    def search_notes(query: str) -> str:
        """Search saved notes by keywords. Matches words in the note content and tags."""
        words = [w.lower().strip("?,.!") for w in query.split() if len(w) > 2]
        results = []
        for note in load_notes():
            text = (note["content"] + " " + " ".join(note["tags"])).lower()
            score = sum(1 for w in words if w in text)
            if score:
                results.append((score, note))
        results.sort(key=lambda r: r[0], reverse=True)
        if not results:
            return "No matching notes found."
        return json.dumps([note for _, note in results], indent=2)

    @mcp.tool()
    def delete_note(note_id: int) -> str:
        """Delete a note by its id."""
        notes = load_notes()
        remaining = [n for n in notes if n["id"] != note_id]
        if len(remaining) == len(notes):
            return f"Note #{note_id} not found."
        write_notes(remaining)
        return f"Deleted note #{note_id}."

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
    "You are a personal assistant with a memory. If the user tells you something to remember, "
    "call save_note with the content and a few short tags. If the user asks about something "
    "they said before, call search_notes with keywords. If asked to forget a note, search for it "
    "and then call delete_note with its id. Answer briefly."
)


async def ask(client, messages, status):
    """Run the tool-use loop against the MCP server. Appends to `messages` and returns the answer."""
    params = StdioServerParameters(command=sys.executable, args=[THIS_FILE, "--server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Convert MCP tool definitions to the function-calling format Groq expects
            mcp_tools = (await session.list_tools()).tools
            tools = [
                {"type": "function", "function": {
                    "name": t.name, "description": t.description, "parameters": t.input_schema,
                }}
                for t in mcp_tools
            ]
            status.write(f"🔌 Connected to MCP server. Tools: `{[t.name for t in mcp_tools]}`")

            # Keep going until the LLM stops asking for tools
            while True:
                response = client.chat.completions.create(
                    model=MODEL, messages=messages, tools=tools, tool_choice="auto",
                )
                msg = response.choices[0].message

                if not msg.tool_calls:
                    messages.append({"role": "assistant", "content": msg.content})
                    return msg.content

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
                    status.markdown(f"🛠️ **LLM calls** `{tc.function.name}({args})`")
                    result = await session.call_tool(tc.function.name, args)
                    output = "".join(c.text for c in result.content if c.type == "text")
                    status.code(output[:500])
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})


st.set_page_config(page_title="Memory Assistant", page_icon="🧠")
st.title("🧠 Personal Assistant with Memory")
st.caption(f"Streamlit MCP client ↔ notes MCP server ↔ Groq `{MODEL}`")

with st.sidebar:
    api_key = st.text_input("GROQ API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    st.caption("Get one at https://console.groq.com/keys")
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
    st.divider()
    st.subheader("📒 Saved notes")
    notes_box = st.container()  # filled at the end so it shows notes saved this turn

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM}]

# Show previous turns (only user messages and final assistant answers)
for m in st.session_state.messages:
    if m["role"] in ("user", "assistant") and m.get("content") and not m.get("tool_calls"):
        st.chat_message(m["role"]).write(m["content"])

query = st.chat_input("e.g. Remember that my exam is on Friday / What did I say about my exam?")
if query and not api_key:
    st.error("Please enter your GROQ API key in the sidebar.")
elif query:
    st.chat_message("user").write(query)
    messages = st.session_state.messages + [{"role": "user", "content": query}]
    with st.chat_message("assistant"):
        with st.status("Tool calls", expanded=False) as status:
            try:
                answer = asyncio.run(ask(Groq(api_key=api_key), messages, status))
                st.session_state.messages = messages  # commit the turn only if it succeeded
                status.update(label="Tool calls (done)", state="complete")
                answer_ok = True
            except Exception as e:
                status.update(label="Error", state="error")
                answer, answer_ok = f"Error: {e}", False
        (st.markdown if answer_ok else st.error)(answer)

with notes_box:
    notes = load_notes()
    if not notes:
        st.write("No notes yet.")
    for n in notes:
        st.markdown(f"**#{n['id']}** {n['content']}  \n`{', '.join(n['tags']) or 'no tags'}` · {n['created']}")
