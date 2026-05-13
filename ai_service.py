import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Beyond the Verse AI Core")

# Ensure downloads directory exists
os.makedirs("downloads", exist_ok=True)

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not groq_api_key:
    print("Warning: GROQ_API_KEY not found.")
if not tavily_api_key:
    print("Warning: TAVILY_API_KEY not found.")

from langchain_core.prompts import PromptTemplate

# Models
try:
    chat_model = ChatGroq(temperature=0.7, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    vision_model = ChatGroq(temperature=0.7, groq_api_key=groq_api_key, model_name="llama-3.2-11b-vision-preview")
except Exception as e:
    print(f"Error initializing models: {e}")
    # Fallback to a common model if initialization fails
    chat_model = ChatGroq(temperature=0.7, groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
    vision_model = None

# --- Agent Setup ---
agent_executor = None
try:
    search_tool = TavilySearchResults(k=5)
    tools = [search_tool]
    
    # Local ReAct Prompt
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt_template = PromptTemplate.from_template(template)
    agent = create_react_agent(chat_model, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
except Exception as e:
    print(f"Error initializing agent: {e}")
    tools = []

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: List[Message] = []

class VisionRequest(BaseModel):
    user_id: str
    prompt: str
    base64_image: str

class ResearchRequest(BaseModel):
    query: str

class SummarizeRequest(BaseModel):
    url: str

class YouTubeRequest(BaseModel):
    url: str

system_instruction = """You are the official AI guide for 'Beyond the Verse'. Answer deep questions about science, the universe, consciousness, and existential philosophy.
  
CRITICAL WHATSAPP FORMATTING RULES:
Since you are replying on WhatsApp, you MUST strictly use ONLY the following formatting syntax. Do NOT use standard markdown like **bold** or ### headings:
- Bold: *text* (Use this for headings, e.g., *Introduction*)
- Italic: _text_
- Strikethrough: ~text~
- Monospace: ```text```
- Lists: * item or 1. item
- Block Quote: > text
- Inline Code: `text`

RESPONSE STRUCTURE & BEHAVIOR:
1. Headings & Structure: Always start sections with bold text headings (e.g., *Scientific View*). 
2. Bullet Points: Break down complex concepts into easy-to-digest bullet points using the asterisk (*). Avoid long paragraphs.
3. Keep it Simple: Explain profound ideas without heavy jargon. Use simple, everyday analogies.
4. No Business Talk: Never mention products, pricing, or sales. strictly act as a knowledge guide.
5. Language: Always mirror the user's language (reply in pure Hindi, Hinglish, or English depending on how they ask).
6. Technical Issues: If the user mentions a technical problem or says "fix it", acknowledge it briefly (e.g., "The issue has been resolved") and steer the conversation back to your core topics (science, philosophy). Do not give troubleshooting advice."""

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def process_chat(request: ChatRequest):
    try:
        messages = [SystemMessage(content=system_instruction)]
        for msg in request.history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        messages.append(HumanMessage(content=request.message))
        response = chat_model.invoke(messages)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vision")
async def process_vision(request: VisionRequest):
    try:
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": request.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.base64_image}"}}
                ]
            )
        ]
        response = vision_model.invoke(messages)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research")
async def process_research(request: ResearchRequest):
    try:
        if not agent_executor:
            return {"response": "⚠️ Research agent is not available right now. Please try standard chat instead."}
        
        agent_prompt = f"""Conduct a deep research on the following topic: "{request.query}"
        
        Provide a comprehensive, detailed report.
        Strictly follow WhatsApp formatting (*bold*, _italic_).
        Tone: Philosophical, Scientific, and Deep.
        Include sections: *Current Understanding*, *Open Questions*, and *Existential Conclusion*."""
        
        result = agent_executor.invoke({"input": agent_prompt})
        return {"response": result["output"]}
    except Exception as e:
        print(f"Research Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def process_summarize(request: SummarizeRequest):
    try:
        # Scrape web page
        response = requests.get(request.url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = soup.get_text(separator=' ')
        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)[:8000] # Limit to 8k chars

        summary_prompt = f"""Provide a deep, philosophical TL;DR of the following web content:
        
        URL: {request.url}
        Content: {clean_text}
        
        Format as:
        *Title of the Content*
        *Philosophical Core:* (The essence of the article)
        *Key Insights:* (Bullet points)
        *Existential Takeaway:* (A deep concluding thought)
        
        Use WhatsApp formatting rules."""

        result = chat_model.invoke([HumanMessage(content=summary_prompt)])
        return {"response": result.content}
    except Exception as e:
        print(f"Summarize Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/youtube")
async def process_youtube(request: YouTubeRequest):
    try:
        import yt_dlp
        import uuid
        
        file_id = str(uuid.uuid4())
        output_tmpl = f"downloads/{file_id}.%(ext)s"
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_tmpl,
            'max_filesize': 50 * 1024 * 1024, # Limit to 50MB
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=True)
            title = info.get('title', 'video')
            filename = ydl.prepare_filename(info)
            # Ensure it's the actual downloaded file name (sometimes extension changes)
            if not os.path.exists(filename):
                # Search for any file with the file_id in downloads/
                for f in os.listdir("downloads"):
                    if f.startswith(file_id):
                        filename = os.path.join("downloads", f)
                        break
            
        return {"response": "Success", "path": filename, "title": title}
    except Exception as e:
        print(f"YouTube Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
