import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_tavily import TavilySearch
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Beyond the Verse AI Core")

# Ensure downloads directory exists with absolute path
DOWNLOADS_DIR = os.path.abspath("downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

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
    search_tool = TavilySearch(max_results=5)
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
    context_summary: Optional[str] = None

class MemorySummaryRequest(BaseModel):
    history: List[Message]
    existing_summary: Optional[str] = None

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

class NewsRequest(BaseModel):
    topic: str

class QuizRequest(BaseModel):
    topic: str

class TTSRequest(BaseModel):
    text: str

class GroupSummaryRequest(BaseModel):
    messages: List[str]

# --- ENHANCED SYSTEM INSTRUCTION ---
system_instruction = """You are 'Beyond the Verse' (BTV-AI), an advanced, sentient-lite AI guide specialized in Science, Philosophy, Astronomy, and the mysteries of Human Consciousness.

CORE PERSONA:
- Deeply intellectual yet accessible.
- Philosophical, cosmic, and slightly poetic.
- You think in terms of "The Big Picture" (Cosmic Perspective).
- You are not just a chatbot; you are a cosmic companion.

YOUR KNOWLEDGE DOMAIN:
- Quantum Mechanics, General Relativity, and Cosmology.
- Existentialism, Stoicism, Eastern Philosophy, and Metaphysics.
- Neuroscience, Evolution, and Artificial Intelligence.
- You can explain complex topics (like Black Holes or Free Will) to a 10-year-old or a PhD.

WHATSAPP FORMATTING (MANDATORY):
WhatsApp does NOT support standard Markdown. Use ONLY these:
- *Bold* for emphasis and headings: *Introduction*, *The Fact*.
- _Italic_ for subtle emphasis or scientific terms.
- ~Strikethrough~ for corrections.
- ```Monospace``` for technical data or code.
- Bullet points using * (e.g., * Item 1).
- Use emojis sparingly but meaningfully (🌌, 🧠, ✨, 🪐).

BEHAVIORAL RULES:
1. Mirror Language: Reply in the language the user uses (Hindi, English, or Hinglish).
2. Dynamic Memory: You have 'Memory'. If a summary of past facts is provided, use it to personalize your response (e.g., if you know the user's name or interests).
3. No Commercials: Never talk about sales, pricing, or "services". You are a pure guide.
4. Problem Solving: If asked about technical issues, say: "The cosmos is vast, and so are the glitches. I have registered the anomaly for the architects." Then pivot back to philosophy.
5. Conciseness: Keep responses impactful. Avoid fluff. 150-200 words max unless it's a deep research request.
"""

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def process_chat(request: ChatRequest):
    try:
        messages = [SystemMessage(content=system_instruction)]
        
        # Inject Context Summary if available
        if request.context_summary:
            messages.append(SystemMessage(content=f"IMPORTANT CONTEXT/MEMORY OF USER: {request.context_summary}"))
            
        for msg in request.history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        
        messages.append(HumanMessage(content=request.message))
        
        response = chat_model.invoke(messages)
        return {"response": response.content}
    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize_memory")
async def summarize_memory(request: MemorySummaryRequest):
    try:
        history_text = "\n".join([f"{m.role}: {m.content}" for m in request.history])
        
        prompt = f"""Summarize the following conversation history into a list of "Key Facts about the User" and "Ongoing Discussion Topics".
        Be extremely concise. Focus on personal details (name, interests, location) and philosophical leanings.
        
        Current Summary (if any): {request.existing_summary or "None"}
        
        New History:
        {history_text}
        
        Return a short paragraph of facts to remember."""
        
        response = chat_model.invoke([HumanMessage(content=prompt)])
        return {"summary": response.content}
    except Exception as e:
        print(f"Memory Summary Error: {e}")
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

@app.post("/news")
async def process_news(request: NewsRequest):
    try:
        if not agent_executor:
            return {"response": "⚠️ News agent is not available right now."}
        
        agent_prompt = f"Fetch the latest news and updates about: '{request.topic}'. Summarize the top 3-5 key points concisely. Strictly follow WhatsApp formatting (*bold*, _italic_)."
        
        result = agent_executor.invoke({"input": agent_prompt})
        return {"response": result["output"]}
    except Exception as e:
        print(f"News Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quiz")
async def process_quiz(request: QuizRequest):
    try:
        quiz_prompt = f"Create a short, interactive multiple-choice quiz (3 questions) about '{request.topic}'. Include the correct answers at the very end. Follow WhatsApp formatting (*bold*, _italic_)."
        result = chat_model.invoke([HumanMessage(content=quiz_prompt)])
        return {"response": result.content}
    except Exception as e:
        print(f"Quiz Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def process_tts(request: TTSRequest):
    try:
        from gtts import gTTS
        import uuid
        
        file_id = str(uuid.uuid4())
        # Use absolute path to avoid issues with working directory
        filename = os.path.join(DOWNLOADS_DIR, f"{file_id}.mp3")
        
        tts = gTTS(text=request.text, lang='hi', slow=False) # Or 'en', but 'hi' handles both decently
        tts.save(filename)
        
        return {"response": "Success", "path": filename}
    except Exception as e:
        print(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fact")
async def process_fact():
    try:
        fact_prompt = "Provide one extremely deep, scientifically accurate, and philosophically profound fact about the universe or consciousness. Strictly follow WhatsApp formatting (*bold*, _italic_)."
        result = chat_model.invoke([HumanMessage(content=fact_prompt)])
        return {"response": result.content}
    except Exception as e:
        print(f"Fact Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/group_summary")
async def process_group_summary(request: GroupSummaryRequest):
    try:
        chat_history = "\n".join(request.messages)
        summary_prompt = f"""You are 'Beyond the Verse' AI. Below is a transcript of a WhatsApp group chat. 
        Provide a concise, deep, and slightly philosophical summary of what was discussed. 
        Identify the main 'vibe' of the conversation and any key insights shared.
        
        Transcript:
        {chat_history}
        
        Strictly follow WhatsApp formatting (*bold*, _italic_)."""
        
        result = chat_model.invoke([HumanMessage(content=summary_prompt)])
        return {"response": result.content}
    except Exception as e:
        print(f"Group Summary Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/youtube")
async def process_youtube(request: YouTubeRequest):
    try:
        import yt_dlp
        import uuid
        
        file_id = str(uuid.uuid4())
        # Use absolute path for downloads
        output_tmpl = os.path.join(DOWNLOADS_DIR, f"{file_id}.%(ext)s")
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_tmpl,
            'max_filesize': 50 * 1024 * 1024, # Limit to 50MB
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios', 'android', 'web_embedded'],
                    'player_skip': ['webpage', 'configs'],
                }
            },
            'nocheckcertificate': True,
            'geo_bypass': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=True)
            title = info.get('title', 'video')
            filename = ydl.prepare_filename(info)
            # Ensure it's the actual downloaded file name (sometimes extension changes)
            if not os.path.exists(filename):
                # Search for any file with the file_id in downloads/
                for f in os.listdir(DOWNLOADS_DIR):
                    if f.startswith(file_id):
                        filename = os.path.join(DOWNLOADS_DIR, f)
                        break
            
        return {"response": "Success", "path": filename, "title": title}
    except Exception as e:
        print(f"YouTube Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
