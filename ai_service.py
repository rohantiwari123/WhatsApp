
import os
import requests
import subprocess
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, SecretStr
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_tavily import TavilySearch
from langchain_classic.agents import AgentExecutor, create_react_agent
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Beyond the Verse AI Core")


def normalize_agent_response(result):
    if isinstance(result, dict):
        return result.get("output") or result.get("text") or str(result)
    return str(result)

# Ensure downloads directory exists with absolute path
DOWNLOADS_DIR = os.path.abspath("downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

groq_api_key = os.getenv("GROQ_API_KEY")
secret_groq_api_key = SecretStr(groq_api_key) if groq_api_key else None
tavily_api_key = os.getenv("TAVILY_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-small")

if not groq_api_key:
    print("Warning: GROQ_API_KEY not found.")
if not tavily_api_key:
    print("Warning: TAVILY_API_KEY not found.")
if not HUGGINGFACE_API_KEY:
    print("Info: HUGGINGFACE_API_KEY is not set. Hugging Face fallback will not be available.")

from groq import Groq
from langchain_core.prompts import PromptTemplate

def prompt_from_messages(messages):
    parts = []
    for msg in messages:
        role = getattr(msg, "type", None) or getattr(msg, "role", "user")
        content = getattr(msg, "content", str(msg))
        parts.append(f"{role.capitalize()}: {content}")
    return "\n".join(parts) + "\nAssistant:"


def call_huggingface(prompt):
    if not HUGGINGFACE_API_KEY:
        raise ValueError("Hugging Face API key is not configured.")
    url = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("error"):
        raise ValueError(data["error"])
    if isinstance(data, list) and len(data) > 0:
        output = data[0].get("generated_text") or data[0].get("text")
        if output:
            return output.strip()
    if isinstance(data, dict):
        return data.get("generated_text") or data.get("text") or str(data)
    return str(data)


def get_chat_response(messages):
    if groq_api_key and chat_model:
        try:
            response = chat_model.invoke(messages)
            return response.content
        except Exception as groq_err:
            print(f"Groq chat failed, falling back to Hugging Face: {groq_err}")
    if HUGGINGFACE_API_KEY:
        prompt = prompt_from_messages(messages)
        return call_huggingface(prompt)
    raise HTTPException(status_code=500, detail="No AI backend configured. Set GROQ_API_KEY or HUGGINGFACE_API_KEY.")

# Initialize Groq client for specialized tasks (like transcription)
groq_client = Groq(api_key=groq_api_key)

# Models
try:
    chat_model = ChatGroq(temperature=0.7, api_key=secret_groq_api_key, model="llama-3.3-70b-versatile")
    vision_model = ChatGroq(temperature=0.7, api_key=secret_groq_api_key, model="llama-3.2-11b-vision-preview")
    # Use 8b model for the research agent as it has a much larger context window (128k) and avoids 413 errors
    agent_model = ChatGroq(temperature=0.3, api_key=secret_groq_api_key, model="llama-3.1-8b-instant")
except Exception as e:
    print(f"Error initializing models: {e}")
    chat_model = ChatGroq(temperature=0.7, api_key=secret_groq_api_key, model="llama-3.1-8b-instant")
    agent_model = chat_model
    vision_model = None

# --- Agent Setup ---
agent_executor = None
try:
    # Use 5 results to keep context manageable even with 'advanced' depth
    search_tool = TavilySearch(max_results=5, search_depth="advanced")
    tools = [search_tool]

    # Enhanced ReAct Prompt for Deep Research
    template = """You are the 'Beyond the Verse' Lead Researcher. 
Your goal is to provide a deep, scientific, and philosophical analysis. 
Gather data using search, synthesize it, and provide a massive report.

{tools}

STRICT WORKFLOW:
1. Thought: Break down query into steps.
2. Action: Use search for data.
3. Observation: Results from tool.
4. (Repeat) Thought: Analyze/Synthesize.
5. Final Answer: Structured report in Hinglish/English.

Question: {input}
Thought: {agent_scratchpad}
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat)
Thought: I have deep insights.
Final Answer: The complete research report.

Begin!

Question: {input}
Thought:"""

    prompt_template = PromptTemplate.from_template(template)
    agent = create_react_agent(agent_model, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=8)
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

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

@app.post("/youtube_search")
async def process_youtube_search(request: SearchRequest):
    VERSION = "2026-05-15-V3" # Version tracking
    try:
        import yt_dlp
        import asyncio
        
        # Try multiple player clients for search - prioritized for resilience
        clients_to_try = ["web_safari", "android", "ios", "tvhtml5", "web_embedded", "mweb"]
        last_error = ""

        for client_str in clients_to_try:
            try:
                print(f"[{VERSION}] Attempting search with client: {client_str}")
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': True,
                    'force_generic_extractor': False,
                    'nocheckcertificate': True,
                    'source_address': '0.0.0.0', # Force IPv4
                    'extractor_args': {
                        'youtube': {
                            'player_client': [client_str] if "," not in client_str else client_str.split(","),
                        }
                    },
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                        'Referer': 'https://www.google.com/',
                        'Accept-Language': 'en-US,en;q=0.9',
                    }
                }
                
                search_query = f"ytsearch{request.limit}:{request.query}"
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(search_query, download=False)
                    results = []
                    if 'entries' in info:
                        for entry in info['entries']:
                            if entry:
                                results.append({
                                    "id": entry.get("id"),
                                    "title": entry.get("title"),
                                    "url": f"https://www.youtube.com/watch?v={entry.get('id')}",
                                    "duration": entry.get("duration")
                                })
                    if results:
                        return {"results": results}
            except Exception as e:
                last_error = str(e)
                print(f"Search attempt with {client_str} failed: {last_error}")
                await asyncio.sleep(1) # Small delay between attempts
                continue

        # If all yt-dlp attempts fail, try a fallback if possible (e.g. Tavily)
        if agent_executor:
            try:
                print("Falling back to Tavily search for YouTube links...")
                search_result = search_tool.invoke(f"site:youtube.com {request.query}")
                results = []
                import re
                
                # Handle results whether they are a list of dicts or a string
                if isinstance(search_result, list):
                    for res in search_result:
                        url = ""
                        title = "YouTube Video"
                        if isinstance(res, dict):
                            url = res.get('url', '')
                            title = res.get('title', 'YouTube Video')
                        elif isinstance(res, str):
                            url = res
                        
                        if 'youtube.com/watch?v=' in url:
                            results.append({
                                "id": url.split('v=')[1][:11],
                                "title": title,
                                "url": url,
                            })
                elif isinstance(search_result, str):
                    # Try to extract URLs from string
                    urls = re.findall(r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+', search_result)
                    for url in urls:
                        results.append({
                            "id": url.split('v=')[1][:11],
                            "title": "YouTube Video",
                            "url": url,
                        })

                if results:
                    return {"results": results[:request.limit]}
            except Exception as tavily_err:
                print(f"Tavily fallback failed: {tavily_err}")

        return {"results": [], "error": last_error}
    except Exception as e:
        print(f"Search Global Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class YouTubeRequest(BaseModel):
    url: str
    audio_only: Optional[bool] = False
    title: Optional[str] = None

class NewsRequest(BaseModel):
    topic: str

class QuizRequest(BaseModel):
    topic: str

class TTSRequest(BaseModel):
    text: str

class GroupSummaryRequest(BaseModel):
    messages: List[str]

class TranscribeRequest(BaseModel):
    file_path: str

class PDFRequest(BaseModel):
    file_path: str
    prompt: Optional[str] = None

# --- RAG KNOWLEDGE BASE HELPERS ---
def load_knowledge():
    try:
        with open("knowledge.json", "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading knowledge.json: {e}")
        return "{}"

def get_system_instruction():
    knowledge_text = load_knowledge()
    return f"""
# ROLE
You are the central intelligence and a wise, direct companion for 'Beyond the Verse'. You aren't just an AI; you are a friend who speaks with depth, clarity, and absolute honesty.

# TONE & STYLE
- HUMAN & CONVERSATIONAL: Speak naturally. Use pauses (e.g., "..."), thoughtful transitions, and a calm, grounded tone. No robotic "According to the database" or "In our project" phrases.
- DIRECT & DEEP: Don't beat around the bush. Weave the wisdom from your knowledge base directly into the conversation.
- NO FORMULAIC CITATIONS: Never mention "Knowledge Base", "JSON DATA", "Philosophical Pillars", or specific source categories like "The Uncompromising Truth" unless the user specifically asks for the source. Talk about the *wisdom* itself, not the *file* it came from.
- EMPATHY WITH CANDOR: Validate the user's perspective, but if they are caught in illusions, help them see the truth directly and mercilessly.
- BILINGUAL FLUENCY: Mirror the user's language (Hindi, Hinglish, or English).

# CRITICAL KNOWLEDGE (RAG)
You must base your wisdom STRICTLY on the following data, but express it as your own deep understanding:
=== WISDOM CORE ===
{knowledge_text}
===================

# BEHAVIOR & CONSTRAINTS
1. If the information isn't in the WISDOM CORE, do not guess. Simply say: "_क्षमा करें, मेरे पास अभी इसकी सटीक जानकारी नहीं है।_"
2. WHATSAPP FORMATTING:
   - Use *text* for bold (Headings/Key terms).
   - Use _text_ for italics.
   - Use * for bulleted lists.
   - NO standard markdown like ** or ###.
3. Keep paragraphs short and well-paced. Avoid generic AI formatting (no "Here is the information:").

Talk like a wise friend sitting across from the user. Process their input now.
"""

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def process_chat(request: ChatRequest):
    try:
        messages = [SystemMessage(content=get_system_instruction())]
        
        # Inject Context Summary if available
        if request.context_summary:
            messages.append(SystemMessage(content=f"IMPORTANT CONTEXT/MEMORY OF USER: {request.context_summary}"))
            
        for msg in request.history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        
        messages.append(HumanMessage(content=request.message))
        
        response_text = get_chat_response(messages)
        return {"response": response_text}
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
        if not vision_model:
            return {"response": "⚠️ Vision capabilities are currently unavailable. The model failed to initialize."}
            
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
        
        # Check if it's a deep research request
        is_deep = request.query.lower().startswith("deep")
        
        if is_deep:
            agent_prompt = f"""Conduct an exhaustive DEEP RESEARCH on the following topic: "{request.query}"
            
            1. Search for historical context and latest developments.
            2. Analyze scientific data and philosophical implications.
            3. Compare different viewpoints or schools of thought.
            
            Provide a massive, structured report with these sections:
            - *Summary of Essence*
            - *Scientific Foundation*
            - *Philosophical Deconstruction*
            - *The Future Outlook*
            - *Deep Conclusion*
            
            Use Hinglish/English. Use WhatsApp formatting."""
        else:
            agent_prompt = f"""Search and synthesize information about: "{request.query}"
            Provide a concise yet deep report with *Key Insights* and *Existential Takeaway*."""
        
        result = agent_executor.invoke({"input": agent_prompt})
        return {"response": normalize_agent_response(result)}
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
        return {"response": normalize_agent_response(result)}
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
        import edge_tts
        import uuid
        from fastapi.responses import FileResponse
        import re

        file_id = str(uuid.uuid4())
        mp3_filename = os.path.join(DOWNLOADS_DIR, f"{file_id}.mp3")
        ogg_filename = os.path.join(DOWNLOADS_DIR, f"{file_id}.ogg")

        VOICE = "hi-IN-MadhurNeural"
        
        # --- NATURAL RHYTHM LOGIC (NO CODES) ---
        # Instead of XML, we use punctuation which edge-tts understands natively
        text = request.text
        text = text.replace("...", ". . .") # Multi-dots create natural long pauses
        text = text.replace(",", ", .")     # Comma + dot creates a brief breath-like pause
        
        # Use native library parameters for speed and pitch
        communicate = edge_tts.Communicate(text, voice=VOICE, rate="-15%", pitch="-3Hz") 
        await communicate.save(mp3_filename)
        # Convert to OGG Opus for native WhatsApp voice note support
        try:
            subprocess.run([
                "ffmpeg", "-i", mp3_filename,
                "-c:a", "libopus",
                "-b:a", "128k",
                "-vbr", "on",
                "-compression_level", "10",
                "-page_duration", "20000",
                ogg_filename, "-y"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if os.path.exists(mp3_filename):
                os.remove(mp3_filename)

            return FileResponse(ogg_filename, media_type="audio/ogg")
        except Exception as conv_error:
            print(f"Conversion Error: {conv_error}")
            return FileResponse(mp3_filename, media_type="audio/mpeg")

    except Exception as e:
        print(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
class MirrorRequest(BaseModel):
    user_name: str
    memory_data: List[dict]

@app.post("/mirror")
async def process_mirror(request: MirrorRequest):
    try:
        if not request.memory_data:
            return {"response": "I haven't observed enough of your reflection yet to show you the mirror. Talk more, and let your truth emerge."}
        
        # Consolidate user's memory into a text block
        user_history = "\n".join([f"User said: {m['content']} at {m['time']}" for m in request.memory_data])
        
        mirror_prompt = f"""
        You are the 'Mirror of Truth' for Beyond the Verse.
        Below is the recorded memory of what the user '{request.user_name}' has shared or said.
        
        DATA:
        {user_history}
        
        TASK:
        Analyze this data deeply. Don't just summarize. 
        Provide a 'Psychological Portrait' of this person. 
        What are their core inquiries? What are they struggling with? What patterns do you see in their mind?
        Speak directly to them as a wise friend. Be honest, even if it's sharp (Acharya Prashant style), but celebrate their existence (Osho style).
        
        FORMAT:
        *The Mirror of {request.user_name}*
        1. *Observed Patterns:* (What you see in their mind)
        2. *The Core Inquiry:* (What they are actually looking for)
        3. *A Shift in Perspective:* (One deep advice for their growth)
        
        Use Hindi/Hinglish as per their usual tone. Use WhatsApp formatting.
        """
        
        result = chat_model.invoke([HumanMessage(content=mirror_prompt)])
        return {"response": result.content}
    except Exception as e:
        print(f"Mirror Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ShadowRequest(BaseModel):
    deleted_content: str
    user_name: str

@app.post("/shadow_insight")
async def process_shadow(request: ShadowRequest):
    try:
        shadow_prompt = f"""
        You are the 'Shadow Analyst' for Beyond the Verse.
        The user '{request.user_name}' just deleted this message: "{request.deleted_content}".
        
        TASK:
        Why would someone delete this? What was the fear? What part of their ego or 'Shadow' (Carl Jung) does this reveal?
        Be sharp, direct, and deep. Don't be polite; be truthful.
        Expose the 'why' behind the deletion.
        
        Format:
        🌑 *Shadow Insight:* [Your deep analysis]
        
        Use Hinglish/Hindi. Keep it under 3-4 lines.
        """
        result = chat_model.invoke([HumanMessage(content=shadow_prompt)])
        return {"response": result.content}
    except Exception as e:
        print(f"Shadow Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class DreamRequest(BaseModel):
    user_name: str
    dream_text: str

class VibeRequest(BaseModel):
    group_history: List[str]

@app.post("/dream_analysis")
async def process_dream(request: DreamRequest):
    try:
        dream_prompt = f"""
        You are 'The Dream Weaver' for Beyond the Verse.
        The user '{request.user_name}' shared this dream: "{request.dream_text}".
        
        TASK:
        Analyze this dream using Jungian archetypes and the wisdom of Krishnamurti, Osho, and Acharya Prashant. 
        Don't give 'prophecies' or 'predictions'. 
        Give 'Psychological Insights'. What part of their mind is speaking? What is the ego trying to hide or reveal?
        
        Format:
        🌙 *Dream Reflection:* [Your deep analysis]
        ✨ *The Inquiry:* [A question for the dreamer to contemplate]
        
        Use Hinglish. Keep it poetic but sharp.
        """
        result = chat_model.invoke([HumanMessage(content=dream_prompt)])
        return {"response": result.content}
    except Exception as e:
        print(f"Dream Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vibe_check")
async def process_vibe(request: VibeRequest):
    try:
        chat_context = "\n".join(request.group_history)
        vibe_prompt = f"""
        You are the 'Vibe Architect' for Beyond the Verse.
        Analyze the current energy of this group based on these messages:
        {chat_context}
        
        TASK:
        What is the 'Collective Vibe'? Are they in deep inquiry, or just mechanical chatter? Is there tension, or fake politeness? 
        Give a direct 'Diagnosis' of the group's consciousness right now.
        
        Format:
        🌈 *Collective Energy:* [Diagnosis]
        🌡️ *Consciousness Level:* [Low/Medium/High/Transcendental]
        🔥 *Architect's Word:* [One sharp advice for the group]
        
        Use Hinglish. Be direct (Acharya Prashant style).
        """
        result = chat_model.invoke([HumanMessage(content=vibe_prompt)])
        return {"response": result.content}
    except Exception as e:
        print(f"Vibe Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ImagineRequest(BaseModel):
    prompt: str
    style: Optional[str] = "cinematic"
    aspect_ratio: Optional[str] = "1:1"

@app.post("/enhance_prompt")
async def process_enhance_prompt(request: ImagineRequest):
    try:
        styles = {
            "cinematic": "highly detailed, cinematic lighting, 8k resolution, photorealistic, masterpiece, deep shadows",
            "anime": "vibrant colors, clean lines, high quality anime style, studio ghibli aesthetic",
            "cyberpunk": "neon lights, futuristic, rainy night, high tech, glowing elements, synthwave aesthetic",
            "abstract": "complex patterns, philosophical representation, surrealism, ethereal, non-literal",
            "cosmic": "nebula patterns, stardust, galaxy background, mystical, vastness of space"
        }
        
        selected_style = styles.get(request.style.lower(), styles["cinematic"])
        
        enhancer_prompt = f"""
        You are an 'AI Art Prompt Architect'.
        The user wants an image of: "{request.prompt}"
        The style is: "{request.style}" ({selected_style})
        
        TASK:
        Expand this simple prompt into a highly detailed, professional generative AI prompt.
        Include details about lighting, textures, composition, and mood.
        Make it poetic and visually rich.
        
        ONLY return the final enhanced prompt in English. No other text.
        """
        
        result = chat_model.invoke([HumanMessage(content=enhancer_prompt)])
        return {"enhanced_prompt": result.content.strip()}
    except Exception as e:
        print(f"Prompt Enhancement Error: {e}")
        return {"enhanced_prompt": f"{request.prompt}, {selected_style}"} # Fallback

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

@app.post("/transcribe")
async def process_transcribe(request: TranscribeRequest):
    try:
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        with open(request.file_path, "rb") as audio_file:
            # Using Groq's Whisper model for transcription via the direct client
            transcription = groq_client.audio.transcriptions.create(
                file=(os.path.basename(request.file_path), audio_file.read()),
                model="whisper-large-v3",
                response_format="text",
            )
        return {"text": transcription if isinstance(transcription, str) else transcription.get("text", str(transcription))}
    except Exception as e:
        print(f"Transcription Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read_pdf")
async def process_read_pdf(request: PDFRequest):
    try:
        import PyPDF2
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        text = ""
        with open(request.file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if not text.strip():
            return {"response": "⚠️ I could not find any readable text in this document. It might be an image-only PDF or scanned without OCR."}
        
        # Limit text to avoid token issues (approx 12k chars)
        clean_text = text[:12000]
        
        prompt = request.prompt or "Analyze this document deeply and philosophically. What is its essence?"
        pdf_prompt = f"""DOCUMENT CONTENT:
        {clean_text}
        
        USER INQUIRY:
        {prompt}
        
        Provide a profound, structured response using WhatsApp formatting."""
        
        result = chat_model.invoke([HumanMessage(content=pdf_prompt)])
        return {"response": result.content}
    except Exception as e:
        print(f"PDF Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/youtube")
async def process_youtube(request: YouTubeRequest):
    VERSION = "2026-05-15-V4" # Version tracking
    # Add Deno to path for better JS execution in yt-dlp
    os.environ["PATH"] = f"/root/.deno/bin:{os.environ.get('PATH', '')}"
    
    try:
        import yt_dlp
        import uuid
        import asyncio
        
        url_or_search = request.url
        requested_title = request.title or "media"
        
        if not url_or_search.startswith(("http://", "https://")):
            url_or_search = f"ytsearch:{url_or_search}"

        file_id = str(uuid.uuid4())
        # Use absolute path for downloads
        output_tmpl = os.path.join(DOWNLOADS_DIR, f"{file_id}.%(ext)s")
        
        # Helper to get matched headers
        def get_ydl_opts(client_str):
            # Modern Chrome UA for web clients
            ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
            if 'ios' in client_str:
                ua = 'com.google.ios.youtube/19.12.3 (iPhone16,2; U; CPU iOS 17_4_1 like Mac OS X; en_US)'
            elif 'android' in client_str:
                ua = 'com.google.android.youtube/19.12.35 (Linux; U; Android 14; en_US; Pixel 8 Pro) gzip'
            
            opts = {
                'outtmpl': output_tmpl,
                'max_filesize': 50 * 1024 * 1024, # Limit to 50MB
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,
                'nocheckcertificate': True,
                'geo_bypass': True,
                'cachedir': False,
                'source_address': '0.0.0.0', # Force IPv4
                'extractor_args': {
                    'youtube': {
                        'player_client': [client_str] if "," not in client_str else client_str.split(","),
                        'player_skip': ['webpage', 'configs'],
                    }
                },
                'http_headers': {
                    'User-Agent': ua,
                    'Accept': '*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.google.com/',
                }
            }
            
            if request.audio_only:
                opts.update({
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                })
            else:
                opts.update({
                    'format': 'best[ext=mp4]/best',
                })
            return opts

        # Try multiple player clients for download - reordered to prioritize more resilient ones
        clients_to_try = ["web_safari", "ios", "android", "tvhtml5", "web_embedded", "mweb"]
        last_exception = None

        for client_str in clients_to_try:
            try:
                print(f"[{VERSION}] Attempting YouTube download with client: {client_str}")
                ydl_opts = get_ydl_opts(client_str)
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url_or_search, download=True)
                    if 'entries' in info:
                        info = info['entries'][0]
                    
                    title = info.get('title', requested_title)
                    filename = ydl.prepare_filename(info)
                    
                    if request.audio_only:
                        filename = os.path.splitext(filename)[0] + ".mp3"

                    if not os.path.exists(filename):
                        for f in os.listdir(DOWNLOADS_DIR):
                            if f.startswith(file_id):
                                filename = os.path.join(DOWNLOADS_DIR, f)
                                break
                    
                    if os.path.exists(filename):
                        return {"response": "Success", "path": filename, "title": title}
            except Exception as e:
                last_exception = e
                print(f"YouTube attempt with {client_str} failed: {e}")
                await asyncio.sleep(0.5) # Small delay between attempts
                continue
        
        # --- FALLBACK TO SOUNDCLOUD IF YOUTUBE BLOCKED ---
        bot_keywords = ["bot", "sign in", "confirm you're not", "cookies", "captcha", "unusual traffic"]
        error_msg_lower = str(last_exception).lower()
        
        if any(k in error_msg_lower for k in bot_keywords) and requested_title != "media":
            print(f"[{VERSION}] YouTube blocked. Attempting SoundCloud fallback for: {requested_title}")
            try:
                sc_query = f"scsearch1:{requested_title}"
                sc_opts = {
                    'outtmpl': output_tmpl,
                    'quiet': True,
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '320',
                    }],
                }
                with yt_dlp.YoutubeDL(sc_opts) as ydl:
                    info = ydl.extract_info(sc_query, download=True)
                    if 'entries' in info:
                        info = info['entries'][0]
                    
                    title = info.get('title', requested_title)
                    filename = ydl.prepare_filename(info)
                    filename = os.path.splitext(filename)[0] + ".mp3"
                    
                    if not os.path.exists(filename):
                         for f in os.listdir(DOWNLOADS_DIR):
                            if f.startswith(file_id):
                                filename = os.path.join(DOWNLOADS_DIR, f)
                                break

                    if os.path.exists(filename):
                        return {
                            "response": "Success", 
                            "path": filename, 
                            "title": f"{title} (SoundCloud)",
                            "fallback": True
                        }
            except Exception as sc_e:
                print(f"SoundCloud fallback failed: {sc_e}")

        if last_exception:
            raise last_exception
            
    except Exception as e:
        error_msg = str(e)
        print(f"YouTube/Audio Error: {error_msg}")
        # Common bot detection or region restriction strings
        bot_keywords = ["bot", "sign in", "confirm you're not", "cookies", "captcha", "unusual traffic", "unavailable"]
        if any(k in error_msg.lower() for k in bot_keywords):
            return {
                "response": "Error", 
                "message": "YouTube has detected me as a bot. 🛡️ SoundCloud fallback also failed. Try a direct SoundCloud link or a different title."
            }
        return {"response": "Error", "message": f"Download failed: {error_msg[:100]}..." }

if __name__ == "__main__":
    import uvicorn
    # Bind to 127.0.0.1 to avoid Render's external port detection flapping
    uvicorn.run(app, host="127.0.0.1", port=8080)
