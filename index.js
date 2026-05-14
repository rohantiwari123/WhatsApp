import "dotenv/config";
import { MongoClient } from 'mongodb';
import pino from "pino";
import makeWASocket, {
  DisconnectReason,
  downloadMediaMessage,
  BufferJSON,
  initAuthCreds,
  proto,
  Browsers,
} from "@whiskeysockets/baileys";
import Groq from "groq-sdk";
import qrcode from "qrcode-terminal";
import cron from "node-cron";
import { tavily } from "@tavily/core";
import fs from "fs";
import readline from "readline";

import express from "express";
const app = express();
const port = process.env.PORT || 3000;

app.get("/", (req, res) => res.send("Beyond the Verse AI is Live!"));
app.listen(port, () => console.log(`🌐 Server running on port ${port}`));

// ----------------------------------------------------
// ⌨️ 0. INTERACTIVE INPUT SETUP
// ----------------------------------------------------
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const question = (text) => new Promise((resolve) => rl.question(text, resolve));

// ----------------------------------------------------
// 🧠 1. PERSISTENT MEMORY SYSTEM (SAVES SESSIONS TO DISK)
// ----------------------------------------------------
const SESSION_FILE = "./user_sessions.json";
let userSessions = new Map();
let groupMsgBuffer = new Map(); // Store last 20 messages per group for summary

// Wait for Python Backend to be ready
async function waitForBackend() {
  console.log("⏳ Waiting for Python AI Core to start on port 8080...");
  let ready = false;
  while (!ready) {
    try {
      const response = await fetch("http://localhost:8080/health");
      if (response.ok) {
        ready = true;
        console.log("✅ Python AI Core is online and ready!");
      }
    } catch (e) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }
}

// Helper for fetch with timeout
async function fetchWithTimeout(url, options = {}, timeout = 60000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

// Load sessions from file if it exists so we don't lose context on restart
function loadSessions() {
  if (fs.existsSync(SESSION_FILE)) {
    try {
      const data = fs.readFileSync(SESSION_FILE, "utf-8");
      const parsed = JSON.parse(data);
      userSessions = new Map(Object.entries(parsed));
      console.log("✅ User memory (sessions) successfully loaded from disk.");
    } catch (e) {
      console.error("⚠️ Failed to load memory:", e);
    }
  }
}

// Save sessions to disk
function saveSessions() {
  try {
    const obj = Object.fromEntries(userSessions);
    fs.writeFileSync(SESSION_FILE, JSON.stringify(obj, null, 2), "utf-8");
  } catch (e) {
    console.error("⚠️ Failed to save memory:", e);
  }
}

// Initialize memory
loadSessions();

// ----------------------------------------------------
// 🤖 2. API CLIENTS SETUP
// ----------------------------------------------------
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY });

const systemInstruction = `You are the official AI guide for 'Beyond the Verse'. Answer deep questions about science, the universe, consciousness, and existential philosophy.
  
CRITICAL WHATSAPP FORMATTING RULES:
Since you are replying on WhatsApp, you MUST strictly use ONLY the following formatting syntax. Do NOT use standard markdown like **bold** or ### headings:
- Bold: *text* (Use this for headings, e.g., *Introduction*)
- Italic: _text_
- Strikethrough: ~text~
- Monospace: \`\`\`text\`\`\`
- Lists: * item or 1. item
- Block Quote: > text
- Inline Code: \`text\`

RESPONSE STRUCTURE & BEHAVIOR:
1. Headings & Structure: Always start sections with bold text headings (e.g., *Scientific View*). 
2. Bullet Points: Break down complex concepts into easy-to-digest bullet points using the asterisk (*). Avoid long paragraphs.
3. Keep it Simple: Explain profound ideas without heavy jargon. Use simple, everyday analogies.
4. No Business Talk: Never mention products, pricing, or sales. strictly act as a knowledge guide.
5. Language: Always mirror the user's language (reply in pure Hindi, Hinglish, or English depending on how they ask).`;

// ----------------------------------------------------
// 🚀 3. MAIN WHATSAPP CONNECTION LOOP
// ----------------------------------------------------
const mongoClient = new MongoClient(process.env.MONGODB_URI);
await mongoClient.connect();
const authCollection = mongoClient.db("whatsapp_bot").collection("auth_session");

let isConnecting = false;
let pairingTimeout = null;
let reconnectAttempts = 0;

async function connectToWhatsApp() {
    if (isConnecting) return;
    isConnecting = true;

    // Clear any existing pairing timeout
    if (pairingTimeout) {
        clearTimeout(pairingTimeout);
        pairingTimeout = null;
    }

    // === MONGODB AUTH SETUP START ===
    console.log("⏳ WhatsApp से कनेक्ट हो रहा है (MongoDB Auth)...");

    const writeData = async (data, id) => {
        try {
            const stringified = JSON.stringify(data, BufferJSON.replacer);
            await authCollection.replaceOne(
                { _id: id },
                { value: stringified },
                { upsert: true }
            );
        } catch (error) {
            console.error(`❌ MongoDB Write Error (${id}):`, error.message);
            throw error;
        }
    };

    const readData = async (id) => {
        try {
            const data = await authCollection.findOne({ _id: id });
            if (!data || !data.value) return null;
            return JSON.parse(data.value, BufferJSON.reviver);
        } catch (error) {
            console.error(`❌ MongoDB Read Error (${id}):`, error.message);
            return null;
        }
    };

    const removeData = async (id) => {
        try {
            await authCollection.deleteOne({ _id: id });
        } catch (error) {
            console.error(`❌ MongoDB Delete Error (${id}):`, error.message);
        }
    };

    console.log("🔑 Authentication क्रेडेंशियल्स लोड हो रहे हैं...");
    let creds = await readData('creds');
    if (!creds) {
        console.log("🆕 कोई पिछला सेशन नहीं मिला। नया क्रेडेंशियल्स जेनरेट हो रहा है...");
        creds = initAuthCreds();
        await writeData(creds, 'creds');
    } else {
        console.log("✅ पिछला सेशन MongoDB से सफलतापूर्वक लोड हुआ।");
    }

    const state = {
        creds,
        keys: {
            get: async (type, ids) => {
                const data = {};
                await Promise.all(
                    ids.map(async (id) => {
                        let value = await readData(`${type}-${id}`);
                        if (type === 'app-state-sync-key' && value) {
                            value = proto.Message.AppStateSyncKeyData.fromObject(value);
                        }
                        data[id] = value;
                    })
                );
                return data;
            },
            set: async (data) => {
                const tasks = [];
                for (const category in data) {
                    for (const id in data[category]) {
                        const value = data[category][id];
                        const key = `${category}-${id}`;
                        tasks.push(value ? writeData(value, key) : removeData(key));
                    }
                }
                await Promise.all(tasks);
            }
        }
    };

    const saveCreds = () => writeData(state.creds, 'creds');
    // === MONGODB AUTH SETUP END ===

  const sock = makeWASocket({
    auth: state,
    printQRInTerminal: false,
    logger: pino({ level: "info" }),
    browser: ["Mac OS", "Chrome", "121.0.6167.184"],
    connectTimeoutMs: 60000,
    defaultQueryTimeoutMs: 0,
    keepAliveIntervalMs: 10000,
  });

  sock.ev.on("creds.update", saveCreds);

  // --- Group Events: Auto-Welcome ---
  sock.ev.on("group-participants.update", async (update) => {
    if (update.action === "add") {
      const groupMetadata = await sock.groupMetadata(update.id);
      for (const participant of update.participants) {
        const welcomeMsg = `🌌 *Welcome to Beyond the Verse!*
        
नमस्ते @${participant.split("@")[0]}! आप इस ब्रह्मांडीय समुदाय का हिस्सा बन चुके हैं। 

यहाँ हम विज्ञान, दर्शन और अस्तित्व के गहरे रहस्यों पर चर्चा करते हैं। शुरू करने के लिए समूह में अपना कोई गहरा सवाल साझा करें या */help* टाइप करें। ✨`;

        await sock.sendMessage(update.id, {
          text: welcomeMsg,
          mentions: [participant],
        });
      }
    }
  });

  // --- Pairing Code Logic ---
  if (!sock.authState.creds.registered) {
    let phoneNumber = process.env.PHONE_NUMBER;
    
    if (!phoneNumber) {
      console.error("\n❌ ERROR: PHONE_NUMBER is missing in environment variables!");
      console.error("Please add PHONE_NUMBER (e.g., 91XXXXXXXXXX) to your Render.com Environment Variables.");
      return; 
    }

    phoneNumber = phoneNumber.replace(/[^0-9]/g, "");
    if (!phoneNumber.startsWith("91") && phoneNumber.length === 10) {
        phoneNumber = "91" + phoneNumber;
    }

    console.log(`\n📲 BOT IS REQUESTING CODE FOR: +${phoneNumber}`);
    
    const requestPairingCodeWithRetry = async () => {
        if (sock.authState.creds.registered || sock.pairingRequested) return;
        sock.pairingRequested = true;
        
        try {
            console.log("⏳ [1/3] Initializing secure connection...");
            await new Promise(resolve => setTimeout(resolve, 3000));
            
            console.log("⏳ [2/3] Synchronizing with WhatsApp (5s delay)...");
            await new Promise(resolve => {
                pairingTimeout = setTimeout(resolve, 5000);
            });

            if (sock.authState.creds.registered) return;
            
            console.log("📡 [3/3] Requesting pairing code from WhatsApp servers...");
            let code = await sock.requestPairingCode(phoneNumber);
            code = code?.match(/.{1,4}/g)?.join("-") || code;
            
            console.log("\n" + "⭐".repeat(20));
            console.log(`✅ YOUR RENDER PAIRING CODE: ${code}`);
            console.log("⭐".repeat(20) + "\n");
            console.log("1. Open WhatsApp > Linked Devices > Link a Device.");
            console.log("2. Select 'Link with phone number instead'.");
            console.log(`3. Enter: ${code}\n`);
        } catch (error) {
            console.error("❌ Pairing Request Failed:", error.message);
            sock.pairingRequested = false;
            console.log("🔄 Retrying in 15 seconds...");
            pairingTimeout = setTimeout(requestPairingCodeWithRetry, 15000);
        }
    };

    requestPairingCodeWithRetry();
  }

  sock.ev.on("connection.update", async (update) => {
    const { connection, lastDisconnect } = update;

    if (connection === "close") {
        isConnecting = false;
        sock.pairingRequested = false; // Reset pairing state
        if (pairingTimeout) {
            clearTimeout(pairingTimeout);
            pairingTimeout = null;
        }

      const statusCode = lastDisconnect.error?.output?.statusCode;
      const isLoggedOut = statusCode === DisconnectReason.loggedOut || statusCode === 401;
      const isConflict = lastDisconnect.error?.message?.includes("conflict") || statusCode === 440;
      
      console.log(`🔄 Connection closed (status: ${statusCode}), reconnecting...`);
      
      let delay = 5000;
      if (isConflict) {
        reconnectAttempts++;
        // Stronger backoff for conflicts: 4s, 8s, 16s, 32s, 64s... up to 5 mins
        delay = Math.min(2000 * Math.pow(2, reconnectAttempts), 300000); 
        console.warn(`⚠️ Conflict Detected: Another instance is likely running. Reconnecting in ${delay/1000}s (Attempt ${reconnectAttempts})`);
        
        if (reconnectAttempts > 10) {
            console.error("🛑 Excessive connection conflicts. Stopping bot to protect account.");
            console.error("Please ensure only ONE instance is running with these credentials.");
            return; // Stop the loop
        }
      }

      if (isLoggedOut) {
        console.log("🧹 Session Unauthorized (401/LoggedOut). Clearing MongoDB session...");
        await authCollection.deleteMany({});
        console.log("✅ Session cleared. Bot will restart fresh.");
      }
      
      setTimeout(() => connectToWhatsApp(), delay);
    } else if (connection === "open") {
        isConnecting = false;
        sock.pairingRequested = false;
        
        // Reset attempts ONLY after 1 minute of stable connection
        setTimeout(() => {
            reconnectAttempts = 0;
            console.log("✅ Connection stable for 60s. Reconnect counter reset.");
        }, 60000);

        if (pairingTimeout) {
            clearTimeout(pairingTimeout);
            pairingTimeout = null;
        }
      console.log("✅ Beyond the Verse AI CodeSandbox पर लाइव है!");
      console.log(`🤖 Bot ID: ${sock.user.id}`);
    }
  });

  // ⏰ 4. DAILY POLL SYSTEM (CRON JOB)
  cron.schedule(
    "0 6 * * *",
    async () => {
      console.log(
        "⏰ सुबह 6 बज गए हैं! 'Beyond the Verse' के लिए पोल जनरेट किया जा रहा है..."
      );

      try {
        const pollPrompt = `Create a deep, thought-provoking multiple-choice question for the 'Beyond the Verse' WhatsApp community. 
The topic MUST blend Science (quantum mechanics, neuroscience, etc.), Philosophy (existentialism, consciousness), and practical human life.

You must return ONLY a valid JSON object. Do not include markdown code blocks. Format:
{
  "question": "The thought-provoking question?",
  "options": ["Option A", "Option B", "Option C", "Option D"]
}`;

        const pollCompletion = await groq.chat.completions.create({
          messages: [{ role: "user", content: pollPrompt }],
          model: "llama-3.3-70b-versatile",
          response_format: { type: "json_object" },
        });

        let responseText = pollCompletion.choices[0]?.message?.content || "{}";
        responseText = responseText
          .replace(/```json/g, "")
          .replace(/```/g, "")
          .trim();

        const pollData = JSON.parse(responseText);
        const TARGET_GROUP_JID = "120363427798992883@g.us";

        await sock.sendMessage(TARGET_GROUP_JID, {
          poll: {
            name: `🌌 *Beyond the Verse: Daily Quest*\n\n${pollData.question}`,
            values: pollData.options,
            selectableCount: 1,
          },
        });
        console.log("✅ सुबह का पोल सफलतापूर्वक ग्रुप में भेज दिया गया है!");
      } catch (error) {
        console.error("❌ Poll Generation Error:", error);
      }
    },
    { timezone: "Asia/Kolkata" }
  );

  // ----------------------------------------------------
  // 💬 5. MESSAGE HANDLING & ADVANCED AI LOGIC
  // ----------------------------------------------------
  sock.ev.on("messages.upsert", async (m) => {
    try {
      if (!m.messages || m.messages.length === 0) return;
      const msg = m.messages[0];

      if (!msg.message || msg.key.fromMe) return;

      const senderId = msg.key.remoteJid;
      const myId = sock.user?.id;
      const myLid = sock.user?.lid;

      if (!myId) {
        console.warn("⚠️ Bot ID not available yet.");
        return;
      }

      // Extract text from text message, extended message, or image caption
      let text =
        msg.message.conversation ||
        msg.message.extendedTextMessage?.text ||
        msg.message.imageMessage?.caption ||
        "";

      const messageType = Object.keys(msg.message)[0];
      const cleanBotNumber = myId.split(":")[0].split("@")[0];
      const isGroup = senderId.endsWith("@g.us");

      // Extract Context & Mentions
      const contextInfo =
        msg.message?.extendedTextMessage?.contextInfo ||
        msg.message?.imageMessage?.contextInfo ||
        {};
      const participant = contextInfo.participant || "";
      const mentionedJids = contextInfo.mentionedJid || [];

      // Check for mentions or replies (more robustly)
      const isMentioned = mentionedJids.some(jid => 
        jid.includes(cleanBotNumber) || (myLid && jid.includes(myLid.split(":")[0]))
      );
      const isRepliedToBot = 
        participant.includes(cleanBotNumber) || (myLid && participant.includes(myLid.split(":")[0]));

      // Image detection
      const isImageMessage = !!msg.message.imageMessage;
      const isQuotedImage =
        !!msg.message.extendedTextMessage?.contextInfo?.quotedMessage
          ?.imageMessage;
      const hasImage = isImageMessage || isQuotedImage;

      // Decision: Should the bot reply?
      const shouldReply = !isGroup || isMentioned || isRepliedToBot;
      
      console.log(`📩 Message from ${senderId} [Group: ${isGroup}, Type: ${messageType}]`);
      console.log(`🔍 Mentions: [${mentionedJids}], RepliedTo: ${participant}`);
      console.log(`🔍 Status: isMentioned=${isMentioned}, isRepliedToBot=${isRepliedToBot}, shouldReply=${shouldReply}`);

      if (!shouldReply) {
        if (isGroup) console.log(`⏭️ Skipping group message from ${senderId} (no mention/reply).`);
        return;
      }

      // UX Feature: Auto-read the message when bot processes it
      try {
        await sock.readMessages([msg.key]);
      } catch (e) {
        console.error("Read Error:", e.message);
      }

      // Clean text (remove mentions)
      text = text.replace(/@\S+/g, "").trim();

      // Context from quoted messages
      const quotedMessageInfo =
        msg.message?.extendedTextMessage?.contextInfo?.quotedMessage;
      let quotedText = "";

      if (quotedMessageInfo) {
        quotedText =
          quotedMessageInfo.conversation ||
          quotedMessageInfo.extendedTextMessage?.text ||
          quotedMessageInfo.imageMessage?.caption ||
          "";
      }

      if (quotedText) {
        const commonErrors = [
          "⚠️ माफ़ करना, अभी सिस्टम में कुछ तकनीकी समस्या आ रही है।",
          "माफ़ करना, मेरे सोचने की क्षमता (API Quota) अभी खत्म हो गई है।",
        ];
        const isErrorQuoted = commonErrors.some((err) => quotedText.includes(err));

        if (!isErrorQuoted) {
          console.log(`📌 Quoted Context: ${quotedText}`);
          text = `[मैंने आपके इस पिछले मैसेज पर रिप्लाई किया है: "${quotedText}"]\n\nमेरा नया सवाल/जवाब: ${text}`;
        }
      }
      // Default greeting in group without direct text but mentioned
      if (!text && isGroup && !hasImage) {
        text = "नमस्ते! मैं 'Beyond the Verse' का AI गाइड हूँ।";
      }

      if (!text && !hasImage) {
        console.log(`⏭️ Skipping empty message from ${senderId}.`);
        return;
      }

      console.log(`💬 User (${senderId}): ${text || "[Image/Media]"}`);

      // Buffer group messages for summary
      if (isGroup && text && !text.startsWith("/")) {
        if (!groupMsgBuffer.has(senderId)) groupMsgBuffer.set(senderId, []);
        const buffer = groupMsgBuffer.get(senderId);
        const participantName = msg.pushName || msg.key.participant?.split("@")[0] || "User";
        buffer.push(`${participantName}: ${text}`);
        if (buffer.length > 20) buffer.shift(); // Keep last 20 messages
      }

      // UX Feature: React with an hourglass to indicate processing
      await sock.sendMessage(senderId, { react: { text: "⏳", key: msg.key } });

      // UX Feature: Send "typing..." status
      try {
        await sock.sendPresenceUpdate("composing", senderId);
      } catch (e) {
        console.warn("Presence Update Error:", e.message);
      }

      try {
      // 🆘 COMMAND: /help (List all commands)
      if (text.toLowerCase() === "/help") {
        const helpMessage = `🌌 *Beyond the Verse AI: Guide*
        
Welcome! I am your advanced AI companion. Here are the ways you can interact with me:

*COMMANDS:*
1.  */research [topic]* - Deep scientific/philosophical research.
2.  */search [query]* - Quick web search and synthesis.
3.  */news [topic]* - Fetch the latest news and updates.
4.  */quiz [topic]* - Generate an interactive 3-question quiz.
5.  */tts [text]* - Convert text to a voice message (Speech).
6.  */imagine [prompt]* - Generates a high-quality AI image.
7.  */summarize [URL]* - Scrapes a webpage for a philosophical TL;DR.
8.  */yt [YouTube URL]* - Downloads and sends a YouTube video directly.
9.  */fact* - Get a deep scientific or philosophical fact.
10. */ping* - Check if the bot is alive.
11. */help* - Shows this guide.

*GROUP FEATURES:*
1.  */everyone* - Tag all members (Admins only).
2.  */summarize_chat* - AI summary of recent group discussion.
3.  *Auto-Welcome:* Greets new members beautifully.

*FEATURES:*
*   *Natural Chat:* Just talk to me! I have persistent memory.
*   *Vision Analysis:* Send me any image with a question.
*   *Language:* I speak Hindi, Hinglish, and English.

*Note:* In groups, please mention me or reply to my message to get a response.`;

        await sock.sendMessage(senderId, { text: helpMessage }, { quoted: msg });
        await sock.sendMessage(senderId, { react: { text: "📖", key: msg.key } });
        return;
      }

      // 📢 COMMAND: /everyone (Tag all members)
      if (text.toLowerCase() === "/everyone" && isGroup) {
        try {
          const groupMetadata = await sock.groupMetadata(senderId);
          const participants = groupMetadata.participants.map(p => p.id);
          const mentionText = `📢 *Attention Everyone!* \n\nTagged by: @${msg.key.participant?.split("@")[0]}`;
          
          await sock.sendMessage(senderId, { text: mentionText, mentions: participants }, { quoted: msg });
          await sock.sendMessage(senderId, { react: { text: "📢", key: msg.key } });
        } catch (e) {
          console.error("Everyone Error:", e);
        }
        return;
      }

      // 📝 COMMAND: /summarize_chat (AI Summary of last 20 messages)
      if (text.toLowerCase() === "/summarize_chat" && isGroup) {
        const buffer = groupMsgBuffer.get(senderId) || [];
        if (buffer.length < 5) {
          await sock.sendMessage(senderId, { text: "⚠️ समरी बनाने के लिए अभी पर्याप्त मैसेज नहीं हैं।" }, { quoted: msg });
          return;
        }

        try {
          const response = await fetchWithTimeout("http://localhost:8080/group_summary", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ messages: buffer }),
          });
          const data = await response.json();
          await sock.sendMessage(senderId, { text: `📝 *Group Insight Summary*\n\n${data.response}` }, { quoted: msg });
          await sock.sendMessage(senderId, { react: { text: "📝", key: msg.key } });
        } catch (e) {
          console.error("Summary Error:", e);
        }
        return;
      }

      // 🧠 COMMAND: /fact
      if (text.toLowerCase() === "/fact") {
        try {
          const response = await fetchWithTimeout("http://localhost:8080/fact", {
            method: "GET",
          });
          const data = await response.json();
          await sock.sendMessage(senderId, { text: `🌌 *Beyond the Verse: Deep Fact*\n\n${data.response}` }, { quoted: msg });
          await sock.sendMessage(senderId, { react: { text: "🧠", key: msg.key } });
        } catch (e) {
          await sock.sendMessage(senderId, { text: "⚠️ तथ्य खोजने में समस्या आ रही है।" }, { quoted: msg });
        }
        return;
      }

      // 🏓 COMMAND: /ping
      if (text.toLowerCase() === "/ping") {
        await sock.sendMessage(senderId, { text: "🏓 *Pong!* I am online and ready. ✨" }, { quoted: msg });
        await sock.sendMessage(senderId, { react: { text: "⚡", key: msg.key } });
        return;
      }

      // 🎥 COMMAND: /yt or /youtube (Download YouTube Video via Python)
      if (
        text.toLowerCase().startsWith("/yt ") ||
        text.toLowerCase().startsWith("/youtube ")
      ) {
        const url = text.toLowerCase().startsWith("/yt ")
          ? text.slice(4).trim()
          : text.slice(9).trim();
        if (!url) return;

        await sock.sendMessage(
          senderId,
          {
            text: "⏳ *Downloading Video:* Please wait while I fetch your video... (Limit: 50MB)",
          },
          { quoted: msg }
        );

        try {
          const response = await fetchWithTimeout("http://localhost:8080/youtube", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
          });

          if (!response.ok) throw new Error("Download Error");

          const data = await response.json();
          const videoPath = data.path;
          const videoTitle = data.title;

          if (fs.existsSync(videoPath)) {
            await sock.sendMessage(
              senderId,
              {
                video: fs.readFileSync(videoPath),
                caption: `🎥 *Video:* ${videoTitle}`,
              },
              { quoted: msg }
            );

            // Clean up file after sending
            fs.unlinkSync(videoPath);
            await sock.sendMessage(senderId, {
              react: { text: "✅", key: msg.key },
            });
          } else {
            throw new Error("File not found after download");
          }
          return;
        } catch (e) {
          console.error(e);
          await sock.sendMessage(
            senderId,
            {
              text: "⚠️ वीडियो डाउनलोड करने में समस्या आई है। कृपया सुनिश्चित करें कि लिंक सही है और वीडियो 50MB से छोटा है।",
            },
            { quoted: msg }
          );
          await sock.sendMessage(senderId, {
            react: { text: "❌", key: msg.key },
          });
          return;
        }
      }

      // 🎨 COMMAND: /imagine (AI Image Generation)
      if (text.toLowerCase().startsWith("/imagine ")) {
        const imagePrompt = text.slice(9).trim();
        const imageUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(
          imagePrompt
        )}?width=1024&height=1024&nologo=true`;

        await sock.sendMessage(
          senderId,
          {
            image: { url: imageUrl },
            caption: `✨ ये रही आपकी तस्वीर: ${imagePrompt}`,
          },
          { quoted: msg }
        );
        // Change reaction to artistic palette
        await sock.sendMessage(senderId, {
          react: { text: "🎨", key: msg.key },
        });
        return;
      }

      // 🧬 COMMAND: /research or /search (Autonomous AI Research Agent via Python)
      if (
        text.toLowerCase().startsWith("/research ") ||
        text.toLowerCase().startsWith("/search ")
      ) {
        const query = text.toLowerCase().startsWith("/research ")
          ? text.slice(10).trim()
          : text.slice(8).trim();
        if (!query) return;

        try {
          const response = await fetchWithTimeout("http://localhost:8080/research", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query }),
          });

          if (!response.ok) throw new Error("Agent Error");

          const data = await response.json();
          const aiResponse = data.response;

          await sock.sendMessage(
            senderId,
            { text: `🧬 *Beyond the Verse: Autonomous Research*\n\n${aiResponse}` },
            { quoted: msg }
          );
          await sock.sendMessage(senderId, {
            react: { text: "🧬", key: msg.key },
          });
          return;
        } catch (e) {
          console.error(e);
          await sock.sendMessage(
            senderId,
            { text: "⚠️ रिसर्च एजेंट को डेटा जुटाने में समस्या आ रही है।" },
            { quoted: msg }
          );
          await sock.sendMessage(senderId, {
            react: { text: "❌", key: msg.key },
          });
          return;
        }
      }

      // 📜 COMMAND: /summarize (Deep Web Scraper & Summarizer via Python)
      if (text.toLowerCase().startsWith("/summarize ")) {
        const url = text.slice(11).trim();
        if (!url) return;

        try {
          const response = await fetchWithTimeout("http://localhost:8080/summarize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
          });

          if (!response.ok) throw new Error("Summarizer Error");

          const data = await response.json();
          const aiResponse = data.response;

          await sock.sendMessage(
            senderId,
            { text: `📜 *Beyond the Verse: Deep Summary*\n\n${aiResponse}` },
            { quoted: msg }
          );
          await sock.sendMessage(senderId, {
            react: { text: "📜", key: msg.key },
          });
          return;
        } catch (e) {
          console.error(e);
          await sock.sendMessage(
            senderId,
            { text: "⚠️ इस लिंक को पढ़ने में समस्या आ रही है।" },
            { quoted: msg }
          );
          await sock.sendMessage(senderId, {
            react: { text: "❌", key: msg.key },
          });
          return;
        }
      }

      // 📰 COMMAND: /news (Fetch latest news via Python)
      if (text.toLowerCase().startsWith("/news ")) {
        const topic = text.slice(6).trim();
        if (!topic) return;

        try {
          const response = await fetchWithTimeout("http://localhost:8080/news", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ topic }),
          });

          if (!response.ok) throw new Error("News Agent Error");

          const data = await response.json();
          await sock.sendMessage(
            senderId,
            { text: `📰 *Beyond the Verse: Latest News*\n\n${data.response}` },
            { quoted: msg }
          );
          await sock.sendMessage(senderId, { react: { text: "📰", key: msg.key } });
          return;
        } catch (e) {
          console.error(e);
          await sock.sendMessage(
            senderId,
            { text: "⚠️ न्यूज़ एजेंट को खबर जुटाने में समस्या आ रही है।" },
            { quoted: msg }
          );
          await sock.sendMessage(senderId, { react: { text: "❌", key: msg.key } });
          return;
        }
      }

      // ❓ COMMAND: /quiz (Generate interactive quiz via Python)
      if (text.toLowerCase().startsWith("/quiz ")) {
        const topic = text.slice(6).trim();
        if (!topic) return;

        try {
          const response = await fetchWithTimeout("http://localhost:8080/quiz", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ topic }),
          });

          if (!response.ok) throw new Error("Quiz Generator Error");

          const data = await response.json();
          await sock.sendMessage(
            senderId,
            { text: `🎯 *Beyond the Verse: Quiz Time!*\n\n${data.response}` },
            { quoted: msg }
          );
          await sock.sendMessage(senderId, { react: { text: "🎯", key: msg.key } });
          return;
        } catch (e) {
          console.error(e);
          await sock.sendMessage(
            senderId,
            { text: "⚠️ क्विज़ जनरेट करने में समस्या आ रही है।" },
            { quoted: msg }
          );
          await sock.sendMessage(senderId, { react: { text: "❌", key: msg.key } });
          return;
        }
      }

      // 🎙️ COMMAND: /tts (Convert Text to Speech Voice Note via Python)
      if (text.toLowerCase().startsWith("/tts ")) {
        const ttsText = text.slice(5).trim();
        if (!ttsText) return;

        await sock.sendMessage(senderId, { react: { text: "⏳", key: msg.key } });

        try {
          const response = await fetchWithTimeout("http://localhost:8080/tts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: ttsText }),
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(`TTS API Error: ${response.status} ${errorData.detail || ""}`);
          }

          const data = await response.json();
          const audioPath = data.path;

          if (fs.existsSync(audioPath)) {
            await sock.sendMessage(
              senderId,
              {
                audio: fs.readFileSync(audioPath),
                mimetype: 'audio/mp4', // More compatible for PTT
                ptt: true, // Send as a voice note
              },
              { quoted: msg }
            );

            // Clean up file after sending
            try {
              fs.unlinkSync(audioPath);
            } catch (unlinkError) {
              console.error("Failed to delete temporary audio file:", unlinkError);
            }
            await sock.sendMessage(senderId, { react: { text: "🎙️", key: msg.key } });
          } else {
            throw new Error(`Generated audio file not found at: ${audioPath}`);
          }
          return;
        } catch (e) {
          console.error("TTS Command Error:", e);
          await sock.sendMessage(
            senderId,
            { text: `⚠️ वॉइस जनरेट करने में समस्या आ रही है। (${e.message})` },
            { quoted: msg }
          );
          await sock.sendMessage(senderId, { react: { text: "❌", key: msg.key } });
          return;
        }
      }

      // 👁️ ADVANCED FEATURE: Vision AI (Image Analysis) via Python Backend
      if (hasImage) {
        let aiResponse = "";
        try {
          const mediaMsgObj = isImageMessage
            ? msg
            : {
                key: msg.key,
                message:
                  msg.message.extendedTextMessage.contextInfo.quotedMessage,
              };
          const buffer = await downloadMediaMessage(
            mediaMsgObj,
            "buffer",
            {},
            { reuploadRequest: sock.updateMediaMessage }
          );
          const base64Image = buffer.toString("base64");

          const prompt =
            text
              .replace(
                /\[मैंने आपके इस पिछले मैसेज पर रिप्लाई किया है: ".*"\]\n\nमेरा नया सवाल\/जवाब: /g,
                ""
              )
              .trim() ||
            "What is in this image? Explain beautifully and deeply like a Beyond the Verse guide.";

          const response = await fetchWithTimeout("http://localhost:8080/vision", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: senderId, prompt, base64_image: base64Image }),
          });
          const data = await response.json();
          aiResponse = data.response || "मुझे इस तस्वीर में कुछ खास समझ नहीं आया।";
        } catch (visionError) {
          console.error("Vision API Error:", visionError);
          aiResponse =
            "⚠️ माफ़ करना, मैं अभी इस तस्वीर का विश्लेषण नहीं कर पा रहा हूँ।";
        }

        await sock.sendMessage(
          senderId,
          { text: `👁️ *Vision Analysis*\n\n${aiResponse}` },
          { quoted: msg }
        );
        await sock.sendMessage(senderId, {
          react: { text: "✨", key: msg.key },
        });
        return;
      }

      // 🧠 NORMAL CHAT: Conversational Memory with Context via Python
      if (!userSessions.has(senderId)) {
        console.log(`🆕 ${senderId} के लिए नई मेमोरी/सेशन शुरू किया गया।`);
        userSessions.set(senderId, { history: [], summary: "" });
      }

      let session = userSessions.get(senderId);
      
      // Migration: If session is still just an array (old format), convert it
      if (Array.isArray(session)) {
          session = { history: session, summary: "" };
          userSessions.set(senderId, session);
      }

      const history = session.history || [];
      const summary = session.summary || "";

      // Clean history to remove system prompts added by the old system
      const cleanHistory = history.filter((msg) => msg.role !== "system");

      try {
        const response = await fetchWithTimeout("http://localhost:8080/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: senderId,
            message: text,
            history: cleanHistory,
            context_summary: summary
          }),
        });

        if (!response.ok) {
          throw new Error(`Python API Error: ${response.statusText}`);
        }

        const data = await response.json();
        const aiResponse =
          data.response || "माफ़ करना, मैं अभी जवाब नहीं दे पा रहा हूँ।";

        cleanHistory.push({ role: "user", content: text });
        cleanHistory.push({ role: "assistant", content: aiResponse });

        // Memory Summarization Logic (ChatGPT-like memory)
        // If history gets long, we summarize it and keep the summary, then trim history
        if (cleanHistory.length > 10) {
            console.log(`♻️ Summarizing memory for ${senderId}...`);
            try {
                const summaryResponse = await fetchWithTimeout("http://localhost:8080/summarize_memory", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        history: cleanHistory,
                        existing_summary: summary
                    }),
                });
                if (summaryResponse.ok) {
                    const summaryData = await summaryResponse.json();
                    session.summary = summaryData.summary;
                    console.log(`✅ New Memory Summary: ${session.summary}`);
                }
            } catch (sumErr) {
                console.error("Summarization failed:", sumErr);
            }
            
            // Keep only the last 4 messages for immediate context, rely on summary for the rest
            cleanHistory.splice(0, cleanHistory.length - 4);
        }

        session.history = cleanHistory;
        userSessions.set(senderId, session);
        saveSessions();

        await sock.sendMessage(senderId, { text: aiResponse }, { quoted: msg });

        // Change reaction to sparkles on success
        await sock.sendMessage(senderId, {
          react: { text: "✨", key: msg.key },
        });
        console.log(`🤖 AI: ${aiResponse}`);
      } catch (chatError) {
        throw chatError;
      }
    } catch (error) {
      console.error("❌ AI Error:", error);

      // Change reaction to cross mark on failure
      try {
        await sock.sendMessage(senderId, { react: { text: "❌", key: msg.key } });
      } catch (e) {}

      const errorMessage = error.toString().toLowerCase();

      if (
        errorMessage.includes("429") ||
        errorMessage.includes("quota") ||
        errorMessage.includes("rate limit")
      ) {
        const quotaMsg =
          "⏳ *Beyond the Verse: System Alert*\n\nमाफ़ करना, मेरे सोचने की क्षमता (API Quota) अभी खत्म हो गई है।\n\nकृपया कुछ देर बाद प्रयास करें! ✨";
        await sock.sendMessage(senderId, { text: quotaMsg }, { quoted: msg });
      } else {
        const genericMsg =
          "⚠️ माफ़ करना, अभी सिस्टम में कुछ तकनीकी समस्या आ रही है। कृपया थोड़ी देर बाद प्रयास करें।";
        await sock.sendMessage(senderId, { text: genericMsg }, { quoted: msg });
      }
    }
  } catch (globalError) {
    console.error("❌ Global Upsert Error:", globalError);
  }
});
}

// 🛡️ Global unhandled rejection handler to keep the bot alive
process.on("unhandledRejection", (reason, promise) => {
  console.error("Unhandled Rejection at:", promise, "reason:", reason);
});

// Start the bot after backend is ready
(async () => {
  await waitForBackend();
  connectToWhatsApp();
})();
