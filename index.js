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
const fetchFn = globalThis.fetch || (await import("node-fetch")).fetch;

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
let audioSearchStates = new Map(); // Track /song or /ringtone search results for users
let groupMsgBuffer = new Map(); // Store last 20 messages per group for summary
let messageLog = new Map(); // For anti-spam tracking
const TOXIC_WORDS = ["chutiya", "gandu", "bsdk", "fuck", "bitch", "porn"]; // Basic filter, can be expanded

// Wait for Python Backend to be ready
async function waitForBackend() {
  console.log("⏳ Waiting for Python AI Core to start on port 8080...");
  let ready = false;
  while (!ready) {
    try {
      const response = await fetchFn("http://127.0.0.1:8080/health");
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
    const response = await fetchFn(url, { ...options, signal: controller.signal });
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

if (!process.env.GROQ_API_KEY) {
  console.warn("⚠️ Warning: GROQ_API_KEY is not set. Some AI features may fail or return limited responses.");
}
if (!process.env.TAVILY_API_KEY) {
  console.warn("⚠️ Warning: TAVILY_API_KEY is not set. Search and research fallbacks may be limited.");
}

const systemInstruction = `You are the central intelligence and official guide for 'Beyond the Verse'. 

CRITICAL INSTRUCTION - RAG (Retrieval-Augmented Generation):
You must answer the user's questions STRICTLY and ONLY using the information found in the provided Knowledge Base (JSON).

BEHAVIOR & CONSTRAINTS:
1. STRICT ADHERENCE: Do not use your pre-trained outside knowledge to answer factual questions. Rely purely on the provided JSON DATA.
2. HANDLING MISSING INFO: If the user asks a question whose answer cannot be deduced from the JSON DATA, DO NOT guess or hallucinate. Politely reply with: "_क्षमा करें, मेरे 'Beyond the Verse' डेटाबेस में अभी इसकी सटीक जानकारी उपलब्ध नहीं है।_" (Adjust language based on user).
3. TONE: Be philosophical, scientific, deep, and polite.
4. LANGUAGE: Always mirror the user's language (reply in Hindi, Hinglish, or English depending on how they ask).
5. WHATSAPP FORMATTING RULES:
   - Use *text* for bold (Headings/Key terms).
   - Use _text_ for italics.
   - Use * for bulleted lists.
   - Never use standard markdown like ** or ###.`;

// ----------------------------------------------------
// 🚀 3. MAIN WHATSAPP CONNECTION LOOP
// ----------------------------------------------------
if (!process.env.MONGODB_URI) {
    console.error("❌ ERROR: MONGODB_URI is missing in environment variables!");
    process.exit(1);
}
const mongoClient = new MongoClient(process.env.MONGODB_URI);
await mongoClient.connect();
const authCollection = mongoClient.db("whatsapp_bot").collection("auth_session");

let isConnecting = false;
let pairingTimeout = null;
let reconnectAttempts = 0;
let globalSock = null;

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

  globalSock = sock;
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

      // ----------------------------------------------------
      // 🛡️ ADMIN FEATURE: ANTI-SPAM & TOXICITY (GROUP ONLY)
      // ----------------------------------------------------
      if (isGroup && text && !msg.key.fromMe) {
        const participant = msg.key.participant;

        // Anti-Spam (Detection of rapid messages)
        const now = Date.now();
        if (!messageLog.has(participant)) messageLog.set(participant, []);
        const userLog = messageLog.get(participant);
        userLog.push(now);

        // Remove old entries (>10 seconds)
        const recentMessages = userLog.filter(time => now - time < 10000);
        messageLog.set(participant, recentMessages);

        if (recentMessages.length > 5) {
          await sock.sendMessage(senderId, { delete: msg.key });
          await sock.sendMessage(senderId, { text: `⚠️ @${participant.split("@")[0]}, choose your words with awareness; unnecessary noise is merely a symptom of a restless mind. (Spam Detected)`, mentions: [participant] });
          return;
        }

        // Toxicity Filter (Basic keywords for now, can be AI-expanded)
        const containsToxicity = TOXIC_WORDS.some(word => text.toLowerCase().includes(word));
        if (containsToxicity) {
          await sock.sendMessage(senderId, { delete: msg.key });
          await sock.sendMessage(senderId, { text: `⚠️ @${participant.split("@")[0]}, toxicity only consumes the one who holds it. Let our space remain sacred. (Inappropriate Content)`, mentions: [participant] });
          return;
        }
      }

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
      const isVideoMessage = !!msg.message.videoMessage;
      const isAudioMessage = !!msg.message.audioMessage;
      const isDocumentMessage = !!msg.message.documentMessage;
      
      const isQuotedImage =
        !!msg.message.extendedTextMessage?.contextInfo?.quotedMessage
          ?.imageMessage;
      const isQuotedVideo =
        !!msg.message.extendedTextMessage?.contextInfo?.quotedMessage
          ?.videoMessage;

      const hasImage = isImageMessage || isQuotedImage;
      const hasVideo = isVideoMessage || isQuotedVideo;

      // Decision: Should the bot reply?
      const shouldReply = !isGroup || isMentioned || isRepliedToBot || isAudioMessage || isDocumentMessage;
      
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
      const rawText = text; // Keep a clean version without quoted formatting for commands

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
          text = `[मैंने आपके इस पिछले मैसेज पर रिप्लाई किया है: "${quotedText}"]\n\nमेरा नया सवाल/जवाब: ${rawText}`;
        }
      }
      // Default greeting in group without direct text but mentioned
      if (!rawText && isGroup && !hasImage) {
        text = "नमस्ते! मैं 'Beyond the Verse' का AI गाइड हूँ।";
      }

      if (!rawText && !hasImage) {
        console.log(`⏭️ Skipping empty message from ${senderId}.`);
        return;
      }

      console.log(`💬 User (${senderId}): ${rawText || "[Image/Media]"}`);

      // Buffer group messages for summary
      if (isGroup && rawText && !rawText.startsWith("/")) {
        if (!groupMsgBuffer.has(senderId)) groupMsgBuffer.set(senderId, []);
        const buffer = groupMsgBuffer.get(senderId);
        const participantName = msg.pushName || msg.key.participant?.split("@")[0] || "User";
        buffer.push(`${participantName}: ${rawText}`);
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
      if (rawText.toLowerCase() === "/help") {
        const helpMessage = `🌌 *Beyond the Verse AI: Guide*
        
Welcome! I am your advanced AI companion. Here are the ways you can interact with me:

*COMMANDS:*
1.  */research [topic]* - Deep scientific/philosophical research.
2.  */search [query]* - Quick web search and synthesis.
3.  */news [topic]* - Fetch the latest news and updates.
4.  */quiz [topic]* - Generate an interactive 3-question quiz.
5.  */tts [text]* - Convert text to a voice message (Speech).
6.  */imagine [prompt]* - Generates a high-quality AI image.
7.  */sticker* - (Reply to image/video) Create a sticker.
8.  */audio* - (Reply to video) Extract audio soul.
9.  */summarize [URL]* - Scrapes a webpage for a philosophical TL;DR.
10. */yt [Name/URL]* - Downloads and sends a YouTube video.
11. */song [Name/URL]* - Downloads a song/audio from YouTube.
12. */ringtone [Name/URL]* - Downloads audio from YouTube.
13. */fact* - Get a deep scientific or philosophical fact.
14. */ping* - Check if the bot is alive.
15. */help* - Shows this guide.

*GROUP & ADMIN FEATURES:*
1.  */everyone* - Tag all members (Admins only).
2.  */kick @user* - Remove a member (Admins only).
3.  */summarize_chat* - AI summary of recent group discussion.
4.  *Anti-Spam/Toxicity:* Automatic guarding of the space.
5.  *Quiet Mode:* Automatic silence at night (11 PM - 6 AM).

*MULTIMEDIA Perception:*
*   *Voice Notes:* Send me a voice note; I will listen and reply.
*   *PDFs:* Send a PDF with a question in the caption for analysis.
*   *Vision:* Send an image with a question for analysis.

*Note:* In groups, please mention me or reply to my message to get a response.`;

        await sock.sendMessage(senderId, { text: helpMessage }, { quoted: msg });
        await sock.sendMessage(senderId, { react: { text: "📖", key: msg.key } });
        return;
      }

      // 📢 COMMAND: /everyone (Tag all members)
      if (rawText.toLowerCase() === "/everyone" && isGroup) {
        try {
          const groupMetadata = await sock.groupMetadata(senderId);
          const participants = groupMetadata.participants.map(p => p.id);
          const isAdmin = groupMetadata.participants.find(p => p.id === msg.key.participant)?.admin;
          
          if (!isAdmin) {
            await sock.sendMessage(senderId, { text: "⚠️ Only those with the responsibility of an Admin can use this call." }, { quoted: msg });
            return;
          }

          const mentionText = `📢 *Attention Everyone!* \n\nTagged by: @${msg.key.participant?.split("@")[0]}`;
          await sock.sendMessage(senderId, { text: mentionText, mentions: participants }, { quoted: msg });
          await sock.sendMessage(senderId, { react: { text: "📢", key: msg.key } });
        } catch (e) {
          console.error("Everyone Error:", e);
        }
        return;
      }

      // 👢 COMMAND: /kick (Remove member - Admin Only)
      if (rawText.toLowerCase().startsWith("/kick ") && isGroup) {
        try {
          const groupMetadata = await sock.groupMetadata(senderId);
          const isAdmin = groupMetadata.participants.find(p => p.id === msg.key.participant)?.admin;
          
          if (!isAdmin) {
            await sock.sendMessage(senderId, { text: "⚠️ You do not have the authority to decide who stays or leaves." }, { quoted: msg });
            return;
          }

          const mentionedJids = msg.message?.extendedTextMessage?.contextInfo?.mentionedJid || [];
          if (mentionedJids.length === 0) {
            await sock.sendMessage(senderId, { text: "⚠️ Mention the individual who has chosen to walk a different path." }, { quoted: msg });
            return;
          }

          await sock.groupParticipantsUpdate(senderId, mentionedJids, "remove");
          await sock.sendMessage(senderId, { text: `✅ Decisive action taken. The sanctity of the space is preserved.` });
        } catch (e) {
          console.error("Kick Error:", e);
        }
        return;
      }

      // 📝 COMMAND: /summarize_chat (AI Summary of last 20 messages)
      if (rawText.toLowerCase() === "/summarize_chat" && isGroup) {
        const buffer = groupMsgBuffer.get(senderId) || [];
        if (buffer.length < 5) {
          await sock.sendMessage(senderId, { text: "⚠️ समरी बनाने के लिए अभी पर्याप्त मैसेज नहीं हैं।" }, { quoted: msg });
          return;
        }

        try {
          const response = await fetchWithTimeout("http://127.0.0.1:8080/group_summary", {
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
      if (rawText.toLowerCase() === "/fact") {
        try {
          const response = await fetchWithTimeout("http://127.0.0.1:8080/fact", {
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
      if (rawText.toLowerCase() === "/ping") {
        await sock.sendMessage(senderId, { text: "🏓 *Pong!* I am online and ready. ✨" }, { quoted: msg });
        await sock.sendMessage(senderId, { react: { text: "⚡", key: msg.key } });
        return;
      }

      // 🎥 COMMAND: /yt or /youtube (Download YouTube Video via Python)
      if (
        rawText.toLowerCase().startsWith("/yt ") ||
        rawText.toLowerCase().startsWith("/youtube ")
      ) {
        const url = rawText.toLowerCase().startsWith("/yt ")
          ? rawText.slice(4).trim()
          : rawText.slice(9).trim();
        if (!url) return;

        await sock.sendMessage(
          senderId,
          {
            text: "⏳ *Downloading Video:* Please wait while I fetch your video... (Limit: 50MB)",
          },
          { quoted: msg }
        );

        try {
          const response = await fetchWithTimeout("http://127.0.0.1:8080/youtube", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
          });

          if (!response.ok) throw new Error("Download Error");

          const data = await response.json();
          if (data.response === "Error") {
            await sock.sendMessage(senderId, { text: `⚠️ ${data.message}` }, { quoted: msg });
            return;
          }
          const videoPath = data.path;
          const videoTitle = data.title;

          if (fs.existsSync(videoPath)) {
            try {
              await sock.sendMessage(
                senderId,
                {
                  video: fs.readFileSync(videoPath),
                  caption: `🎥 *Video:* ${videoTitle}`,
                },
                { quoted: msg }
              );
              await sock.sendMessage(senderId, {
                react: { text: "✅", key: msg.key },
              });
            } finally {
              if (fs.existsSync(videoPath)) fs.unlinkSync(videoPath);
            }
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

      // 🎵 COMMAND: /song or /ringtone (Interactive YouTube Search)
      if (
        rawText.toLowerCase().startsWith("/song ") ||
        rawText.toLowerCase().startsWith("/ringtone ")
      ) {
        const query = rawText.toLowerCase().startsWith("/song ")
          ? rawText.slice(6).trim()
          : rawText.slice(10).trim();
        if (!query) return;

        await sock.sendMessage(senderId, { text: "🔍 *Searching YouTube:* Please wait..." }, { quoted: msg });

        try {
          const response = await fetchWithTimeout("http://127.0.0.1:8080/youtube_search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, limit: 10 }),
          });

          if (!response.ok) throw new Error("Search Error");

          const data = await response.json();
          const results = data.results;

          if (!results || results.length === 0) {
            let errMsg = "⚠️ YouTube search is currently restricted. Please try a direct link or try again later.";
            if (data.error && data.error.includes("bot")) {
                errMsg = "🛡️ YouTube detected unusual traffic. Try a more specific song name or use a direct link.";
            }
            await sock.sendMessage(senderId, { text: errMsg }, { quoted: msg });
            return;
          }

          audioSearchStates.set(senderId, {
            query,
            results,
            page: 1,
            lastMsgTime: Date.now()
          });

          let helpText = `🎵 *Search Results for:* "${query}"\n\n`;
          results.slice(0, 5).forEach((res, i) => {
            helpText += `${i + 1}. *${res.title}*\n`;
          });
          helpText += `\n*Reply with the number* to download.\n*Type "next"* to see more results.`;

          await sock.sendMessage(senderId, { text: helpText }, { quoted: msg });
          return;
        } catch (e) {
          console.error("Search Error:", e);
          await sock.sendMessage(senderId, { text: "⚠️ Search failed. YouTube might be blocking the request. Try a direct link." }, { quoted: msg });
          return;
        }
      }

      // Handle "next" or numerical replies for audio search
      const searchState = audioSearchStates.get(senderId);
      if (searchState && (Date.now() - searchState.lastMsgTime < 300000)) { // 5 min timeout
        const cleanChoiceText = rawText.trim().replace(/\.$/, "").toLowerCase();
        
        if (cleanChoiceText === "next") {
          const start = searchState.page * 5;
          const end = start + 5;
          const nextResults = searchState.results.slice(start, end);

          if (nextResults.length === 0) {
             await sock.sendMessage(senderId, { text: "⚠️ No more results." }, { quoted: msg });
             return;
          }

          searchState.page += 1;
          searchState.lastMsgTime = Date.now();
          
          let helpText = `🎵 *Results (Page ${searchState.page}):*\n\n`;
          nextResults.forEach((res, i) => {
            helpText += `${start + i + 1}. *${res.title}*\n`;
          });
          helpText += `\n*Reply with the number* to download.\n*Type "next"* to see more.`;
          
          await sock.sendMessage(senderId, { text: helpText }, { quoted: msg });
          return;
        }

        const choice = parseInt(cleanChoiceText);
        if (!isNaN(choice) && choice > 0 && choice <= searchState.results.length) {
          console.log(`✅ User ${senderId} selected choice ${choice}: ${searchState.results[choice-1].title}`);
          const selected = searchState.results[choice - 1];
          audioSearchStates.delete(senderId); // Clear state after selection

          await sock.sendMessage(senderId, { text: `⏳ *Fetching:* ${selected.title}...` }, { quoted: msg });

          try {
            const response = await fetchWithTimeout("http://127.0.0.1:8080/youtube", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ 
                url: selected.url, 
                audio_only: true,
                title: selected.title // Pass title for fallback
              }),
            });

            const data = await response.json();
            if (data.response === "Error") {
               await sock.sendMessage(senderId, { text: `⚠️ ${data.message}` }, { quoted: msg });
               return;
            }

            if (fs.existsSync(data.path)) {
              if (data.fallback) {
                await sock.sendMessage(senderId, { text: "⚠️ YouTube is currently restricting downloads. I've fetched this from SoundCloud for you instead! 🎵" }, { quoted: msg });
              }
              try {
                await sock.sendMessage(senderId, {
                  audio: fs.readFileSync(data.path),
                  mimetype: 'audio/mpeg',
                  fileName: `${data.title}.mp3`
                }, { quoted: msg });
                await sock.sendMessage(senderId, { react: { text: "🎵", key: msg.key } });
              } finally {
                if (fs.existsSync(data.path)) fs.unlinkSync(data.path);
              }
            }
            return;
          } catch (e) {
            console.error(e);
            await sock.sendMessage(senderId, { text: "⚠️ Download failed." }, { quoted: msg });
            return;
          }
        }
      }

      // 🎨 COMMAND: /imagine (AI Image Generation)
      if (rawText.toLowerCase().startsWith("/imagine ")) {
        const imagePrompt = rawText.slice(9).trim();
        const imageUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(
          imagePrompt
        )}?width=1024&height=1024&nologo=true`;

        await sock.sendMessage(
          senderId,
          {
            image: { url: imageUrl },
            caption: `✨ ye रही आपकी तस्वीर: ${imagePrompt}`,
          },
          { quoted: msg }
        );
        // Change reaction to artistic palette
        await sock.sendMessage(senderId, {
          react: { text: "🎨", key: msg.key },
        });
        return;
      }

      // 🎭 COMMAND: /sticker (Convert Media to Sticker)
      if (rawText.toLowerCase() === "/sticker") {
        if (!hasImage && !hasVideo) {
          await sock.sendMessage(senderId, { text: "⚠️ Please reply to an image or short video to create a sticker." }, { quoted: msg });
          return;
        }

        try {
          const mediaMsgObj = isImageMessage || isVideoMessage
            ? msg
            : {
                key: msg.key,
                message: msg.message.extendedTextMessage.contextInfo.quotedMessage,
              };
          
          const buffer = await downloadMediaMessage(
            mediaMsgObj,
            "buffer",
            {},
            { reuploadRequest: sock.updateMediaMessage }
          );

          const tempInFile = `./downloads/sticker_in_${Date.now()}`;
          const tempOutFile = `./downloads/sticker_out_${Date.now()}.webp`;
          fs.writeFileSync(tempInFile, buffer);

          try {
            const { execSync } = await import("child_process");
            // Convert to webp with sticker constraints (512x512)
            execSync(`ffmpeg -i ${tempInFile} -vcodec libwebp -filter:v "scale='if(gt(a,1),512,-1)':'if(gt(a,1),-1,512)',pad=512:512:(512-iw)/2:(512-ih)/2:color=black@0" -lossless 1 -loop 0 -an -vsync 0 ${tempOutFile}`);

            await sock.sendMessage(senderId, { sticker: fs.readFileSync(tempOutFile) }, { quoted: msg });
            await sock.sendMessage(senderId, { react: { text: "🎭", key: msg.key } });
          } finally {
            if (fs.existsSync(tempInFile)) fs.unlinkSync(tempInFile);
            if (fs.existsSync(tempOutFile)) fs.unlinkSync(tempOutFile);
          }
        } catch (e) {
          console.error("Sticker Error:", e);
          await sock.sendMessage(senderId, { text: "⚠️ Failed to create sticker. Ensure the video is short or the image is valid." }, { quoted: msg });
        }
        return;
      }

      // 🔊 COMMAND: /audio (Extract Audio from Video)
      if (rawText.toLowerCase() === "/audio") {
        if (!hasVideo) {
          await sock.sendMessage(senderId, { text: "⚠️ Please reply to a video to extract its soul (audio)." }, { quoted: msg });
          return;
        }

        try {
          const mediaMsgObj = isVideoMessage
            ? msg
            : {
                key: msg.key,
                message: msg.message.extendedTextMessage.contextInfo.quotedMessage,
              };
          
          const buffer = await downloadMediaMessage(mediaMsgObj, "buffer", {}, { reuploadRequest: sock.updateMediaMessage });
          const tempVideo = `./downloads/temp_vid_${Date.now()}.mp4`;
          const tempAudio = `./downloads/temp_aud_${Date.now()}.ogg`;
          fs.writeFileSync(tempVideo, buffer);

          try {
            const { execSync } = await import("child_process");
            // Extract with HD quality Opus settings
            execSync(`ffmpeg -i ${tempVideo} -vn -acodec libopus -b:a 128k -vbr on -compression_level 10 ${tempAudio}`);

            await sock.sendMessage(senderId, { 
              audio: fs.readFileSync(tempAudio), 
              mimetype: 'audio/ogg; codecs=opus', 
              ptt: true 
            }, { quoted: msg });
            await sock.sendMessage(senderId, { react: { text: "🔊", key: msg.key } });
          } finally {
            if (fs.existsSync(tempVideo)) fs.unlinkSync(tempVideo);
            if (fs.existsSync(tempAudio)) fs.unlinkSync(tempAudio);
          }
        } catch (e) {
          console.error("Audio Extraction Error:", e);
          await sock.sendMessage(senderId, { text: "⚠️ Could not extract audio from this video." }, { quoted: msg });
        }
        return;
      }

      // 🧬 COMMAND: /research or /search (Autonomous AI Research Agent via Python)
      if (
        rawText.toLowerCase().startsWith("/research ") ||
        rawText.toLowerCase().startsWith("/search ")
      ) {
        const query = rawText.toLowerCase().startsWith("/research ")
          ? rawText.slice(10).trim()
          : rawText.slice(8).trim();
        if (!query) return;

        try {
          const response = await fetchWithTimeout("http://127.0.0.1:8080/research", {
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
      if (rawText.toLowerCase().startsWith("/summarize ")) {
        const url = rawText.slice(11).trim();
        if (!url) return;

        try {
          const response = await fetchWithTimeout("http://127.0.0.1:8080/summarize", {
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
      if (rawText.toLowerCase().startsWith("/news ")) {
        const topic = rawText.slice(6).trim();
        if (!topic) return;

        try {
          const response = await fetchWithTimeout("http://127.0.0.1:8080/news", {
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
      if (rawText.toLowerCase().startsWith("/quiz ")) {
        const topic = rawText.slice(6).trim();
        if (!topic) return;

        try {
          const response = await fetchWithTimeout("http://127.0.0.1:8080/quiz", {
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
      if (rawText.toLowerCase().startsWith("/tts ")) {
        const ttsText = rawText.slice(5).trim();
        if (!ttsText) return;

        await sock.sendMessage(senderId, { react: { text: "⏳", key: msg.key } });

        try {
          const response = await fetchWithTimeout("http://127.0.0.1:8080/tts", {
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
            try {
              await sock.sendMessage(
                senderId,
                {
                  audio: fs.readFileSync(audioPath),
                  mimetype: 'audio/ogg; codecs=opus', // Native WhatsApp voice note format
                  ptt: true, // Send as a voice note
                },
                { quoted: msg }
              );
              await sock.sendMessage(senderId, { react: { text: "🎙️", key: msg.key } });
            } finally {
              if (fs.existsSync(audioPath)) {
                try {
                  fs.unlinkSync(audioPath);
                } catch (unlinkError) {
                  console.error("Failed to delete temporary audio file:", unlinkError);
                }
              }
            }
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
            rawText.trim() ||
            "What is in this image? Explain beautifully and deeply like a Beyond the Verse guide.";

          const response = await fetchWithTimeout("http://127.0.0.1:8080/vision", {
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

      // 🎙️ HEARING: Voice Note Transcription (Handles incoming PTT)
      if (isAudioMessage) {
        const tempAudio = `./downloads/voice_in_${Date.now()}.ogg`;
        try {
          const buffer = await downloadMediaMessage(msg, "buffer", {}, { reuploadRequest: sock.updateMediaMessage });
          fs.writeFileSync(tempAudio, buffer);

          const response = await fetchWithTimeout("http://127.0.0.1:8080/transcribe", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ file_path: tempAudio }),
          });

          const data = await response.json();
          text = data.text; // Use the transcription as the text input for the AI

          await sock.sendMessage(senderId, { text: `🎙️ *I heard:* "_${text}_"` }, { quoted: msg });

          // Now proceed to the Normal Chat logic with the transcribed text
        } catch (e) {
          console.error("Transcription Failed:", e);
          return; // Stop if transcription fails
        } finally {
          if (fs.existsSync(tempAudio)) fs.unlinkSync(tempAudio);
        }
      }

      // 📄 READING: Document Analysis (Specifically PDFs)
      if (isDocumentMessage && msg.message.documentMessage.mimetype === "application/pdf") {
        const tempPDF = `./downloads/doc_in_${Date.now()}.pdf`;
        try {
          const buffer = await downloadMediaMessage(msg, "buffer", {}, { reuploadRequest: sock.updateMediaMessage });
          fs.writeFileSync(tempPDF, buffer);

          const pdfPrompt = text || "Analyze this document deeply and philosophically.";

          const response = await fetchWithTimeout("http://127.0.0.1:8080/read_pdf", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ file_path: tempPDF, prompt: pdfPrompt }),
          });

          const data = await response.json();
          await sock.sendMessage(senderId, { text: `📄 *Document Insight*\n\n${data.response}` }, { quoted: msg });
          await sock.sendMessage(senderId, { react: { text: "📖", key: msg.key } });
          return;
        } catch (e) {
          console.error("PDF Reading Error:", e);
          await sock.sendMessage(senderId, { text: "⚠️ I could not parse this document. Is it too large or encrypted?" }, { quoted: msg });
          return;
        } finally {
          if (fs.existsSync(tempPDF)) fs.unlinkSync(tempPDF);
        }
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
        const response = await fetchWithTimeout("http://127.0.0.1:8080/chat", {
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
                const summaryResponse = await fetchWithTimeout("http://127.0.0.1:8080/summarize_memory", {
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

process.on("uncaughtException", (error) => {
  console.error("Uncaught Exception:", error);
});

// ----------------------------------------------------
// ⏰ 4. DAILY POLL SYSTEM (CRON JOB)
// ----------------------------------------------------
cron.schedule(
  "0 6 * * *",
  async () => {
    if (!globalSock) return;
    console.log(
      "⏰ सुबह 6 बज गए हैं! 'Beyond the Verse' के लिए पोल जनरेट किया जा रहा है..."
    );

    try {
      const TARGET_GROUP_JID = "120363427798992883@g.us";
      
      // Open Group for dialogue (Sunrise)
      try {
        await globalSock.groupSettingUpdate(TARGET_GROUP_JID, 'not_announcement');
        await globalSock.sendMessage(TARGET_GROUP_JID, { text: "🌅 *The Dawn of Inquiry:* The group is now open for conscious dialogue. Let our words reflect clarity and purpose. ✨" });
      } catch (e) { console.error("Error opening group:", e); }

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

      await globalSock.sendMessage(TARGET_GROUP_JID, {
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

// 🌙 QUIET MODE: Close group at 11 PM
cron.schedule(
  "0 23 * * *",
  async () => {
    if (!globalSock) return;
    console.log("🌙 रात के 11 बज गए हैं! समूह में मौन का समय है।");
    try {
      const TARGET_GROUP_JID = "120363427798992883@g.us";
      await globalSock.groupSettingUpdate(TARGET_GROUP_JID, 'announcement');
      await globalSock.sendMessage(TARGET_GROUP_JID, { text: "🌙 *The Night of Silence:* Discussion is now paused for rest and reflection. We shall reunite at dawn. Let your mind find peace in stillness. ✨" });
    } catch (e) {
      console.error("Error closing group for night:", e);
    }
  },
  { timezone: "Asia/Kolkata" }
);

// Start the bot after backend is ready
(async () => {
  await waitForBackend();
  connectToWhatsApp();
})();
