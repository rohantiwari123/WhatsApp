import "dotenv/config";
import Groq from "groq-sdk";

// 1. Groq का सेटअप (यह .env फाइल से GROQ_API_KEY उठाएगा)
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

async function listGroqModels() {
  console.log("🔍 Groq सर्वर पर उपलब्ध मॉडल्स की लिस्ट खोजी जा रही है...");

  try {
    const models = await groq.models.list();

    // मॉडल्स के नाम साफ़ तरीके से निकालना
    const modelList = models.data.map((m) => m.id);

    console.log(
      "✅ 'Beyond the Verse' के लिए आप इन मॉडल्स का उपयोग कर सकते हैं:\n"
    );
    console.log(modelList);

    console.log(
      "\n💡 सलाह: सबसे बेस्ट रिस्पॉन्स के लिए 'llama3-70b-8192' का इस्तेमाल करें।"
    );
  } catch (error) {
    console.error(
      "❌ Error: शायद आपकी Groq API Key गलत है या लोड नहीं हो रही।"
    );
    console.error(error.message);
  }
}

listGroqModels();
