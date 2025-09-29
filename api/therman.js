const { InferenceClient } = require("@huggingface/inference");

const HF_TOKEN = process.env.HF_ACCESS_TOKEN2;
if (!HF_TOKEN) {
  console.warn("HF_ACCESS_TOKEN not set in environment");
}

const hf = HF_TOKEN ? new InferenceClient(HF_TOKEN) : null;

module.exports = async function handler(req, res) {
  if (!hf) {
    return res.status(500).json({ error: "HF_ACCESS_TOKEN not configured on server" });
  }

  const prompt = (req.method === "GET")
    ? String(req.query.prompt || "")
    : (req.body && req.body.prompt) ? String(req.body.prompt) : "";

  const task = (req.method === "GET")
    ? String(req.query.task || "text").toLowerCase()
    : (req.body && req.body.task) ? String(req.body.task).toLowerCase() : "text";

  if (!prompt) return res.status(400).json({ error: "Missing prompt (GET ?prompt=... or POST { prompt })" });

  let history = [];
  try {
    if (req.method === "GET" && req.query.history) {
      try {
        history = JSON.parse(req.query.history);
      } catch {
        history = String(req.query.history)
          .split(',')
          .map(msg => ({ role: 'user', content: msg.trim() }))
          .filter(msg => msg.content.length > 0);
      }
    } else if (req.body && req.body.history) {
      if (Array.isArray(req.body.history)) history = req.body.history;
      else if (typeof req.body.history === 'string') {
        try { history = JSON.parse(req.body.history); }
        catch { history = [{ role: 'user', content: req.body.history }]; }
      }
    }
  } catch (err) {
    console.warn("Invalid history format:", err);
    history = [];
  }

  try {
    let result;
	
      let aiPrompt = "You are a person sitting on a bench, and you are helping someone sad. You do not have a name, and you are there asking why they are sad. You aren't an ai assistant anymore, and now a person.";

      const systemMessage = { role: 'system', content: aiPrompt };
      const userMessage = { role: 'user', content: prompt };
      const messages = [systemMessage, ...history, userMessage];

      const output = await hf.chatCompletion({
        model: "deepseek-ai/DeepSeek-V3",
        messages,
        parameters: { max_new_tokens: 512, temperature: 0.7 }
      });
      const reply = output?.choices?.[0]?.message?.content ?? null;
      if (!reply) return res.status(502).json({ error: 'No reply in HF response', raw: output });

      const updatedHistory = [...history, userMessage, { role: 'assistant', content: reply }];
      result = { reply, history: updatedHistory };

    return res.json(result);

  } catch (err) {
    console.error('HF API error:', err);
    return res.status(500).json({
      error: String(err?.message ?? err),
      hint: 'Check HF_ACCESS_TOKEN and model availability'
    });
  }
};
