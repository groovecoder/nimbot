# 🧠 Nimbot

**Nimbot** is a lightweight, crawl-and-chat tool that scrapes content from [https://experimenter.info](https://experimenter.info), chunks it, embeds it, and lets you ask questions about it via a local chatbot powered by OpenAI + LangChain.

It supports Redis for vector storage but can fall back to FAISS if Redis isn't available — making it fast, flexible, and offline-friendly.

---

## 🚀 Features

- 🌐 Crawls an entire website (starting with `https://experimenter.info`)
- 📰 Extracts `<article>` content only
- ✂️ Splits content into clean, overlapping chunks
- 🔍 Embeds with OpenAI and stores vectors in Redis or FAISS
- 💬 Chat interface using GPT-4 + RetrievalQA
- 🧠 Remembers URLs + titles for context-rich answers
- 🕸 Smart crawler filters out low-value or duplicate links

---

## 📦 Requirements
### 📝 Environment Variables (required)
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 📝 Environment Variables (optional)
```bash
export USER_AGENT="nimbot/1.0 (+https://github.com/groovecoder/nimbot)"
export REDIS_URL="redis://localhost:6379"
```

### 🐍 Python
```bash
pip install -r requirements.txt
```

## 🧰 Usage
```bash
python nimbot.py
```

You'll see output like:
```
🕸 Starting crawl ...
📄 Fetching https://experimenter.info/docs/targeting
✅ Added article: Targeting users in Firefox experiments...

Nimbot is ready! Ask your questions below...

💬 Ask nimbot (or type 'exit'):
```

Type in a question you might ask in #ask-experimenter like:
```
So when I go to Release, I can run an experiment with a small fixed % of users to determine the "winning" branch of the 5 branches. And then I thought I saw a "convert to roll-out" option in Experimenter?
```

And Nimbot will respond with an answer based on crawled context, along with the original source URLs.

## 🔧 Redis Setup (optional but recommended)
To use Redis as a vector store, run it locally:
```bash
docker run -p 6379:6379 redis/redis-stack-server:latest
```
Or set `REDIS_URL` in your environment to point to a Redis instance.

If Redis is not available, Nimbot will fallback to using FAISS with a local index stored in `faiss_index/`.
