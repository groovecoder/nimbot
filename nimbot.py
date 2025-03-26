import os
from urllib.parse import urljoin, urldefrag

import requests
from bs4 import BeautifulSoup
import redis

from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS, Redis

from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI


redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
redis_index_name = "nimbox-index"


def crawl_site(start_url, max_pages=1000):
    visited = set()
    to_visit = [start_url]
    collected_docs = []
    headers = {
        "User-Agent": os.environ.get("USER_AGENT", "requests")
    }

    print("🕸 Starting crawl ...")
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            print(f"📄 Fetching {url}")
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            for a in soup.find_all("a", href=True):
                raw_link = urljoin(url, a["href"])
                clean_link, _ = urldefrag(raw_link)
                if (
                    clean_link.startswith(start_url)
                    and clean_link not in visited
                    and "#__docusaurus" not in raw_link
                ):
                    to_visit.append(raw_link)

            article = soup.find("article")
            if not article:
                print(f"⚠️ No <article> found: {url}")
                continue

            page_text = article.get_text(strip=True)
            if len(page_text) < 100:
                print(f"⏭ Skipping article shorter than 100: {url}")
                continue

            title = soup.title.string.strip() if soup.title and soup.title.string else "Unntitled"
            doc_dict = {"url": url, "title": title, "text": page_text}
            collected_docs.append(doc_dict)
            print(f"✅ Added article: {title[:60]}...")

        except Exception as e:
            print(f"❌ failed to crawl {url}: {e}")

    print(f"\n✅ Finished crawling. Collected {len(collected_docs)} pages.\n")
    return collected_docs

docs = crawl_site("https://experimenter.info/", max_pages=1000)
print("🧱 Converting crawled data into LangChain documents...")
lc_docs = [Document(page_content=doc["text"], metadata={"source": doc["url"], "title": doc["title"]}) for doc in docs]
print("✂️ Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(lc_docs)
print(f"📚 Created {len(split_docs)} chunks from {len(lc_docs)} documents.\n")

def get_vectorstore(docs, redis_url="redis://localhost:6379", index_name="nimbot-index", faiss_path="faiss_index"):
    embedding = OpenAIEmbeddings()

    # Try connecting to Redis
    try:
        r = redis.Redis.from_url(redis_url)
        r.ping()
        print("🧠 Connected to Redis, using Redis as vector store.")
        return Redis.from_documents(
            documents=docs,
            embedding=embedding,
            redis_url=redis_url,
            index_name=index_name
        )
    except redis.exceptions.ConnectionError:
        print("⚠️ Redis not available, falling back to FAISS.")

    # Check if FAISS index already exists
    if os.path.exists(faiss_path):
        print(f"📦 Loading FAISS index from {faiss_path}")
        return FAISS.load_local(faiss_path, embeddings=embedding)

    # Otherwise, build and persist a new FAISS index
    print("🔧 Creating new FAISS index...")
    faiss_store = FAISS.from_documents(documents=docs, embedding=embedding)
    faiss_store.save_local(faiss_path)
    print(f"💾 Saved FAISS index to {faiss_path}")
    return faiss_store

print("📊 Loading vector store...")
vectorstore = get_vectorstore(lc_docs, redis_url=redis_url, index_name=redis_index_name)

print("💬 Setting up the LLM + retrieval chain...\n")
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(
        search_kwargs={"distance_threshold": 0.3}
    ),
    return_source_documents=True
)

print("Nimbot is ready! Ask your questions below...\n")
while True:
    question = input("💬 Ask nimbot (or type 'exit'): ")
    if question.lower() == "exit":
        break
    response = qa_chain.invoke({"query":question})
    print("🤖 Nimbot says:\n")
    print(response["result"])
    print("🕸 Sources:")
    for doc in response["source_documents"]:
        print("-", doc.metadata.get("source"))
