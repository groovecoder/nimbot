import os
from urllib.parse import urljoin, urldefrag

import requests
from bs4 import BeautifulSoup
import redis

from langchain.chains import RetrievalQA, StuffDocumentsChain, LLMChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Redis
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI


# configs from env
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
start_url = os.environ.get("START_URL", "https://experimenter.info/")
user_agent = os.environ.get("USER_AGENT", "nimbot/1.0 (+https://github.com/groovecoder/nimbot)")
max_pages = int(os.environ.get("MAX_PAGES", 1000))

# static configs
redis_index_name = "nimbot-index"
faiss_path="faiss_index"
headers = {"User-Agent": user_agent}

# qa chains
qa_chain_35 = None
qa_chain_4 = None

def crawl_site(start_url):
    visited = set()
    to_visit = [start_url]
    collected_docs = []

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

            title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"
            doc_dict = {"url": url, "title": title, "text": page_text}
            collected_docs.append(doc_dict)
            print(f"✅ Added article: {title[:60]}...")

        except Exception as e:
            print(f"❌ failed to crawl {url}: {e}")

    print(f"\n✅ Finished crawling. Collected {len(collected_docs)} pages.\n")
    return collected_docs


def clear_redis_index(redis_client, index_name):
    print(f"🧹 Clearing Redis index: {index_name}")
    pattern = f"doc:{index_name}:*"
    keys = redis_client.keys(pattern)
    if keys:
        redis_client.delete(*keys)
        print(f"🗑️ Deleted {len(keys)} old vectors from Redis.")
    else:
        print("📭 No existing vectors to delete.")


def get_vectorstore(docs):
    embedding = OpenAIEmbeddings()

    # Try connecting to Redis
    try:
        r = redis.Redis.from_url(redis_url)
        r.ping()
        print("🧠 Connected to Redis, using Redis as vector store.")
        clear_redis_index(r, redis_index_name)
        return Redis.from_documents(
            documents=docs,
            embedding=embedding,
            redis_url=redis_url,
            index_name=redis_index_name
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


def build_combine_chain_gpt35():
    # Format each document as raw text
    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )

    # Prompt that wraps all retrieved chunks
    combine_prompt = PromptTemplate.from_template("""
You are a helpful assistant answering questions about Firefox Experimenter.

Answer the question using only the context below. If the answer cannot be found in the context, say "I don't know."

Context:
{context}

Question: {question}
""")

    llm_chain = LLMChain(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        prompt=combine_prompt
    )

    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name="context"
    )


def build_qa_chains():
    global qa_chain_35, qa_chain_4
    doc_dicts = crawl_site("https://experimenter.info/")
    print("🧱 Converting crawled data into LangChain documents...")
    lc_docs = [Document(page_content=doc["text"], metadata={"source": doc["url"], "title": doc["title"]}) for doc in doc_dicts]
    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(lc_docs)
    print(f"📚 Created {len(split_docs)} chunks from {len(lc_docs)} documents.\n")
    print("📊 Loading vector store...")
    vectorstore = get_vectorstore(lc_docs)

    print("💬 Setting up the LLM + retrieval chain...\n")
    # gpt3.5 is cheaper: get more chunks even with lower similarity
    qa_chain_35 = build_combine_chain_gpt35()

    # gpt4 💸: get fewer chunks with higher similarity
    retriever4 = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "distance_threshold": 0.2
        }
    )
    qa_chain_4 = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=retriever4,
        return_source_documents=True
    )

    return qa_chain_35, qa_chain_4

def invoke_with_fallback(query, force_gpt4=False, score_threshold=0.75, max_docs=5):
    if force_gpt4:
        print("⚡️ Forcing GPT-4 ...")
        return qa_chain_4.invoke({"query": query})

    print("🧠 Running similarity search with scores ...")
    results = qa_chain_4.retriever.vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k=max_docs
    )

    if not results:
        print("⚠️ No relevant documents found - using GPT-4.")
        return qa_chain_4.invoke({"query": query})

    docs_with_scores = [(doc, score) for doc, score in results if score >= score_threshold]

    if not docs_with_scores:
        print("⚠️ All retrieved chunks had low similarity - using GPT-4.")
        return qa_chain_4.invoke({"query": query})

    print(f"✅ {len(docs_with_scores)} documents passed relevance threshold (≥ {score_threshold})")

    context_docs = [doc for doc, _ in docs_with_scores]
    response_35 = qa_chain_35.run(input_documents=context_docs, question=query)

    fallback_phrases = ["Unfortunately", "I'm not sure", "I don't know", "cannot find", "not enough context", "do not have specific", "no specific"]
    low_confidence = any(phrase in response_35.lower() for phrase in fallback_phrases) or len(response_35.strip()) < 40

    if low_confidence:
        print("🪂 GPT-3.5 was unsure, retrying with GPT-4...")
        return qa_chain_4.invoke({"query": query})

    return {
        "result": response_35,
        "source_documents": context_docs
    }
