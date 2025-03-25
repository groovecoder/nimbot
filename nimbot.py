import os
from urllib.parse import urljoin, urldefrag

from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

import requests
from bs4 import BeautifulSoup

def crawl_site(start_url, max_pages=60):
    visited = set()
    to_visit = [start_url]
    collected_docs = []
    headers = {
        "User-Agent": os.environ.get("USER_AGENT", "requests")
    }

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            print(f"Fetching {url}")
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
                print(f"No <article> found: {url}")
                continue

            page_text = article.get_text(strip=True)
            if len(page_text) < 100:
                print(f"Skipping article shorter than 100: {url}")
                continue

            collected_docs.append({"url": url, "text": page_text})

        except Exception as e:
            print(f"failed to crawl {url}: {e}")

    collected_urls = [doc["url"] for doc in collected_docs]
    print(f"Collected urls: {collected_urls}")
    return collected_docs

docs = crawl_site("https://experimenter.info/", max_pages=100)

"""
for doc in docs:
    print(doc["url"])
    print(doc["text"][:200])
"""

lc_docs = [Document(page_content=doc["text"], metadata={"source": doc["url"]}) for doc in docs]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(lc_docs)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

question = input("Ask nimbot: ")
print(qa_chain.invoke(question))
