import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.nimbot_core import build_qa_chain

print("🧠 Loading Nimbot... (this may take a moment)")
qa_chain = build_qa_chain()

print("🤖 Nimbot is ready! Ask your questions below...\n")

while True:
    question = input("💬 Ask nimbot (or type 'exit'): ")
    if question.lower() == "exit":
        print("👋 Bye!")
        break

    try:
        response = qa_chain.invoke({"query": question})
        print("\n🤖 Nimbot says:\n")
        print(response["result"])

        print("\n🕸 Sources:")
        for doc in response["source_documents"]:
            print("-", doc.metadata.get("source"))
        print("\n" + "-" * 60 + "\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
