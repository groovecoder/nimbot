import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.nimbot_core import build_qa_chains

print("🔁 Running Nimbot refresh job...")
build_qa_chains()
print("✅ Nimbot vector store updated!")
