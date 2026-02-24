import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from kg.reasoning import diagnose_root_cause
from rag.retriever import explain_with_rag

diagnosis = diagnose_root_cause(["CADC0872"])
explanation = explain_with_rag(diagnosis)

print("\n=== EXPLAINABLE DIAGNOSIS ===\n")
print(explanation)