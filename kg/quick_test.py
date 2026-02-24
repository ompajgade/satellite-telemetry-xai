import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from kg.reasoning import diagnose_root_cause

result = diagnose_root_cause(["CADC0872", "CADC0873"])

print(result)