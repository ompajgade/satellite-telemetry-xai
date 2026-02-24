from collections import defaultdict
from kg.adcs_graph import build_adcs_graph
from kg.channel_mapping import CHANNEL_MAPPING
from kg.fault_rules import FAULT_RULES

def diagnose_root_cause(anomalous_channels):
    G = build_adcs_graph()

    fault_scores = defaultdict(float)
    affected_components = set()
    affected_subsystems = set()

    # Step 1: Channel → Component → Subsystem
    for ch in anomalous_channels:
        if ch not in CHANNEL_MAPPING:
            continue

        component = CHANNEL_MAPPING[ch]["component"]
        subsystem = CHANNEL_MAPPING[ch]["subsystem"]

        affected_components.add(component)
        affected_subsystems.add(subsystem)

        # Component-level faults
        for fault, weight in FAULT_RULES.get(component, {}).items():
            fault_scores[fault] += weight

    # Step 2: Subsystem-level escalation
    if len(affected_components) > 1:
        for subsystem in affected_subsystems:
            for fault, weight in FAULT_RULES.get(subsystem, {}).items():
                fault_scores[fault] += weight

    # Normalize scores
    total = sum(fault_scores.values())
    if total > 0:
        for k in fault_scores:
            fault_scores[k] /= total

    # Sort results
    ranked_faults = sorted(
        fault_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "affected_components": list(affected_components),
        "affected_subsystems": list(affected_subsystems),
        "ranked_root_causes": ranked_faults
    }