def explanation_prompt(context, diagnosis):
    return f"""
You are an expert satellite operations engineer.

CONTEXT (retrieved documents):
{context}

DIAGNOSIS OUTPUT:
Affected Components: {diagnosis['affected_components']}
Affected Subsystems: {diagnosis['affected_subsystems']}
Ranked Root Causes: {diagnosis['ranked_root_causes']}

TASK:
1. Explain the most likely root cause in simple engineering terms.
2. Explain why this fault occurred.
3. Suggest safe and practical corrective actions.

RULES:
- Do NOT invent new faults.
- Base explanation only on the given diagnosis and context.
- Keep explanation concise and professional.
"""