"""
Compare old vs new prompt quality on 3 test queries.
Old prompts: taken from git HEAD (inline strings in nodes.py)
New prompts: loaded from prompts/*.txt files
"""
import asyncio
import textwrap
import time
from datetime import datetime

from utils.call_llm_async import call_llm_async

# ── OLD PROMPTS (from git HEAD) ────────────────────────────────────────────────

OLD_MODEL_A_SYSTEM = (
    "You are Model A, a direct and practical AI assistant.\n"
    "Your style is concise, action-oriented, and focused on immediate usefulness.\n"
    "Provide clear, straightforward answers that the user can act on right away."
)
OLD_MODEL_A_USER = """{context}

### YOUR TASK
Provide your best answer to the user's question. Be direct, practical, and actionable.
Keep your response concise but complete."""

OLD_MODEL_B_SYSTEM = (
    "You are Model B, a cautious and analytical AI assistant.\n"
    "Your style is thorough, risk-aware, and focused on identifying edge cases, assumptions, and potential problems.\n"
    "Always consider what could go wrong and what the user might be missing."
)
OLD_MODEL_B_USER = """{context}

### YOUR TASK
Provide your best answer to the user's question, but be sure to:
1. Identify key assumptions you're making
2. Point out edge cases or exceptions the user should consider
3. Highlight any risks or caveats
4. Suggest what information might be missing for a complete answer

Keep your response concise but thorough."""

OLD_MODEL_C_SYSTEM = (
    "You are Model C, a creative and alternative-thinking AI assistant.\n"
    "Your style is innovative, looking at problems from different angles and considering non-obvious solutions.\n"
    "You challenge conventional thinking and offer fresh perspectives."
)
OLD_MODEL_C_USER = """{context}

### YOUR TASK
Provide your best answer to the user's question from a unique or alternative perspective.
Consider:
1. Are there non-obvious approaches the user hasn't considered?
2. Can you reframe the problem in a useful way?
3. What creative solutions exist beyond the conventional answer?
4. How might different contexts or stakeholders view this differently?

Keep your response concise but insightful."""

OLD_DEBATE_SYSTEM = (
    "You are facilitating a debate between three AI models.\n"
    "Your job is to produce a concise critique round where each model reviews the others' answers.\n"
    "Be objective and focus on substantive points."
)

OLD_JUDGE_SYSTEM = (
    "You are the Judge, a senior expert reviewer.\n"
    "Your job is to synthesize multiple model responses and their debate into one final, high-quality answer.\n"
    "You must resolve conflicts, select the best ideas, and produce a coherent response.\n"
    "Never simply copy one model's answer - always synthesize."
)

# ── NEW PROMPTS (from prompts/*.txt) ──────────────────────────────────────────

def _prompt(name: str) -> str:
    import os
    path = os.path.join(os.path.dirname(__file__), "prompts", name)
    with open(path, encoding="utf-8") as f:
        return f.read()

NEW_MODEL_A_SYSTEM = _prompt("model_a_system.txt")
NEW_MODEL_A_USER   = _prompt("model_a_user.txt")
NEW_MODEL_B_SYSTEM = _prompt("model_b_system.txt")
NEW_MODEL_B_USER   = _prompt("model_b_user.txt")
NEW_MODEL_C_SYSTEM = _prompt("model_c_system.txt")
NEW_MODEL_C_USER   = _prompt("model_c_user.txt")
NEW_DEBATE_SYSTEM  = _prompt("debate_round_system.txt")
NEW_JUDGE_SYSTEM   = _prompt("judge_system.txt")

# ── PIPELINE ──────────────────────────────────────────────────────────────────

def build_context(query: str) -> str:
    return f"""### CHAT HISTORY
(no prior history)

### CURRENT USER QUESTION
user: {query}

Current Date: {datetime.now().date()}
"""

def build_debate_prompt(context, model_a, model_b, model_c, label_a, label_b, label_c) -> str:
    return f"""{context}

### INITIAL RESPONSES FROM THREE MODELS

**Model A ({label_a}):**
{model_a}

**Model B ({label_b}):**
{model_b}

**Model C ({label_c}):**
{model_c}

### DEBATE ROUND TASK
Produce a concise debate critique. For each model, identify:
1. What are the strongest points in their answer?
2. What are the weaknesses or gaps?
3. What would improve their response?

Format as a structured critique. Be specific and constructive."""

def build_judge_prompt_old(context, model_a, model_b, model_c, debate_critique) -> str:
    return f"""{context}

### MODEL RESPONSES

**Model A (Direct & Practical):**
{model_a}

**Model B (Cautious & Analytical):**
{model_b}

**Model C (Creative & Alternative):**
{model_c}

### DEBATE CRITIQUE
{debate_critique}

### YOUR TASK AS JUDGE
Synthesize the above into ONE final answer for the user. You must:
1. Resolve any conflicts or disagreements between models
2. Combine the strongest points from each response
3. Remove weak or unsupported claims
4. Ensure the final answer is clear, accurate, and directly addresses the user's question
5. Maintain an enthusiastic and helpful tone

Return ONLY the final answer. Do not explain your judging process."""

def build_debate_prompt_new(context, model_a, model_b, model_c) -> str:
    def _fmt(label, response):
        if response.startswith("Error:"):
            return f"**{label}:**\n[This model failed to respond — skip in critique]"
        return f"**{label}:**\n{response}"

    return f"""{context}

### MODEL RESPONSES

{_fmt("Model A (Pragmatic Senior Engineer)", model_a)}

{_fmt("Model B (Cautious Technical Lead)", model_b)}

{_fmt("Model C (Unconventional Systems Thinker)", model_c)}

### YOUR TASK
Evaluate each model's response on the four dimensions (Accuracy, Completeness, Actionability, Intellectual Honesty).
End with "Key Takeaways for Judge" listing 3-5 specific insights."""

def build_judge_prompt_new(context, model_a, model_b, model_c, debate_critique) -> str:
    def _fmt(label, response):
        if response.startswith("Error:"):
            return f"**{label}:**\n[This model failed to respond]"
        return f"**{label}:**\n{response}"

    return f"""{context}

### MODEL RESPONSES

{_fmt("Model A (Pragmatic Senior Engineer)", model_a)}

{_fmt("Model B (Cautious Technical Lead)", model_b)}

{_fmt("Model C (Unconventional Systems Thinker)", model_c)}

### DEBATE CRITIQUE
{debate_critique}

### YOUR TASK
Synthesise these into one definitive answer using the Key Takeaways as your guide.
Return ONLY the final answer."""


async def run_pipeline(query: str, use_new: bool) -> dict:
    label = "NEW" if use_new else "OLD"
    context = build_context(query)

    # ── Models in parallel ────────────────────────────────────────────────────
    t0 = time.monotonic()
    model_a_sys = NEW_MODEL_A_SYSTEM if use_new else OLD_MODEL_A_SYSTEM
    model_b_sys = NEW_MODEL_B_SYSTEM if use_new else OLD_MODEL_B_SYSTEM
    model_c_sys = NEW_MODEL_C_SYSTEM if use_new else OLD_MODEL_C_SYSTEM
    model_a_user_tmpl = NEW_MODEL_A_USER if use_new else OLD_MODEL_A_USER
    model_b_user_tmpl = NEW_MODEL_B_USER if use_new else OLD_MODEL_B_USER
    model_c_user_tmpl = NEW_MODEL_C_USER if use_new else OLD_MODEL_C_USER

    ra, rb, rc = await asyncio.gather(
        call_llm_async(model_a_user_tmpl.replace("{context}", context), system_prompt=model_a_sys),
        call_llm_async(model_b_user_tmpl.replace("{context}", context), system_prompt=model_b_sys),
        call_llm_async(model_c_user_tmpl.replace("{context}", context), system_prompt=model_c_sys),
    )
    t_models = time.monotonic() - t0

    # ── Debate round ──────────────────────────────────────────────────────────
    t1 = time.monotonic()
    if use_new:
        debate_prompt = build_debate_prompt_new(context, ra, rb, rc)
        debate_sys = NEW_DEBATE_SYSTEM
    else:
        debate_prompt = build_debate_prompt(context, ra, rb, rc,
                                            "Direct & Practical", "Cautious & Analytical", "Creative & Alternative")
        debate_sys = OLD_DEBATE_SYSTEM
    debate = await call_llm_async(debate_prompt, system_prompt=debate_sys)
    t_debate = time.monotonic() - t1

    # ── Judge ─────────────────────────────────────────────────────────────────
    t2 = time.monotonic()
    if use_new:
        judge_prompt = build_judge_prompt_new(context, ra, rb, rc, debate)
        judge_sys = NEW_JUDGE_SYSTEM
    else:
        judge_prompt = build_judge_prompt_old(context, ra, rb, rc, debate)
        judge_sys = OLD_JUDGE_SYSTEM
    answer = await call_llm_async(judge_prompt, system_prompt=judge_sys)
    t_judge = time.monotonic() - t2

    return {
        "label": label,
        "query": query,
        "model_a": ra,
        "model_b": rb,
        "model_c": rc,
        "debate": debate,
        "answer": answer,
        "t_models": t_models,
        "t_debate": t_debate,
        "t_judge": t_judge,
        "t_total": time.monotonic() - t0,
    }


def _wrap(text: str, width=90, indent=4) -> str:
    pad = " " * indent
    lines = text.strip().splitlines()
    result = []
    for line in lines:
        if line.strip() == "":
            result.append("")
        else:
            result.extend(textwrap.wrap(line, width=width, initial_indent=pad, subsequent_indent=pad))
    return "\n".join(result)


def print_comparison(old: dict, new: dict):
    SEP = "=" * 100
    HALF = "-" * 48

    print(f"\n{SEP}")
    print(f"QUERY: {old['query']}")
    print(SEP)

    for stage, old_key, new_key in [
        ("MODEL A", "model_a", "model_a"),
        ("MODEL B", "model_b", "model_b"),
        ("MODEL C", "model_c", "model_c"),
        ("DEBATE CRITIQUE", "debate", "debate"),
        ("JUDGE (FINAL ANSWER)", "answer", "answer"),
    ]:
        old_text = old[old_key]
        new_text = new[new_key]
        print(f"\n{'─'*100}")
        print(f"  {stage}")
        print(f"{'─'*100}")
        print(f"  ── OLD ({len(old_text):,} chars) {'─'*40}")
        print(_wrap(old_text))
        print(f"\n  ── NEW ({len(new_text):,} chars) {'─'*40}")
        print(_wrap(new_text))

    print(f"\n{'─'*100}")
    print(f"  TIMING")
    print(f"{'─'*100}")
    print(f"  {'':30s} {'OLD':>10s}   {'NEW':>10s}")
    print(f"  {'Models (parallel)':30s} {old['t_models']:>9.1f}s   {new['t_models']:>9.1f}s")
    print(f"  {'Debate round':30s} {old['t_debate']:>9.1f}s   {new['t_debate']:>9.1f}s")
    print(f"  {'Judge':30s} {old['t_judge']:>9.1f}s   {new['t_judge']:>9.1f}s")
    print(f"  {'Total':30s} {old['t_total']:>9.1f}s   {new['t_total']:>9.1f}s")
    print()


QUERIES = [
    "Should I use PostgreSQL or MongoDB for my new SaaS app?",
    "How should I handle authentication in my REST API?",
    "Should we rewrite our monolith as microservices?",
]


async def main():
    for query in QUERIES:
        print(f"\nRunning OLD pipeline for: {query!r}...")
        old = await run_pipeline(query, use_new=False)
        print(f"Running NEW pipeline for: {query!r}...")
        new = await run_pipeline(query, use_new=True)
        print_comparison(old, new)


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    asyncio.run(main())
