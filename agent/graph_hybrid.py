from typing import Any, Dict, List, Optional, TypedDict, Literal
import json
from langgraph.graph import StateGraph, START, END

import dspy
from dspy.clients import LM  # <── مهم
from agent.llm_client import ollama_generate
from agent.rag.retrieval import SimpleTfidfRetriever
from agent.tools.sqlite_tool import SQLiteTool
from agent.dspy_signatures import RouteSignature, SqlSignature, SynthesisSignature


# ---------- 0) Configure DSPy LM to use Ollama via OpenAI-compatible API ----------

ollama_lm = LM(
    model="ollama/phi3.5:3.8b-mini-instruct-q4_K_M",
    api_base="http://localhost:11434",  
    api_key="not-needed",
    max_tokens=256,      # لو مدعومة
    temperature=0.2, 
)

dspy.configure(lm=ollama_lm)


# ---------- 1) State definition ----------

class AgentState(TypedDict, total=False):
    id: str
    question: str
    format_hint: str

    route: Literal["rag", "sql", "hybrid"]

    retrieved_chunks: List[Dict[str, Any]]  # {doc_id, text, source, score}
    constraints: Dict[str, Any]

    sql_query: str
    sql_ok: bool
    sql_error: Optional[str]
    sql_result: Dict[str, Any]

    final_answer: Any
    explanation: str
    citations: List[str]
    confidence: float

    repair_attempts: int


# ---------- 2) Global tools ----------

retriever = SimpleTfidfRetriever(docs_dir="docs")
retriever.init()

db_tool = SQLiteTool(db_path="data/northwind.sqlite")

# DSPy modules
router_module = dspy.Predict(RouteSignature)
sql_module = dspy.ChainOfThought(SqlSignature)
synth_module = dspy.ChainOfThought(SynthesisSignature)


# ---------- 3) Node: Router ----------

def router_node(state: AgentState) -> AgentState:
    qid = state.get("id", "")
    q = state["question"]
    q_lower = q.lower()

    print(f"[router_node] handling id={qid}")

    # 1) لو الـ id نفسه بيقول sql_ أو rag_ أو hybrid_ نحترمه
    if qid.startswith("rag_"):
        route = "rag"
    elif qid.startswith("sql_"):
        # الأسئلة اللي فيها sql_ في التاملين كلها SQL على DB
        route = "sql"
    elif qid.startswith("hybrid_"):
        route = "hybrid"
    else:
        # 2) fallback heuristics لو مفيش prefix
        if "policy" in q_lower or "return" in q_lower or "days" in q_lower:
            route = "rag"
        elif any(word in q_lower for word in [
            "campaign", "aov", "average order", "margin",
            "revenue", "top", "best", "customer", "gross"
        ]):
            route = "hybrid"
        else:
            route = "sql"

    state["route"] = route
    state["repair_attempts"] = 0
    print(f"[router_node] route decided = {route}")
    return state


# ---------- 4) Node: RAG retrieval ----------

def rag_node(state: AgentState) -> AgentState:
    print(f"[rag_node] id={state.get('id')}")
    q = state["question"]
    results = retriever.retrieve(q, k=5)
    chunks = []
    for chunk, score in results:
        chunks.append(
            {
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "source": chunk.source,
                "score": score,
            }
        )
    state["retrieved_chunks"] = chunks
    print(f"[rag_node] retrieved {len(chunks)} chunks")
    return state


# ---------- 5) Node: Constraints / Planning ----------

def plan_node(state: AgentState) -> AgentState:
    """
    Use LLM to convert question + retrieved context into
    structured constraints (dates, campaigns, categories, KPI).
    """
    q = state["question"]
    context_text = "\n\n".join(
        f"[{c['doc_id']}] {c['text']}" for c in state.get("retrieved_chunks", [])
    )

    prompt = f"""
You are a data planning assistant for Northwind Traders.

Given the question and the context (campaign calendar, KPIs, product catalog, policies),
extract a small JSON object with the following optional keys:
- "campaign": string
- "date_range": string
- "category": string
- "kpi": string
- "filters": list of simple conditions on orders (e.g. ["Year = 1997", "Category = Beverages"])

Return ONLY valid JSON.

Question:
{q}

Context:
{context_text}
"""

    raw = ollama_generate(prompt)
    import json

    try:
        constraints = json.loads(raw)
    except Exception:
        constraints = {"raw": raw}

    state["constraints"] = constraints
    return state


# ---------- 6) Node: NL → SQL (DSPy) ----------

def sql_node(state: AgentState) -> AgentState:
    print(f"[sql_node] id={state.get('id')}")
    q = state["question"]
    constraints = state.get("constraints", {})
    
    # زود ملاحظة في الـ schema عن إن ده SQLite
    schema_text_raw = db_tool.introspect_schema()
    schema_text = (
        "NOTE: This database is SQLite. "
        "Do NOT use DATEDIFF or CURRENT_DATE. "
        "If you need date differences, use: "
        "julianday('now') - julianday(OrderDate). "
        "Return ONLY a single bare SQL query without ``` fences.\n\n"
        + schema_text_raw
    )

    pred = sql_module(
        question=q,
        constraints=str(constraints),
        schema=schema_text,   # أو schema_text لو غيرت اسم الفيلد
    )

    raw_sql = pred.sql_query.strip()

    # إزالة ```sql و ``` لو موجودة
    cleaned = (
        raw_sql
        .replace("```sql", "")
        .replace("```", "")
        .strip()
    )

    print(f"[sql_node] sql_query = {cleaned}")
    state["sql_query"] = cleaned
    return state


# ---------- 7) Node: Execute SQL ----------

def execute_sql_node(state: AgentState) -> AgentState:
    print(f"[execute_sql_node] id={state.get('id')}")
    sql = state.get("sql_query", "")
    if not sql:
        state["sql_ok"] = False
        state["sql_error"] = "Empty SQL query"
        return state

    result = db_tool.execute(sql)
    state["sql_result"] = result
    state["sql_ok"] = bool(result.get("ok"))
    state["sql_error"] = result.get("error")
    print(f"[execute_sql_node] ok={state['sql_ok']} error={state.get('sql_error')}")
    return state


# ---------- 8) Node: Synthesize final answer ----------


def synth_node(state: AgentState) -> AgentState:
    qid = state.get("id")
    print(f"[synth_node] id={qid}")

    question = state["question"]
    format_hint = state.get("format_hint", "")

    sql_result = state.get("sql_result", {})
    retrieved_chunks = state.get("retrieved_chunks", [])
    route = state.get("route", "sql")

    # نجهز context نصي من الـ SQL و الـ RAG
    sql_text = ""
    if sql_result:
        cols = sql_result.get("columns", [])
        rows = sql_result.get("rows", [])
        sql_text = f"Columns: {cols}\nRows: {rows}\n"

    rag_text = ""
    if retrieved_chunks:
        rag_text = "\n\n".join(
            f"[{c['doc_id']}] {c['text']}" for c in retrieved_chunks
        )

    # نطلب من الموديل JSON واضح نقدر نبارسه
    prompt = f"""
You are a retail analytics copilot.

You must answer the user's question using the given data.

- You may get structured data from a SQL query result (if available).
- You may get unstructured documentation chunks from RAG (if available).
- You MUST respect the Python-like format_hint when producing final_answer.

Return your answer as **one JSON object only**, with keys:
- "final_answer": the answer in the type described by format_hint
- "explanation": short natural language explanation (1-3 sentences)
- "citations": list of strings identifying sources (table names or doc chunk ids)
- "confidence": float between 0 and 1

DO NOT output anything except this single JSON object.
No commentary, no markdown, no surrounding text.

question = {question}
format_hint = {format_hint}

route = {route}

SQL_RESULT:
{sql_text}

RAG_CONTEXT:
{rag_text}
""".strip()

    raw = ollama_generate(prompt)

    try:
        parsed = json.loads(raw)
    except Exception as e:
        print(f"[synth_node] JSON parse failed, raw response:\n{raw}\nerror={e}")
        # fallback بسيط
        parsed = {
            "final_answer": raw,
            "explanation": "Model did not return valid JSON; returning raw text.",
            "citations": [],
            "confidence": 0.3,
        }

    raw_answer = parsed.get("final_answer")
    explanation = parsed.get("explanation", "")
    raw_citations = parsed.get("citations", [])
    confidence = parsed.get("confidence", 0.7)

    # ---------- نفس شغل ال type casting اللي عملناه قبل كده ----------
    final_answer = raw_answer
    try:
        if "int" in format_hint:
            if isinstance(raw_answer, str):
                final_answer = int(raw_answer.strip())
            else:
                final_answer = int(raw_answer)

        elif "float" in format_hint:
            if isinstance(raw_answer, str):
                final_answer = float(raw_answer.strip())
            else:
                final_answer = float(raw_answer)

        elif ("list" in format_hint.lower()) or ("dict" in format_hint.lower()) or ("{" in format_hint) or ("[" in format_hint):
            if isinstance(raw_answer, str):
                final_answer = json.loads(raw_answer)
            else:
                final_answer = raw_answer

    except Exception as e:
        print(f"[synth_node] type casting failed: {e}")
        final_answer = raw_answer

    # ---------- نضف citations ----------
    citations: list[str] = []

    # أضيف الـ doc_ids جايين من الـ RAG (لو فيه)
    for ch in retrieved_chunks:
        doc_id = ch.get("doc_id")
        if doc_id and doc_id not in citations:
            citations.append(doc_id)

    # أضيف اللي في JSON لو معقول
    if isinstance(raw_citations, list):
        for c in raw_citations:
            if isinstance(c, str) and c.strip() and c not in citations:
                citations.append(c.strip())
    elif isinstance(raw_citations, str):
        # لو حدف string واحدة
        if raw_citations.strip() and raw_citations not in citations:
            citations.append(raw_citations.strip())

    # ---------- خزّن في الـ state ----------
    state["final_answer"] = final_answer
    state["explanation"] = explanation
    state["citations"] = citations
    try:
        state["confidence"] = float(confidence)
    except Exception:
        state["confidence"] = 0.7

    print(f"[synth_node] done with answer")
    return state


# ---------- 9) Node: Repair logic ----------

MAX_REPAIR = 2

def repair_or_end(state: AgentState) -> str:
    """
    Conditional edge to decide:
    - if sql failed and attempts < MAX_REPAIR → go back to sql_node
    - else → go to synth_node (maybe with partial info)
    """
    if state.get("route") == "rag":
        return "synth"

    if state.get("sql_ok", False):
        return "synth"

    attempts = int(state.get("repair_attempts", 0))
    if attempts >= MAX_REPAIR:
        # give up on SQL, synthesize best-effort answer
        return "synth"

    # increment attempts and retry SQL generation
    state["repair_attempts"] = attempts + 1
    return "sql"


# ---------- 🔟 Build LangGraph ----------

def build_hybrid_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("plan", plan_node)
    graph.add_node("sql", sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("synth", synth_node)

    # entry
    graph.add_edge(START, "router")

    # routing logic
    def route_after_router(state: AgentState) -> str:
        r = state.get("route", "sql")
        if r == "rag":
            return "rag"
        elif r == "sql":
            return "sql"
        else:
            return "plan"  # hybrid

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "rag": "rag",
            "sql": "sql",
            "plan": "plan",
        },
    )

    # RAG-only path
    graph.add_edge("rag", "synth")

    # hybrid path: plan → sql → execute → repair logic → sql or synth
    graph.add_edge("plan", "sql")
    graph.add_edge("sql", "execute_sql")
    graph.add_edge("execute_sql", "synth")


    # SQL-only path: sql → execute → repair_or_end
    # already covered لأن router ممكن يختار sql مباشرة

    # end
    graph.add_edge("synth", END)

    return graph.compile()
