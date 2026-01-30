# agent/dspy_signatures.py
import dspy


class RouteSignature(dspy.Signature):
    """Route the question to RAG / SQL / Hybrid."""

    question = dspy.InputField(desc="user natural language question")
    format_hint = dspy.InputField(desc="Python-like type hint for expected answer")
    route = dspy.OutputField(desc="one of: rag, sql, hybrid")


class SqlSignature(dspy.Signature):
    """Generate a single SQL query for Northwind to answer the question."""

    question = dspy.InputField()
    constraints = dspy.InputField(desc="structured info: dates, campaign, category, KPI, etc.")
    schema_text = dspy.InputField(desc="short description of available tables & columns")
    sql_query = dspy.OutputField(desc="SQL query string")


class SynthesisSignature(dspy.Signature):
    """Turn SQL result + RAG context into final structured answer."""

    question = dspy.InputField()
    format_hint = dspy.InputField()
    sql_result = dspy.InputField(desc="rows & columns from DB, e.g. JSON-like")
    retrieved_context = dspy.InputField(desc="top docs chunks from RAG")
    final_answer = dspy.OutputField(desc="must follow the format_hint exactly")
    explanation = dspy.OutputField(desc="1-3 sentence explanation")
    citations = dspy.OutputField(desc="list of sources: table names + chunk ids")
    confidence = dspy.OutputField(desc="float between 0 and 1")
