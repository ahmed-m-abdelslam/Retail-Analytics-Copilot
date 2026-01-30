# agent/tools/sqlite_tool.py
import sqlite3
from typing import Any, Dict, List, Tuple


class SQLiteTool:
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path

    def execute(self, sql: str) -> Dict[str, Any]:
        """
        Execute SQL and return {'columns': [...], 'rows': [...]} or error.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            if rows:
                columns = rows[0].keys()
            else:
                columns = [d[0] for d in cur.description] if cur.description else []
            data_rows = [tuple(row[c] for c in columns) for row in rows]
            return {
                "ok": True,
                "columns": list(columns),
                "rows": data_rows,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            conn.close()

    def introspect_schema(self) -> str:
        """
        Return a short textual description of key tables/columns
        to feed into the NL→SQL module.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            tables = {}
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            for (table_name,) in cur.fetchall():
                cur2 = conn.cursor()
                cur2.execute(f"PRAGMA table_info('{table_name}');")
                cols = [row[1] for row in cur2.fetchall()]
                tables[table_name] = cols

            # convert to compact text description
            parts = []
            for t, cols in tables.items():
                cols_str = ", ".join(cols)
                parts.append(f"Table {t} has columns: {cols_str}")
            return "\n".join(parts)
        finally:
            conn.close()
