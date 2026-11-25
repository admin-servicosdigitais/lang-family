"""Fake em memória do plano de dados do Pinecone (upsert, query, delete, stats).

O emulador oficial (Pinecone Local) foi descontinuado em 2026-10-07. Este fake
exercita o SDK real contra um servidor local; a prova contra o serviço de verdade
é o teste ``live``.
"""

import math
from typing import Any

Record = dict[str, Any]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norms if norms else 0.0


def _matches(record: Record, conditions: dict[str, Any] | None) -> bool:
    return all(
        record["metadata"].get(field) == condition["$eq"]
        for field, condition in (conditions or {}).items()
    )


class FakePinecone:
    def __init__(self) -> None:
        self.namespaces: dict[str, dict[str, Record]] = {}

    def handle(self, method: str, path: str, body: Any) -> tuple[int, Any]:
        body = body or {}
        records = self.namespaces.setdefault(body.get("namespace", ""), {})
        if path == "/vectors/upsert":
            records.update({r["id"]: r for r in body["vectors"]})
            return 200, {"upsertedCount": len(body["vectors"])}
        if path == "/query":
            scored = [
                {
                    "id": r["id"],
                    "score": _cosine(body["vector"], r["values"]),
                    "values": [],
                    "metadata": r["metadata"],
                }
                for r in records.values()
                if _matches(r, body.get("filter"))
            ]
            scored.sort(key=lambda match: match["score"], reverse=True)
            return 200, {"matches": scored[: body["topK"]], "namespace": body.get("namespace", "")}
        if path == "/vectors/delete":
            if body.get("deleteAll"):
                records.clear()
            elif "ids" in body:
                for record_id in body["ids"]:
                    records.pop(record_id, None)
            else:
                for record_id in [i for i, r in records.items() if _matches(r, body["filter"])]:
                    del records[record_id]
            return 200, {}
        if path == "/describe_index_stats":
            namespaces = {
                name: {"vectorCount": len(items)}
                for name, items in self.namespaces.items()
                if items
            }
            total = sum(n["vectorCount"] for n in namespaces.values())
            return 200, {
                "namespaces": namespaces,
                "dimension": 8,
                "indexFullness": 0.0,
                "totalVectorCount": total,
            }
        return 404, {"error": {"code": "NOT_FOUND", "message": f"{method} {path}"}}
