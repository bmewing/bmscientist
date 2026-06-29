from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from bmscientist.ingestion import ENCRYPTED_PARQUET_SUFFIX, read_encrypted_table
from bmscientist.llm import DeepSeekLLM
from bmscientist.models import GraphEntityMatch, GraphEntityMatchBuckets, GraphQueryPlan, GraphQueryResult, GraphTableSchema
from bmscientist.material_alias import normalize_alias_text
from bmscientist.prompt_library import PROMPTS


DEFAULT_GRAPH_PATH = Path("data/graph")


@dataclass(frozen=True)
class GraphTableSource:
    table_name: str
    parquet_path: Path
    scope: str
    encrypted: bool = False


class DuckDBGraphQueryEngine:
    def __init__(
        self,
        graph_path: Path | None = None,
        *,
        public_graph_path: Path | None = None,
        private_graph_path: Path | None = None,
        decryption_key: bytes | None = None,
    ):
        self._graph_path = public_graph_path if public_graph_path is not None else graph_path if graph_path is not None else DEFAULT_GRAPH_PATH
        self._private_graph_path = private_graph_path
        self._decryption_key = decryption_key
        if self._private_graph_path is not None and self._decryption_key is None:
            raise ValueError("A private graph path requires a session decryption key.")

    def list_tables(self) -> list[GraphTableSchema]:
        return self._effective_table_schemas()

    def schema_summary(self) -> str:
        sections = []
        for table in self.list_tables():
            columns = ", ".join(table.columns) if table.columns else "(no columns)"
            sections.append(f"{table.table_name}: {columns}")
        return "\n".join(sections)

    def match_entities(self, question: str, top_k_per_label: int = 3) -> list[GraphEntityMatch]:
        normalized_question = normalize_alias_text(question)
        if not normalized_question:
            return []

        labels = [
            ("Product", "product_id"),
            ("MaterialFamily", "material_family_id"),
            ("MaterialGrade", "material_grade_id"),
            ("Market", "market_id"),
            ("Application", "application_id"),
            ("Company", "company_id"),
            ("CriticalToQuality", "ctq_id"),
            ("Endpoint", "endpoint_id"),
        ]
        matches_by_label: dict[str, list[GraphEntityMatch]] = {}
        for label, key in labels:
            sources = self._sources_for_table(label, category="nodes")
            if not sources:
                continue
            con = duckdb.connect(database=":memory:")
            try:
                self._register_table_view(con, label, sources)
                rows = con.execute(f"SELECT * FROM {quote_ident(label)}").fetchall()
                columns = [item[0] for item in con.description or []]
            finally:
                con.close()
            for row in rows:
                payload = dict(zip(columns, row, strict=False))
                node_id = str(payload.get(key) or "")
                if not node_id:
                    continue
                name = str(payload.get("name") or "")
                for candidate_text, basis in entity_candidate_texts(payload):
                    score = score_entity_match(normalized_question, candidate_text)
                    if score < 0.45:
                        continue
                    matches_by_label.setdefault(label, []).append(
                        GraphEntityMatch(
                            node_label=label,
                            node_id=node_id,
                            name=name or candidate_text,
                            matched_text=candidate_text,
                            score=score,
                            match_basis=basis,
                        )
                    )

        merged: list[GraphEntityMatch] = []
        for label, matches in matches_by_label.items():
            deduped: dict[str, GraphEntityMatch] = {}
            for match in sorted(matches, key=lambda item: item.score, reverse=True):
                existing = deduped.get(match.node_id)
                if existing is None or match.score > existing.score:
                    deduped[match.node_id] = match
            merged.extend(sorted(deduped.values(), key=lambda item: item.score, reverse=True)[:top_k_per_label])
        merged.sort(key=lambda item: (item.score, item.node_label, item.name), reverse=True)
        return merged

    def entity_match_summary(self, question: str, top_k_per_label: int = 3) -> str:
        matches = self.match_entities(question, top_k_per_label=top_k_per_label)
        return self.entity_match_summary_from_matches(matches)

    @staticmethod
    def entity_match_summary_from_matches(matches: list[GraphEntityMatch]) -> str:
        if not matches:
            return "No strong entity matches found in the graph."
        lines = []
        for match in matches:
            lines.append(
                f"{match.node_label}: {match.name} [{match.node_id}] score={match.score:.2f} matched_on={match.matched_text} basis={match.match_basis}"
            )
        return "\n".join(lines)

    def query(self, sql: str, limit: int = 200) -> GraphQueryResult:
        normalized_sql = self._validate_sql(sql)
        con = duckdb.connect(database=":memory:")
        try:
            self._register_all_table_views(con)
            wrapped_sql = f"SELECT * FROM ({normalized_sql}) AS graph_query_result LIMIT {int(limit) + 1}"
            rel = con.execute(wrapped_sql)
            rows = rel.fetchall()
            columns = [item[0] for item in rel.description or []]
            truncated = len(rows) > limit
            visible_rows = rows[:limit]
            payload_rows = [dict(zip(columns, row, strict=False)) for row in visible_rows]
            return GraphQueryResult(
                sql=normalized_sql,
                columns=columns,
                rows=payload_rows,
                row_count=len(payload_rows),
                truncated=truncated,
            )
        finally:
            con.close()

    def _table_sources(self) -> list[GraphTableSource]:
        sources: list[GraphTableSource] = []
        for category in ("nodes", "edges", "enrichment"):
            base = self._graph_path / category
            if not base.exists():
                continue
            for path in sorted(base.glob("*.parquet")):
                sources.append(GraphTableSource(path.stem, path, "public", encrypted=False))
        if self._private_graph_path is not None:
            for category in ("nodes", "edges", "enrichment", "chunks"):
                base = self._private_graph_path / category
                if not base.exists():
                    continue
                for path in sorted(base.glob(f"*{ENCRYPTED_PARQUET_SUFFIX}")):
                    table_name = path.name[: -len(ENCRYPTED_PARQUET_SUFFIX)]
                    sources.append(GraphTableSource(table_name, path, "private", encrypted=True))
        return sources

    def _sources_by_table(self) -> dict[str, list[GraphTableSource]]:
        grouped: dict[str, list[GraphTableSource]] = {}
        for source in self._table_sources():
            grouped.setdefault(source.table_name, []).append(source)
        return grouped

    def _sources_for_table(self, table_name: str, *, category: str | None = None) -> list[GraphTableSource]:
        sources = self._sources_by_table().get(table_name, [])
        if category is None:
            return sources
        return [source for source in sources if source.parquet_path.parent.name == category]

    def _effective_table_schemas(self) -> list[GraphTableSchema]:
        tables: list[GraphTableSchema] = []
        for table_name, sources in sorted(self._sources_by_table().items()):
            if len(sources) == 1:
                source = sources[0]
                tables.append(
                    GraphTableSchema(
                        table_name=table_name,
                        parquet_path=str(source.parquet_path.resolve()),
                        columns=self._columns_for_source(source),
                    )
                )
                continue
            public_sources = [source for source in sources if source.scope == "public"]
            private_sources = [source for source in sources if source.scope == "private"]
            if public_sources and private_sources and self._same_columns(sources):
                tables.append(
                    GraphTableSchema(
                        table_name=table_name,
                        parquet_path=", ".join(str(source.parquet_path.resolve()) for source in sources),
                        columns=["graph_scope", *self._columns_for_source(sources[0])],
                    )
                )
            else:
                for source in sources:
                    exposed_name = table_name if source.scope == "public" else f"private_{table_name}"
                    tables.append(
                        GraphTableSchema(
                            table_name=exposed_name,
                            parquet_path=str(source.parquet_path.resolve()),
                            columns=self._columns_for_source(source),
                        )
                    )
        return tables

    def _register_all_table_views(self, con: duckdb.DuckDBPyConnection) -> None:
        for table_name, sources in self._sources_by_table().items():
            self._register_table_view(con, table_name, sources)

    def _register_table_view(self, con: duckdb.DuckDBPyConnection, table_name: str, sources: list[GraphTableSource]) -> None:
        if not sources:
            return
        if len(sources) == 1:
            relation_name = self._register_source_relation(con, sources[0])
            con.execute(f"CREATE VIEW {quote_ident(table_name)} AS SELECT * FROM {quote_ident(relation_name)}")
            return
        if self._same_columns(sources):
            relation_names = [self._register_source_relation(con, source) for source in sources]
            selects = [
                f"SELECT {quote_literal(source.scope)} AS graph_scope, * FROM {quote_ident(relation_name)}"
                for source, relation_name in zip(sources, relation_names, strict=False)
            ]
            con.execute(f"CREATE VIEW {quote_ident(table_name)} AS {' UNION ALL '.join(selects)}")
            return
        for source in sources:
            exposed_name = table_name if source.scope == "public" else f"private_{table_name}"
            relation_name = self._register_source_relation(con, source)
            con.execute(f"CREATE VIEW {quote_ident(exposed_name)} AS SELECT * FROM {quote_ident(relation_name)}")

    def _register_source_relation(self, con: duckdb.DuckDBPyConnection, source: GraphTableSource) -> str:
        relation_name = f"__{source.scope}_{source.table_name}"
        if source.encrypted:
            if self._decryption_key is None:
                raise ValueError("Cannot query encrypted private graph tables without a decryption key.")
            con.register(relation_name, read_encrypted_table(source.parquet_path, self._decryption_key))
        else:
            con.execute(
                f"CREATE VIEW {quote_ident(relation_name)} AS SELECT * FROM read_parquet({quote_literal(str(source.parquet_path.resolve()))})",
            )
        return relation_name

    def _same_columns(self, sources: list[GraphTableSource]) -> bool:
        column_sets = [self._columns_for_source(source) for source in sources]
        return bool(column_sets) and all(columns == column_sets[0] for columns in column_sets[1:])

    def _columns_for_source(self, source: GraphTableSource) -> list[str]:
        if source.encrypted:
            if self._decryption_key is None:
                raise ValueError("Cannot inspect encrypted private graph tables without a decryption key.")
            return read_encrypted_table(source.parquet_path, self._decryption_key).column_names
        return self._columns_for_path(source.parquet_path)

    def _columns_for_path(self, path: Path) -> list[str]:
        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path.resolve())]).fetchall()
            return [str(row[0]) for row in rows]
        finally:
            con.close()

    @staticmethod
    def _validate_sql(sql: str) -> str:
        cleaned = sql.strip().strip(";")
        lowered = cleaned.lower()
        if not cleaned:
            raise ValueError("Graph SQL cannot be empty.")
        if not (lowered.startswith("select") or lowered.startswith("with")):
            raise ValueError("Only read-only SELECT or WITH queries are allowed.")
        if any(token in lowered for token in (" insert ", " update ", " delete ", " create ", " alter ", " drop ", " attach ", " copy ")):
            raise ValueError("Only read-only graph queries are allowed.")
        if re.search(r";\s*\S", cleaned):
            raise ValueError("Only a single SQL statement is allowed.")
        return cleaned


class GraphQueryAgent:
    def __init__(self, llm: DeepSeekLLM, engine: DuckDBGraphQueryEngine):
        self._llm = llm
        self._engine = engine

    def plan(self, question: str) -> GraphQueryPlan:
        system_prompt = PROMPTS.render("graph_query_agent", "compose.system")
        matches = self._engine.match_entities(question)
        user_prompt = PROMPTS.render(
            "graph_query_agent",
            "compose.user",
            graph_schema=self._engine.schema_summary(),
            graph_entity_matches=self._engine.entity_match_summary_from_matches(matches),
            user_question=question,
        )
        return self._llm.complete_json(GraphQueryPlan, system_prompt, user_prompt)

    def inspect(self, question: str) -> tuple[GraphQueryPlan, list[GraphEntityMatch]]:
        matches = self._engine.match_entities(question)
        system_prompt = PROMPTS.render("graph_query_agent", "compose.system")
        user_prompt = PROMPTS.render(
            "graph_query_agent",
            "compose.user",
            graph_schema=self._engine.schema_summary(),
            graph_entity_matches=self._engine.entity_match_summary_from_matches(matches),
            user_question=question,
        )
        plan = self._llm.complete_json(GraphQueryPlan, system_prompt, user_prompt)
        return plan, matches

    def run(self, question: str, limit: int = 200) -> GraphQueryResult:
        plan, matches = self.inspect(question)
        result = self._engine.query(plan.sql, limit=limit)
        return result.model_copy(
            update={
                "rationale": plan.rationale,
                "assumptions": plan.assumptions,
                "matched_entities": matches,
                "matched_entity_buckets": group_entity_matches(matches),
            }
        )


def format_graph_query_result(result: GraphQueryResult) -> str:
    payload = {
        "sql": result.sql,
        "rationale": result.rationale,
        "assumptions": result.assumptions,
        "matched_entities": [item.model_dump(mode="json") for item in result.matched_entities],
        "matched_entity_buckets": result.matched_entity_buckets.model_dump(mode="json"),
        "columns": result.columns,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "rows": result.rows,
    }
    return json.dumps(payload, indent=2, default=str)


def quote_ident(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def quote_literal(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def entity_candidate_texts(payload: dict[str, Any]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for field_name in ("name", "canonical_name", "normalized_name", "trade_name", "grade_name", "material_family_name", "primary_slug"):
        text = str(payload.get(field_name) or "").strip()
        if text:
            values.append((text, field_name))
    aliases = parse_json_list(payload.get("aliases_json"))
    for alias in aliases:
        values.append((alias, "aliases_json"))
    return dedupe_candidate_texts(values)


def dedupe_candidate_texts(values: list[tuple[str, str]]) -> list[tuple[str, str]]:
    deduped: dict[str, tuple[str, str]] = {}
    for text, basis in values:
        normalized = normalize_alias_text(text)
        if normalized and normalized not in deduped:
            deduped[normalized] = (text, basis)
    return list(deduped.values())


def parse_json_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item]


def score_entity_match(normalized_question: str, candidate_text: str) -> float:
    normalized_candidate = normalize_alias_text(candidate_text)
    if not normalized_candidate:
        return 0.0
    if normalized_candidate == normalized_question:
        return 1.0

    question_tokens = normalized_tokens(normalized_question)
    candidate_tokens = normalized_tokens(normalized_candidate)
    if not candidate_tokens:
        return 0.0

    substring_score = 0.0
    if normalized_candidate in normalized_question:
        substring_score = 0.95
    elif normalized_question in normalized_candidate:
        substring_score = 0.75

    overlap = len(question_tokens & candidate_tokens) / max(len(candidate_tokens), 1)
    sequence = difflib.SequenceMatcher(None, normalized_candidate, normalized_question).ratio()
    score = max(substring_score, overlap * 0.9, sequence * 0.7)
    if candidate_tokens and question_tokens and candidate_tokens <= question_tokens:
        score = max(score, 0.9)
    return min(score, 1.0)


def normalized_tokens(value: str) -> set[str]:
    tokens = set()
    for token in normalize_alias_text(value).split():
        if len(token) <= 1:
            continue
        tokens.add(singularize_token(token))
    return tokens


def singularize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def group_entity_matches(matches: list[GraphEntityMatch]) -> GraphEntityMatchBuckets:
    buckets = GraphEntityMatchBuckets()
    for match in matches:
        if match.node_label in ("Product", "MaterialFamily", "MaterialGrade"):
            buckets.materials.append(match)
        elif match.node_label == "Market":
            buckets.markets.append(match)
        elif match.node_label == "Application":
            buckets.applications.append(match)
        elif match.node_label == "Company":
            buckets.companies.append(match)
        elif match.node_label == "CriticalToQuality":
            buckets.ctqs.append(match)
        elif match.node_label == "Endpoint":
            buckets.endpoints.append(match)
        else:
            buckets.other.append(match)
    return buckets
