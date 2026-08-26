"""Convert an HPO OBO release into graph JSON, JSONL, and CSV exports.

The graph JSON is the runtime source for HPOCatalog. JSONL and CSV are flat,
inspection-friendly exports and are not required by the new retrieval pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


HPO_URI_PREFIX = "http://purl.obolibrary.org/obo/HP_"


def unescape_obo(value: str) -> str:
    replacements = {"\\": "\\", '"': '"', "n": "\n", "r": "\r", "t": "\t"}
    return re.sub(r'\\([\\"nrt])', lambda match: replacements[match.group(1)], value)


def quoted_value(value: str) -> str:
    match = re.match(r'^\s*"((?:\\.|[^"\\])*)"', value)
    return unescape_obo(match.group(1)).strip() if match else ""


def predicate_for_scope(scope: str) -> str:
    return {
        "EXACT": "hasExactSynonym",
        "RELATED": "hasRelatedSynonym",
        "BROAD": "hasBroadSynonym",
        "NARROW": "hasNarrowSynonym",
    }.get(scope, "hasRelatedSynonym")


def parse_obo(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    terms: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    header: dict[str, str] = {}
    current: dict[str, Any] | None = None
    section = ""

    def finish_term() -> None:
        if current is None or not current.get("id", "").startswith("HP:"):
            return
        term_id = current["id"]
        uri = f"{HPO_URI_PREFIX}{term_id.split(':', 1)[1]}"
        meta: dict[str, Any] = {}
        if current.get("definition"):
            meta["definition"] = {"val": current["definition"]}
        if current.get("synonyms"):
            meta["synonyms"] = current["synonyms"]
        if current.get("comments"):
            meta["comments"] = current["comments"]
        if current.get("xrefs"):
            meta["xrefs"] = [{"val": value} for value in current["xrefs"]]
        if current.get("alt_ids"):
            meta["basicPropertyValues"] = [
                {"pred": "http://www.geneontology.org/formats/oboInOwl#hasAlternativeId", "val": value}
                for value in current["alt_ids"]
            ]
        if current.get("replaced_by"):
            meta["replaced_by"] = current["replaced_by"]
        if current.get("obsolete"):
            meta["deprecated"] = True

        terms.append(
            {
                "id": uri,
                "lbl": current.get("name", ""),
                "type": "CLASS",
                "meta": meta,
            }
        )
        for parent in current.get("parents", []):
            edges.append(
                {
                    "sub": uri,
                    "pred": "is_a",
                    "obj": f"{HPO_URI_PREFIX}{parent.split(':', 1)[1]}",
                }
            )

    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("!"):
                continue
            if line == "[Term]":
                finish_term()
                current = {"synonyms": [], "parents": [], "comments": [], "xrefs": [], "alt_ids": []}
                section = "term"
                continue
            if line.startswith("["):
                finish_term()
                current = None
                section = line
                continue
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            value = value.strip()
            if section != "term" or current is None:
                if key in {"format-version", "data-version", "ontology", "property_value"}:
                    header[key] = value
                continue

            if key == "id":
                current["id"] = value
            elif key == "name":
                current["name"] = value
            elif key == "def":
                current["definition"] = quoted_value(value)
            elif key == "synonym":
                synonym_value = quoted_value(value)
                scope_match = re.match(r'^\s*"(?:\\.|[^"\\])*"\s+([A-Z]+)', value)
                scope = scope_match.group(1) if scope_match else "RELATED"
                if synonym_value:
                    current["synonyms"].append(
                        {"pred": predicate_for_scope(scope), "val": synonym_value}
                    )
            elif key == "is_a":
                parent = value.split("!", 1)[0].strip()
                if parent.startswith("HP:"):
                    current["parents"].append(parent)
            elif key == "comment":
                current["comments"].append(value)
            elif key == "xref":
                current["xrefs"].append(value)
            elif key == "alt_id":
                current["alt_ids"].append(value)
            elif key == "replaced_by":
                current["replaced_by"] = value
            elif key in {"is_obsolete", "deprecated"}:
                current["obsolete"] = value.casefold() == "true"

    finish_term()
    if not terms:
        raise ValueError(f"No HPO terms were found in {path}")
    return {"nodes": terms, "edges": edges, "propertyChainAxioms": []}, header


def term_id(node: dict[str, Any]) -> str:
    return f"HP:{node['id'].rsplit('/HP_', 1)[-1]}"


def write_exports(obo_path: Path, output_json: Path, output_jsonl: Path, output_csv: Path) -> None:
    graph, header = parse_obo(obo_path)
    graph["id"] = "http://purl.obolibrary.org/obo/hp.owl"
    graph["meta"] = {
        "source_obo": obo_path.name,
        "data_version": header.get("data-version", ""),
        "format_version": header.get("format-version", ""),
    }
    payload = {"graphs": [graph]}

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8-sig")

    edges_by_child: dict[str, list[str]] = {}
    for edge in graph["edges"]:
        child = term_id({"id": edge["sub"]})
        parent = term_id({"id": edge["obj"]})
        edges_by_child.setdefault(child, []).append(parent)

    rows: list[dict[str, Any]] = []
    for node in graph["nodes"]:
        hpo_id = term_id(node)
        meta = node.get("meta", {})
        synonyms = meta.get("synonyms", [])
        rows.append(
            {
                "HPOid": hpo_id,
                "name": node.get("lbl", ""),
                "definition": (meta.get("definition") or {}).get("val", ""),
                "synonyms": [
                    {"text": synonym.get("val", ""), "predicate": synonym.get("pred", "")}
                    for synonym in synonyms
                ],
                "comments": meta.get("comments", []),
                "parents": edges_by_child.get(hpo_id, []),
                "is_obsolete": bool(meta.get("deprecated", False)),
                "replaced_by": meta.get("replaced_by", ""),
            }
        )

    with output_jsonl.open("w", encoding="utf-8-sig") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = [
        "HPOid",
        "name",
        "definition",
        "synonyms",
        "comments",
        "parents",
        "is_obsolete",
        "replaced_by",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "synonyms": json.dumps(row["synonyms"], ensure_ascii=False),
                    "comments": json.dumps(row["comments"], ensure_ascii=False),
                    "parents": json.dumps(row["parents"], ensure_ascii=False),
                }
            )

    print(f"Source: {obo_path} ({header.get('data-version', 'unknown version')})")
    print(f"Terms: {len(rows):,}; is_a edges: {len(graph['edges']):,}")
    print(f"JSON: {output_json}")
    print(f"JSONL: {output_jsonl}")
    print(f"CSV: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HPO OBO to JSON, JSONL, and CSV")
    parser.add_argument("--input", type=Path, default=Path("hp260623.obo"))
    parser.add_argument("--json", dest="output_json", type=Path)
    parser.add_argument("--jsonl", dest="output_jsonl", type=Path)
    parser.add_argument("--csv", dest="output_csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stem = args.input.with_suffix("")
    write_exports(
        args.input,
        args.output_json or stem.with_suffix(".json"),
        args.output_jsonl or stem.with_suffix(".jsonl"),
        args.output_csv or stem.with_suffix(".csv"),
    )


if __name__ == "__main__":
    main()
