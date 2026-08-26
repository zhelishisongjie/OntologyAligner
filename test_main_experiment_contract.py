from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

import run_main_experiment as main


ROOT = Path(__file__).resolve().parent


def test_production_modules_do_not_import_ablation_runner() -> None:
    for name in (
        "run_main_experiment.py",
        "generate_main_experiment_report.py",
    ):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8-sig"))
        imported = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert "run_test_ablation" not in imported


def notebook_code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def assigned_integer(code: str, name: str) -> int:
    for node in ast.parse(code).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            return int(ast.literal_eval(node.value))
    raise AssertionError(f"{name} was not assigned")


def synthetic_records() -> tuple[dict, dict, dict]:
    terms = main.core.load_ontology()
    candidate_ids = list(terms)[:20]
    sample = {
        "sample_id": "fgdd:000000",
        "dataset_key": "fgdd",
        "source_sample_id": 0,
        "source_excel_row": 2,
        "query": "synthetic phenotype",
        "preferred_name": "Synthetic phenotype",
        "gold_id": candidate_ids[0],
        "accepted_ids": [candidate_ids[0]],
        "metadata": {"patient_ids": "", "pmids": ""},
    }
    oar = {
        **sample,
        "method": "OAR",
        "candidate_ids": candidate_ids,
        "candidate_scores": [round(1.0 - rank / 100.0, 10) for rank in range(20)],
        "candidate_matched_surfaces": [terms[ontology_id].name for ontology_id in candidate_ids],
        "predicted_id": candidate_ids[0],
        "no_match": False,
        "correct": True,
    }
    final = {
        **oar,
        "method": "OntologyAligner",
        "ranking": candidate_ids,
        "initial_ranking": candidate_ids,
        "initial_predicted_id": candidate_ids[0],
        "graph_rerank_applied": False,
        "hpo_graph_context": [],
    }
    return sample, oar, final


def test_notebooks_share_top20_contract() -> None:
    precompute_code = notebook_code(ROOT / "01_run_precompute.ipynb")
    rerank_code = notebook_code(ROOT / "02_run_LLM_rerank.ipynb")

    assert assigned_integer(precompute_code, "RETRIEVAL_TOP_K") == 20
    assert assigned_integer(rerank_code, "LLM_CANDIDATE_COUNT") == 20
    assert "retrieval_top200" not in rerank_code
    assert "Top-200" not in precompute_code
    assert "Top-200" not in rerank_code


def test_runner_paths_match_notebook_names(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "PRECOMPUTE_RESULTS_DIR", tmp_path / "precompute")
    monkeypatch.setattr(main, "RERANK_RESULTS_DIR", tmp_path / "rerank")
    monkeypatch.setattr(main.core, "CONFIG_PATH", ROOT / "LLM_config.example.json")

    assert main.precompute_workbook_path("fgdd").name == (
        "fgdd_phenotype_retrieval_top20_"
        "text-embedding-3-large_oar_dense-only.xlsx"
    )
    model_tag = main.safe_tag(str(main.core.load_config()["llm"]["model"]))
    assert main.rerank_workbook_path("fgdd").name == (
        "fgdd_phenotype_llm_rerank_candidates20_"
        f"text-embedding-3-large_oar_{model_tag}_hpo-graph.xlsx"
    )


def test_records_map_to_notebook_columns() -> None:
    terms = main.core.load_ontology()
    _, oar, final = synthetic_records()

    precompute_frame = main.build_precompute_frame([oar], terms)
    rerank_frame = main.build_rerank_frame([final], terms)

    assert len(json.loads(precompute_frame.iloc[0]["candidate_ids_json"])) == 20
    assert len(json.loads(rerank_frame.iloc[0]["ranked_ids_json"])) == 20
    assert precompute_frame.iloc[0]["top_01_id"] == oar["candidate_ids"][0]
    assert rerank_frame.iloc[0]["predicted_top1_id"] == final["predicted_id"]


def test_runner_reads_notebook_precompute_workbook(tmp_path: Path) -> None:
    terms = main.core.load_ontology()
    sample, oar, _ = synthetic_records()
    workbook = tmp_path / "precompute.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        main.build_precompute_frame([oar], terms).to_excel(
            writer, sheet_name="Samples", index=False
        )
        main.config_frame(
            {
                "dataset_key": "fgdd_phenotype",
                "retrieval_top_k": 20,
                "retrieval_method": "oar_dense_cosine",
            }
        ).to_excel(writer, sheet_name="Run_Config", index=False)

    loaded = main.load_precompute_workbook(workbook, "fgdd", [sample])

    assert loaded[0]["candidate_ids"] == oar["candidate_ids"]
    assert loaded[0]["candidate_scores"] == oar["candidate_scores"]


def test_lcr_allows_no_match_and_hgr_rejects_it() -> None:
    terms = main.core.load_ontology()
    _, oar, final = synthetic_records()
    final["hpo_graph_context"] = [{"relation": "synthetic"}]

    lcr_task = main.core.build_lcr_task(oar, terms, "TEST_LCR", "test-model")
    hgr_task = main.core.build_hgr_task(final, terms, "test-model", "TEST_HGR")

    assert lcr_task["allow_no_match"] is True
    assert hgr_task["allow_no_match"] is False
    assert main.core.parse_ranking("No Match", lcr_task["allowed_ids"], True) == (
        [],
        True,
    )
    with pytest.raises(ValueError, match="HGR must return a complete ranking"):
        main.core.parse_ranking("No Match", hgr_task["allowed_ids"], False)
