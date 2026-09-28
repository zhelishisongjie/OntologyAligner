from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import chromadb
import torch

import ontology_aligner_runtime as core
from ontology_aligner_oar import load_default_oar_runtime

from . import config, oar, report, runtime


def experiment_config() -> dict[str, Any]:
    settings = core.load_config()
    values = {
        "protocol": "OntologyAligner Ablation v1",
        "seed": config.SEED,
        "full_sample_count": config.FULL_SAMPLE_COUNT,
        "subset_sample_count": config.SUBSET_SAMPLE_COUNT,
        "subset_rows_per_dataset": config.SUBSET_ROWS_PER_DATASET,
        "dataset_order": list(config.DATASET_ORDER),
        "candidate_k_values": list(config.CANDIDATE_K_VALUES),
        "max_concurrency": config.MAX_CONCURRENCY,
        "requests_per_minute": config.REQUESTS_PER_MINUTE,
        "max_attempts": config.MAX_ATTEMPTS,
        "pause_on_429_seconds": config.PAUSE_ON_429_SECONDS,
        "gpt_model": settings["llm"]["model"],
        "claude_model": settings["llm2"]["model"],
        "temperature": core.LLM_TEMPERATURE,
        "hgr_probe_top_k": core.HPO_GRAPH_PROBE_TOP_K,
        "hgr_max_ancestor_distance": core.HPO_MAX_ANCESTOR_DISTANCE,
        "oar_training": {
            "epochs": config.OAR_EPOCHS,
            "checkpoint_epoch": config.OAR_CHECKPOINT_EPOCH,
            "batch_size": config.OAR_BATCH_SIZE,
            "learning_rate": config.OAR_LEARNING_RATE,
            "temperature": config.OAR_TEMPERATURE,
            "weight_decay": config.OAR_WEIGHT_DECAY,
            "identity_regularization": config.OAR_IDENTITY_REGULARIZATION,
            "gradient_clip": config.OAR_GRADIENT_CLIP,
            "hard_negative_query_k": config.OAR_HARD_NEGATIVE_QUERY_K,
            "negative_concepts": config.OAR_NEGATIVE_CONCEPTS,
        },
        "backbones": [asdict(spec) for spec in config.BACKBONES],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
        "created_at": core.utc_now(),
    }
    return values


def ensure_config() -> dict[str, Any]:
    config.ensure_directories()
    path = config.RESULTS_DIR / "experiment_config.json"
    values = experiment_config()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    core.write_json(path, values)
    return values


def ensure_main_outputs() -> None:
    from run_main_experiment import dataset_dir, run_dataset

    for dataset in config.DATASET_ORDER:
        main_dataset = config.MAIN_DATASET_KEYS[dataset]
        complete = (dataset_dir(main_dataset) / "summary.json").exists()
        if complete and all(runtime.main_record_path(dataset, stage).exists() for stage in ("oar", "lcr", "final")):
            continue
        print(f"Preparing main experiment results for {main_dataset}", flush=True)
        run_dataset(main_dataset, config.MAX_CONCURRENCY, config.REQUESTS_PER_MINUTE)


def annotate(
    records: Sequence[dict[str, Any]],
    experiment: str,
    condition: str,
    **values: Any,
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        item = dict(record)
        item.update(
            {
                "experiment": experiment,
                "condition": condition,
                **values,
            }
        )
        output.append(item)
    return output


def final_stage(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        item = dict(record)
        ranking = list(item.get("ranking", item.get("candidate_ids", [])))
        item["ranking"] = ranking
        item.setdefault("initial_ranking", ranking)
        item.setdefault(
            "initial_predicted_id", ranking[0] if ranking else "No Match"
        )
        item.setdefault("graph_triggered", False)
        item.setdefault("graph_rerank_applied", False)
        item["predicted_id"] = ranking[0] if ranking else "No Match"
        item["no_match"] = not ranking
        item["correct"] = item["predicted_id"] in set(item["accepted_ids"])
        output.append(item)
    return output


def paired(
    baseline: Sequence[dict[str, Any]],
    treatment: Sequence[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    return core.paired_comparison(
        final_stage(baseline), final_stage(treatment), label
    )


def base_run_config(
    experiment: str,
    scope: str,
    rows: int,
) -> dict[str, Any]:
    ensure_config()
    return {
        "experiment": experiment,
        "scope": scope,
        "rows": rows,
        "datasets": list(config.DATASET_ORDER),
        "temperature": core.LLM_TEMPERATURE,
        "max_concurrency": config.MAX_CONCURRENCY,
        "requests_per_minute": config.REQUESTS_PER_MINUTE,
        "max_attempts": config.MAX_ATTEMPTS,
        "pause_on_429_seconds": config.PAUSE_ON_429_SECONDS,
        "completed_at": core.utc_now(),
    }


def run_e2() -> Path:
    full = runtime.load_full_stage("oar")
    records = annotate(final_stage(full), "E2", "OAR only", candidate_k=20)
    baseline = runtime.load_full_stage("final")
    run_config = base_run_config("E2", "full", len(records))
    run_config["definition"] = "current OAR Top-1; no LCR or HGR"
    run_config["paired_comparisons"] = {
        "OVERALL": paired(baseline, records, "Full OntologyAligner -> E2 OAR only")
    }
    return report.write_experiment_workbook("E2", records, [], run_config)


def run_e3() -> Path:
    full = runtime.load_full_stage("lcr")
    records = annotate(final_stage(full), "E3", "OAR + LCR", candidate_k=20)
    baseline = runtime.load_full_stage("final")
    run_config = base_run_config("E3", "full", len(records))
    run_config["definition"] = "current OAR Top-20 -> LCR; no HGR"
    run_config["paired_comparisons"] = {
        "OVERALL": paired(baseline, records, "Full OntologyAligner -> E3 OAR + LCR")
    }
    return report.write_experiment_workbook("E3", records, [], run_config)


def checkpoint_path(experiment: str, dataset: str, stage: str) -> Path:
    return config.RECORDS_DIR / experiment / dataset / f"{stage}.jsonl"


def run_llm_pipeline(
    experiment: str,
    retrieval: Sequence[dict[str, Any]],
    config_key: str,
    stage_suffix: str = "",
) -> list[dict[str, Any]]:
    grouped = runtime.group_by_dataset(retrieval)
    lcr_all: list[dict[str, Any]] = []
    for dataset in config.DATASET_ORDER:
        path = checkpoint_path(experiment, dataset, f"lcr{stage_suffix}")
        records = runtime.read_complete_checkpoint(path, len(grouped[dataset]))
        if records is None:
            records = runtime.run_lcr(
                grouped[dataset], config_key, f"ABLATION_{experiment}{stage_suffix}_LCR"
            )
            runtime.write_checkpoint(path, records)
        lcr_all.extend(records)
    lcr_grouped = runtime.group_by_dataset(lcr_all)
    final_all: list[dict[str, Any]] = []
    for dataset in config.DATASET_ORDER:
        path = checkpoint_path(experiment, dataset, f"final{stage_suffix}")
        records = runtime.read_complete_checkpoint(path, len(lcr_grouped[dataset]))
        if records is None:
            records = runtime.run_hgr(
                lcr_grouped[dataset],
                config_key,
                f"ABLATION_{experiment}{stage_suffix}_HGR",
            )
            runtime.write_checkpoint(path, records)
        final_all.extend(records)
    return final_all


def raw_retrieval_full_te3l(experiment: str = "E1") -> list[dict[str, Any]]:
    spec = config.BACKBONE_BY_KEY["text_embedding_3_large"]
    collection = runtime.raw_collection(spec)
    samples = runtime.load_full_stage("oar")
    grouped = runtime.group_by_dataset(samples)
    output: list[dict[str, Any]] = []
    for dataset in config.DATASET_ORDER:
        path = checkpoint_path(experiment, dataset, "retrieval")
        records = runtime.read_complete_checkpoint(path, len(grouped[dataset]))
        if records is None:
            vectors, stats = runtime.resolve_query_embeddings(
                spec, [record["query"] for record in grouped[dataset]]
            )
            print(f"{experiment}/{dataset}: embedding stats={stats}", flush=True)
            records = runtime.retrieve_records(
                grouped[dataset], collection, vectors, "Raw TE3L cosine"
            )
            runtime.write_checkpoint(path, records)
        output.extend(records)
    return output


def run_e1() -> Path:
    retrieval = raw_retrieval_full_te3l("E1")
    final = run_llm_pipeline("E1", retrieval, "llm")
    records = annotate(
        final,
        "E1",
        "Raw text-embedding-3-large",
        embedding_model="text-embedding-3-large",
        candidate_k=20,
    )
    baseline = runtime.load_full_stage("final")
    run_config = base_run_config("E1", "full", len(records))
    run_config.update(
        {
            "definition": "Raw TE3L cosine -> Top-20 -> GPT LCR -> HGR",
            "llm_model": core.load_config()["llm"]["model"],
            "raw_collection": config.BACKBONE_BY_KEY[
                "text_embedding_3_large"
            ].collection_name,
            "paired_comparisons": {
                "OVERALL": paired(
                    baseline, records, "Full OntologyAligner -> E1 without OAR"
                )
            },
        }
    )
    return report.write_experiment_workbook("E1", records, [], run_config)


def run_e4(llm_backbone: str = "claude-opus-5") -> Path:
    if llm_backbone not in config.E4_LLM_CONFIG_KEYS:
        raise ValueError(f"Unknown E4 LLM backbone: {llm_backbone}")
    config_key = config.E4_LLM_CONFIG_KEYS[llm_backbone]
    settings = core.load_config()[config_key]
    model = str(settings["model"])
    if model != llm_backbone:
        raise ValueError(
            f"E4 config section {config_key!r} declares model {model!r}, "
            f"expected {llm_backbone!r}"
        )
    retrieval = runtime.load_subset_stage("oar")
    experiment_key = (
        "E4"
        if llm_backbone == "claude-opus-5"
        else f"E4_{llm_backbone.replace('-', '_')}"
    )
    final = run_llm_pipeline(experiment_key, retrieval, config_key)
    records = annotate(
        final,
        "E4",
        model,
        embedding_model="text-embedding-3-large",
        candidate_k=20,
    )
    baseline = runtime.load_subset_stage("final")
    run_config = base_run_config(
        "E4",
        "Ablation_Subset",
        len(records),
    )
    run_config.update(
        {
            "definition": f"current OAR Top-20 -> {model} LCR -> {model} HGR",
            "llm_config_section": config_key,
            "llm_model": model,
            "llm_request_options": core.llm_request_options(model),
            "paired_comparisons": {
                "OVERALL": paired(
                    baseline, records, f"GPT baseline -> {model} E4"
                )
            },
        }
    )
    return report.write_experiment_workbook(
        "E4", records, [], run_config, config.E4_LLM_FILES[llm_backbone]
    )


def subset_from_full(
    full: Sequence[dict[str, Any]], subset: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    lookup = {
        (record["dataset_key"], int(record["source_excel_row"])): record
        for record in full
    }
    return [
        lookup[(record["dataset_key"], int(record["source_excel_row"]))]
        for record in subset
    ]


def add_raw_candidates(
    final: Sequence[dict[str, Any]], raw: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    raw_by_id = {record["sample_id"]: record for record in raw}
    output = []
    for record in final:
        source = raw_by_id[record["sample_id"]]
        item = dict(record)
        item["raw_candidate_ids"] = list(source["candidate_ids"])
        item["raw_candidate_scores"] = list(source["candidate_scores"])
        item["raw_candidate_matched_surfaces"] = list(
            source["candidate_matched_surfaces"]
        )
        output.append(item)
    return output


def run_e5_backbone(
    spec: config.BackboneSpec, subset: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    experiment_key = f"E5_{spec.key}"
    if not spec.train_oar:
        raw_full = raw_retrieval_full_te3l("E1")
        raw = subset_from_full(raw_full, subset)
        final = runtime.load_subset_stage("final")
        return add_raw_candidates(final, raw)

    raw_collection = runtime.raw_collection(spec)
    projected_collection, _ = oar.build_projected_collection(spec)
    vectors, stats = runtime.resolve_query_embeddings(
        spec, [record["query"] for record in subset]
    )
    print(f"E5/{spec.key}: embedding stats={stats}", flush=True)
    raw_path = config.RECORDS_DIR / experiment_key / "raw_retrieval.jsonl"
    raw = runtime.read_complete_checkpoint(raw_path, len(subset))
    if raw is None:
        raw = runtime.retrieve_records(
            subset, raw_collection, vectors, f"Raw {spec.model_name} cosine"
        )
        runtime.write_checkpoint(raw_path, raw)
    projected_path = config.RECORDS_DIR / experiment_key / "projected_retrieval.jsonl"
    projected = runtime.read_complete_checkpoint(projected_path, len(subset))
    if projected is None:
        projected_vectors = oar.project_queries(spec, vectors)
        projected = runtime.retrieve_records(
            subset,
            projected_collection,
            projected_vectors,
            f"OAR {spec.key} cosine",
        )
        runtime.write_checkpoint(projected_path, projected)
    final = run_llm_pipeline(experiment_key, projected, "llm", f"_{spec.key}")
    return add_raw_candidates(final, raw)


def run_e5(backbone: str = "all") -> Path:
    subset = runtime.load_subset_stage("oar")
    keys = (
        [spec.key for spec in config.BACKBONES]
        if backbone == "all"
        else [backbone]
    )
    invalid = set(keys) - set(config.BACKBONE_BY_KEY)
    if invalid:
        raise ValueError(f"Unknown E5 backbones: {sorted(invalid)}")
    records: list[dict[str, Any]] = []
    by_key: dict[str, list[dict[str, Any]]] = {}
    for key in keys:
        spec = config.BACKBONE_BY_KEY[key]
        values = run_e5_backbone(spec, subset)
        values = annotate(
            values,
            "E5",
            spec.key,
            backbone_key=spec.key,
            embedding_model=spec.model_name,
            candidate_k=20,
        )
        by_key[key] = values
        records.extend(values)
    if backbone != "all":
        cache = config.RECORDS_DIR / "E5" / f"{backbone}_formal_records.jsonl"
        runtime.write_checkpoint(cache, records)
        print(f"E5 backbone {backbone} completed; run --backbone all to assemble Excel")
        return cache
    baseline = by_key["text_embedding_3_large"]
    comparisons = {
        key: paired(baseline, values, f"TE3L OAR baseline -> {key}")
        for key, values in by_key.items()
    }
    run_config = base_run_config("E5", "Ablation_Subset x 6 backbones", len(records))
    run_config.update(
        {
            "definition": "six raw backbones; five independently trained native-D OAR models; projected Top-20 -> GPT LCR -> HGR",
            "backbones": [asdict(spec) for spec in config.BACKBONES],
            "oar_training": ensure_config()["oar_training"],
            "llm_model": core.load_config()["llm"]["model"],
            "paired_comparisons": comparisons,
        }
    )
    return report.write_experiment_workbook(
        "E5", records, ["backbone_key"], run_config
    )


def run_e6() -> Path:
    source = runtime.load_subset_stage("oar")
    main_final = runtime.load_subset_stage("final")
    records: list[dict[str, Any]] = []
    by_k: dict[int, list[dict[str, Any]]] = {}
    for candidate_k in config.CANDIDATE_K_VALUES:
        if candidate_k == 1:
            final = final_stage(runtime.prefix_candidates(source, 1))
        elif candidate_k == 20:
            final = main_final
        else:
            retrieval = runtime.prefix_candidates(source, candidate_k)
            final = run_llm_pipeline(
                f"E6_K{candidate_k}", retrieval, "llm", f"_K{candidate_k}"
            )
        values = annotate(
            final,
            "E6",
            f"K={candidate_k}",
            candidate_k=candidate_k,
            embedding_model="text-embedding-3-large",
        )
        by_k[candidate_k] = values
        records.extend(values)
    baseline = by_k[20]
    comparisons = {
        str(candidate_k): paired(
            baseline, values, f"K=20 baseline -> K={candidate_k}"
        )
        for candidate_k, values in by_k.items()
    }
    run_config = base_run_config("E6", "Ablation_Subset x 5 K values", len(records))
    run_config.update(
        {
            "definition": "OAR Top-20 ordered prefixes; K=1 derived, K=3/5/10 rerun, K=20 reused",
            "candidate_k_values": list(config.CANDIDATE_K_VALUES),
            "llm_model": core.load_config()["llm"]["model"],
            "paired_comparisons": comparisons,
        }
    )
    return report.write_experiment_workbook("E6", records, ["candidate_k"], run_config)


def generate_summary() -> Path:
    return report.generate_summary_workbook(ensure_config())


def run_experiment(
    experiment: str,
    backbone: str = "all",
    llm_backbone: str = "claude-opus-5",
) -> Path:
    ensure_main_outputs()
    ensure_config()
    functions = {
        "E1": run_e1,
        "E2": run_e2,
        "E3": run_e3,
        "E4": lambda: run_e4(llm_backbone),
        "E5": lambda: run_e5(backbone),
        "E6": run_e6,
    }
    return functions[experiment]()


def run_all() -> list[Path]:
    ensure_main_outputs()
    paths = [run_e2(), run_e3(), run_e1(), run_e4(), run_e5("all"), run_e6()]
    paths.append(generate_summary())
    return paths
