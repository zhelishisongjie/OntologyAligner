from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import ontology_aligner_runtime as core
from run_main_experiment import (
    DATASET_ORDER,
    dataset_dir,
    experiment_dir,
    llm_cache_dir,
)


METHODS = ("OAR", "LCR", "OntologyAligner")
RESULTS_DIR = experiment_dir()
LLM_CACHE_DIR = llm_cache_dir()


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_complete_records(dataset: str, name: str) -> list[dict[str, Any]]:
    path = dataset_dir(dataset) / name
    records = core.load_jsonl(path)
    if not records:
        raise FileNotFoundError(f"Missing completed main-experiment output: {path}")
    if any(not record.get("gold_id") for record in records):
        raise ValueError(f"{path} has not been evaluated")
    return records


def aggregate_method(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(record["correct"]) for record in records)
    return {
        "rows": len(records),
        "correct": correct,
        "micro_top1": correct / len(records),
        "no_match": sum(bool(record.get("no_match")) for record in records),
    }


def cache_usage(path: Path) -> dict[str, Any]:
    successful = {
        str(item["key"]): item
        for item in core.load_jsonl(path)
        if item.get("success")
    }
    attempts = sum(int(item.get("attempts", 0)) for item in successful.values())
    return {
        "unique_requests": len(successful),
        "attempts": attempts,
        "retries": attempts - len(successful),
        "prompt_tokens": sum(
            int(item.get("prompt_tokens", 0)) for item in successful.values()
        ),
        "completion_tokens": sum(
            int(item.get("completion_tokens", 0)) for item in successful.values()
        ),
        "total_tokens": sum(
            int(item.get("total_tokens", 0)) for item in successful.values()
        ),
        "api_seconds": round(
            sum(float(item.get("api_seconds", 0.0)) for item in successful.values()), 3
        ),
    }


def build_summary() -> dict[str, Any]:
    dataset_summaries: dict[str, Any] = {}
    all_records = {method: [] for method in METHODS}
    per_dataset_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    file_names = {
        "OAR": "oar_records.jsonl",
        "LCR": "lcr_records.jsonl",
        "OntologyAligner": "final_records.jsonl",
    }
    for dataset in DATASET_ORDER:
        summary_path = dataset_dir(dataset) / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Dataset is not complete: {dataset}")
        dataset_summaries[dataset] = json.loads(
            summary_path.read_text(encoding="utf-8-sig")
        )
        per_dataset_records[dataset] = {}
        for method, file_name in file_names.items():
            records = load_complete_records(dataset, file_name)
            per_dataset_records[dataset][method] = records
            all_records[method].extend(records)

    methods: dict[str, Any] = {}
    for method in METHODS:
        metrics = aggregate_method(all_records[method])
        metrics["macro_top1"] = float(
            np.mean(
                [
                    dataset_summaries[dataset]["methods"][method]["top1_accuracy"]
                    for dataset in DATASET_ORDER
                ]
            )
        )
        methods[method] = metrics

    comparisons = {
        "OAR_to_LCR": core.paired_comparison(
            all_records["OAR"], all_records["LCR"], "OAR -> LCR"
        ),
        "LCR_to_OntologyAligner": core.paired_comparison(
            all_records["LCR"],
            all_records["OntologyAligner"],
            "LCR -> OntologyAligner",
        ),
    }
    hgr = {
        "triggered_samples": sum(
            dataset_summaries[dataset]["hgr"]["triggered_samples"]
            for dataset in DATASET_ORDER
        ),
        "changed_top1": sum(
            dataset_summaries[dataset]["hgr"]["changed_top1"]
            for dataset in DATASET_ORDER
        ),
        "wrong_to_right": sum(
            dataset_summaries[dataset]["hgr"]["wrong_to_right"]
            for dataset in DATASET_ORDER
        ),
        "right_to_wrong": sum(
            dataset_summaries[dataset]["hgr"]["right_to_wrong"]
            for dataset in DATASET_ORDER
        ),
    }
    hgr["trigger_rate"] = hgr["triggered_samples"] / methods["OntologyAligner"]["rows"]

    summary = {
        "generated_at": core.utc_now(),
        "llm_model": str(core.load_config()["llm"]["model"]),
        "dataset_order": list(DATASET_ORDER),
        "dataset_summaries": dataset_summaries,
        "methods": methods,
        "comparisons": comparisons,
        "hgr": hgr,
        "api_usage": {
            "LCR": cache_usage(LLM_CACHE_DIR / "lcr_responses.jsonl"),
            "HGR": cache_usage(LLM_CACHE_DIR / "hgr_responses.jsonl"),
        },
    }
    core.write_json(RESULTS_DIR / "main_experiment_summary.json", summary)
    return summary


def write_summary_workbook(summary: dict[str, Any]) -> None:
    rows = []
    for dataset in summary["dataset_order"]:
        methods = summary["dataset_summaries"][dataset]["methods"]
        hgr = summary["dataset_summaries"][dataset]["hgr"]
        rows.append(
            {
                "dataset": dataset,
                "samples": methods["OntologyAligner"]["rows"],
                "OAR_top1": methods["OAR"]["top1_accuracy"],
                "LCR_top1": methods["LCR"]["top1_accuracy"],
                "OntologyAligner_top1": methods["OntologyAligner"]["top1_accuracy"],
                "HGR_triggered": hgr["triggered_samples"],
                "HGR_wrong_to_right": hgr["wrong_to_right"],
                "HGR_right_to_wrong": hgr["right_to_wrong"],
            }
        )
    rows.append(
        {
            "dataset": "Macro",
            "samples": summary["methods"]["OntologyAligner"]["rows"],
            "OAR_top1": summary["methods"]["OAR"]["macro_top1"],
            "LCR_top1": summary["methods"]["LCR"]["macro_top1"],
            "OntologyAligner_top1": summary["methods"]["OntologyAligner"]["macro_top1"],
            "HGR_triggered": summary["hgr"]["triggered_samples"],
            "HGR_wrong_to_right": summary["hgr"]["wrong_to_right"],
            "HGR_right_to_wrong": summary["hgr"]["right_to_wrong"],
        }
    )
    pd.DataFrame(rows).to_excel(
        RESULTS_DIR / "main_experiment_summary.xlsx",
        index=False,
        sheet_name="Top1",
    )


def generate_html(summary: dict[str, Any]) -> str:
    methods = summary["methods"]
    hgr = summary["hgr"]
    comparison = summary["comparisons"]["LCR_to_OntologyAligner"]
    dataset_rows = "".join(
        "<tr>"
        f"<td><a href=\"{html.escape(dataset)}/OntologyAligner_predictions.xlsx\">{html.escape(dataset)}</a></td>"
        f"<td>{metrics['OntologyAligner']['rows']:,}</td>"
        f"<td>{pct(metrics['OAR']['top1_accuracy'])}</td>"
        f"<td>{pct(metrics['LCR']['top1_accuracy'])}</td>"
        f"<td><strong>{pct(metrics['OntologyAligner']['top1_accuracy'])}</strong></td>"
        f"<td class=\"{'positive' if delta > 0 else 'negative' if delta < 0 else 'neutral'}\">{delta * 100:+.2f} pp</td>"
        f"<td>{dataset_hgr['wrong_to_right']} / {dataset_hgr['right_to_wrong']}</td>"
        "</tr>"
        for dataset in summary["dataset_order"]
        for dataset_summary in [summary["dataset_summaries"][dataset]]
        for metrics in [dataset_summary["methods"]]
        for dataset_hgr in [dataset_summary["hgr"]]
        for delta in [
            metrics["OntologyAligner"]["top1_accuracy"]
            - metrics["LCR"]["top1_accuracy"]
        ]
    )
    usage_rows = "".join(
        "<tr>"
        f"<td>{stage}</td><td>{usage['unique_requests']:,}</td>"
        f"<td>{usage['retries']:,}</td><td>{usage['prompt_tokens']:,}</td>"
        f"<td>{usage['completion_tokens']:,}</td><td>{usage['total_tokens']:,}</td>"
        "</tr>"
        for stage, usage in summary["api_usage"].items()
    )
    macro_delta = (
        methods["OntologyAligner"]["macro_top1"] - methods["LCR"]["macro_top1"]
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OntologyAligner Main Experiment</title>
<style>
:root{{--paper:#f5f2ea;--surface:#fffdf7;--ink:#15201c;--muted:#657069;--line:#bdc6c0;--teal:#006b67;--rust:#a13f2d;--signal:#b8e2dc}}
*{{box-sizing:border-box}}html{{background:var(--paper)}}body{{margin:0;color:var(--ink);font:15px/1.55 "Microsoft YaHei","Noto Sans CJK SC",sans-serif;letter-spacing:0}}main{{max-width:1180px;margin:auto;padding:34px 28px 64px}}
header{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:28px;align-items:end;padding:26px 0 24px;border-top:8px solid var(--ink);border-bottom:1px solid var(--ink)}}.eyebrow{{font:700 12px/1.2 Consolas,monospace;color:var(--teal)}}h1{{font:700 38px/1.12 Georgia,"Microsoft YaHei",serif;margin:9px 0;overflow-wrap:anywhere}}header p,p{{margin:0;color:var(--muted)}}.stamp{{font:12px/1.7 Consolas,monospace;text-align:right;color:var(--muted)}}
.hero{{display:grid;grid-template-columns:1.3fr .7fr;margin:28px 0;border:1px solid var(--ink);background:var(--surface)}}.hero-copy{{padding:28px}}.hero h2{{font:700 24px/1.3 Georgia,"Microsoft YaHei",serif;margin:0 0 8px}}.hero-number{{display:grid;place-content:center;text-align:center;border-left:1px solid var(--ink);background:var(--signal)}}.hero-number strong{{font:700 38px/1 Consolas,monospace;color:var(--teal)}}.hero-number span{{margin-top:9px;font-size:12px}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--line);background:var(--surface)}}.metric{{padding:18px;border-right:1px solid var(--line)}}.metric:last-child{{border:0}}.metric span{{display:block;color:var(--muted);font-size:12px}}.metric strong{{display:block;margin-top:5px;font:700 25px/1.2 Consolas,monospace}}
section{{margin-top:34px}}section h2{{margin:0 0 11px;font:700 19px/1.3 Georgia,"Microsoft YaHei",serif}}.table-wrap{{overflow:auto;border-top:2px solid var(--ink);border-bottom:1px solid var(--line);background:var(--surface)}}table{{width:100%;min-width:760px;border-collapse:collapse}}th,td{{padding:11px 13px;text-align:left;border-bottom:1px solid var(--line)}}th{{font-size:12px;color:var(--muted);background:#ebe9e2}}tbody tr:last-child td{{border-bottom:0}}a{{color:var(--ink);text-decoration-thickness:1px;text-underline-offset:3px}}.positive{{color:var(--teal);font-weight:700}}.negative{{color:var(--rust);font-weight:700}}.neutral{{color:var(--muted);font-weight:700}}
.note{{padding:15px 18px;border-left:4px solid var(--teal);background:#e5eeea;color:var(--ink)}}footer{{margin-top:38px;padding-top:15px;border-top:1px solid var(--ink);color:var(--muted);font-size:12px}}
@media(max-width:760px){{main{{padding:20px 12px 40px}}header{{grid-template-columns:1fr;gap:16px}}.stamp{{text-align:left}}h1{{font-size:29px}}.hero{{grid-template-columns:1fr}}.hero-copy{{padding:22px 18px}}.hero-number{{border-left:0;border-top:1px solid var(--ink);padding:22px}}.metrics{{grid-template-columns:1fr 1fr}}.metric:nth-child(2){{border-right:0}}.metric:nth-child(-n+2){{border-bottom:1px solid var(--line)}}.audit{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div><div class="eyebrow">MAIN EXPERIMENT</div><h1>OntologyAligner<br>七数据集主实验</h1><p>OAR → LCR → HGR · 全量样本</p></div><div class="stamp">7 datasets / {methods['OntologyAligner']['rows']:,} rows<br>HPO 2026-06-23 / Top-20</div></header>
<div class="hero"><div class="hero-copy"><h2>完整流程 Macro Top-1</h2><p>七个数据集等权平均；Micro Top-1 为 {pct(methods['OntologyAligner']['micro_top1'])}。</p></div><div class="hero-number"><strong>{pct(methods['OntologyAligner']['macro_top1'])}</strong><span>OntologyAligner</span></div></div>
<div class="metrics"><div class="metric"><span>样本总数</span><strong>{methods['OntologyAligner']['rows']:,}</strong></div><div class="metric"><span>OAR Macro</span><strong>{pct(methods['OAR']['macro_top1'])}</strong></div><div class="metric"><span>LCR Macro</span><strong>{pct(methods['LCR']['macro_top1'])}</strong></div><div class="metric"><span>HGR Macro Δ</span><strong class="{'positive' if macro_delta > 0 else 'negative' if macro_delta < 0 else 'neutral'}">{macro_delta * 100:+.2f} pp</strong></div></div>
<section><h2>逐数据集 Top-1</h2><div class="table-wrap"><table><thead><tr><th>数据集</th><th>样本</th><th>OAR</th><th>LCR</th><th>OntologyAligner</th><th>HGR Δ</th><th>改对 / 改错</th></tr></thead><tbody>{dataset_rows}</tbody></table></div></section>
<section><h2>HGR 总体行为</h2><div class="table-wrap"><table><thead><tr><th>触发样本</th><th>触发率</th><th>改变 Top-1</th><th>改对</th><th>改错</th><th>Macro 95% CI</th><th>McNemar p</th></tr></thead><tbody><tr><td>{hgr['triggered_samples']:,}</td><td>{pct(hgr['trigger_rate'])}</td><td>{hgr['changed_top1']:,}</td><td class="positive">{hgr['wrong_to_right']}</td><td class="negative">{hgr['right_to_wrong']}</td><td>[{comparison['bootstrap_95_ci'][0] * 100:+.2f}, {comparison['bootstrap_95_ci'][1] * 100:+.2f}] pp</td><td>{comparison['mcnemar_exact_p']:.4f}</td></tr></tbody></table></div></section>
<section><h2>主实验 API 使用</h2><div class="table-wrap"><table><thead><tr><th>阶段</th><th>唯一请求</th><th>重试</th><th>Prompt tokens</th><th>Completion tokens</th><th>Total tokens</th></tr></thead><tbody>{usage_rows}</tbody></table></div></section>
<section><p class="note"><strong>口径：</strong>全部 13,390 行均参与评分，重复 mention 保留；相同 prompt 在本次主实验内部按 key 复用。LCR 的 No Match 作为有效未匹配预测保留，HGR 必须返回完整排序；没有 retrieval fallback。主指标为七数据集等权 Macro Top-1。</p></section>
<footer>生成时间：{html.escape(summary['generated_at'])} · 模型：{html.escape(summary['llm_model'])} · 结果目录：{html.escape(str(RESULTS_DIR))}</footer>
</main></body></html>"""


def main() -> None:
    summary = build_summary()
    write_summary_workbook(summary)
    report_path = RESULTS_DIR / "main_experiment_report.html"
    report_path.write_text(generate_html(summary), encoding="utf-8-sig")
    print(f"summary={RESULTS_DIR / 'main_experiment_summary.json'}")
    print(f"workbook={RESULTS_DIR / 'main_experiment_summary.xlsx'}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
