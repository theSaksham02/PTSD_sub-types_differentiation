"""
tables.py — Copy-paste-ready LaTeX + Markdown result tables for the manuscript.

Two tables:
  1. Emotion classification  ->  Emotion Class | Precision | Recall | F1 | AUC
  2. PTSD subtype classification ->  Metric | Value  (Accuracy, Macro-F1, Kappa)

Both are produced from plain Python dicts (no placeholders), so they can be
generated directly from the evaluation module's outputs.
"""
from __future__ import annotations


def _fmt(v, nd: int = 3) -> str:
    """Format a float, leaving integers / strings untouched."""
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:.{nd}f}"


def emotion_table_markdown(rows: list[dict], classes: list[str] | None = None) -> str:
    """`rows` is a list of dicts with keys: class, precision, recall, f1, auc."""
    header = "| Emotion Class | Precision | Recall | F1 | AUC |\n|---|---|---|---|---|"
    lines = [header]
    order = classes or [r["class"] for r in rows]
    by_class = {r["class"]: r for r in rows}
    for c in order:
        r = by_class[c]
        lines.append(
            f"| {r['class']} | {_fmt(r['precision'])} | {_fmt(r['recall'])} "
            f"| {_fmt(r['f1'])} | {_fmt(r['auc'])} |"
        )
    return "\n".join(lines)


def emotion_table_latex(rows: list[dict], classes: list[str] | None = None,
                        caption: str = "Per-class emotion recognition performance.",
                        label: str = "tab:emotion_metrics") -> str:
    """Return a journal-style LaTeX `table` environment (booktabs)."""
    order = classes or [r["class"] for r in rows]
    by_class = {r["class"]: r for r in rows}
    body = []
    for c in order:
        r = by_class[c]
        body.append(
            f"        {r['class']} & {_fmt(r['precision'])} & {_fmt(r['recall'])} "
            f"& {_fmt(r['f1'])} & {_fmt(r['auc'])} \\\\"
        )
    return "\n".join([
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{" + caption + "}",
        "    \\label{" + label + "}",
        "    \\begin{tabular}{lcccc}",
        "        \\toprule",
        "        Emotion Class & Precision & Recall & F1 & AUC \\\\",
        "        \\midrule",
        *body,
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
    ])


def subtype_table_markdown(metrics: dict) -> str:
    """`metrics` = {accuracy, macro_f1, cohen_kappa, n_samples}."""
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {_fmt(metrics['accuracy'])} |",
        f"| Macro-F1 | {_fmt(metrics['macro_f1'])} |",
        f"| Cohen's Kappa | {_fmt(metrics['cohen_kappa'])} |",
        f"| N (samples) | {metrics.get('n_samples', '-')} |",
    ]
    return "\n".join(lines)


def subtype_table_latex(metrics: dict,
                        caption: str = "PTSD subtype classification performance.",
                        label: str = "tab:subtype_metrics") -> str:
    """Return a journal-style LaTeX table for the 3-way subtype classifier."""
    return "\n".join([
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{" + caption + "}",
        "    \\label{" + label + "}",
        "    \\begin{tabular}{lc}",
        "        \\toprule",
        "        Metric & Value \\\\",
        "        \\midrule",
        f"        Accuracy & {_fmt(metrics['accuracy'])} \\\\",
        f"        Macro-F1 & {_fmt(metrics['macro_f1'])} \\\\",
        f"        Cohen's Kappa & {_fmt(metrics['cohen_kappa'])} \\\\",
        f"        N (samples) & {metrics.get('n_samples', '-')} \\\\",
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
    ])


def save_tables(emotion_rows: list[dict], subtype_metrics: dict,
                out_dir: str, classes: list[str] | None = None) -> dict:
    """Write both tables to .md and .tex files and return the written paths."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    emo_md = emotion_table_markdown(emotion_rows, classes)
    emo_tex = emotion_table_latex(emotion_rows, classes)
    sub_md = subtype_table_markdown(subtype_metrics)
    sub_tex = subtype_table_latex(subtype_metrics)

    paths["emotion_md"] = os.path.join(out_dir, "table_emotion_metrics.md")
    paths["emotion_tex"] = os.path.join(out_dir, "table_emotion_metrics.tex")
    paths["subtype_md"] = os.path.join(out_dir, "table_subtype_metrics.md")
    paths["subtype_tex"] = os.path.join(out_dir, "table_subtype_metrics.tex")

    with open(paths["emotion_md"], "w") as f:
        f.write(emo_md + "\n")
    with open(paths["emotion_tex"], "w") as f:
        f.write(emo_tex + "\n")
    with open(paths["subtype_md"], "w") as f:
        f.write(sub_md + "\n")
    with open(paths["subtype_tex"], "w") as f:
        f.write(sub_tex + "\n")

    paths["emotion_markdown"] = emo_md
    paths["emotion_latex"] = emo_tex
    paths["subtype_markdown"] = sub_md
    paths["subtype_latex"] = sub_tex
    return paths
