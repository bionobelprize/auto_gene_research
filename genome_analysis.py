#!/usr/bin/env python3
"""
genome_analysis.py
==================
Genomic data analysis and visualization pipeline.

For each genome sub-folder found under a root directory the script:
  1. Reads the raw genome FASTA file (*genomic.fna, excluding cds_from_genomic.fna)
     to calculate genome size, GC content and N50.
  2. Reads the CDS FASTA file (cds_from_genomic.fna) to count genes and
     compute length statistics.
  3. Parses the annotation GFF3 file (genomic.gff) to count genes, exons, CDS
     features and estimate intron counts.
  4. Aggregates all per-genome metrics into a summary table that is written to
     a CSV file.
  5. Generates the following figures:
       - CDS-length distribution histogram (per genome, overlaid)
       - GC-content distribution bar chart
       - Genome-size vs gene-count scatter plot
  6. Produces a self-contained HTML report embedding the figures and table.

Usage
-----
    python genome_analysis.py /path/to/root_folder [--output /path/to/output_dir]

Dependencies
------------
    biopython, pandas, matplotlib, seaborn, numpy
"""

import argparse
import base64
import glob
import io
import logging
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO

# Suppress BioPython GFF-related warnings so the output stays clean.
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Genome / FASTA helpers
# ---------------------------------------------------------------------------

def calculate_gc_content(sequence: str) -> float:
    """Return the GC percentage of *sequence* (0–100).

    Parameters
    ----------
    sequence:
        Nucleotide string (upper- or lower-case, may contain ``N``).

    Returns
    -------
    float
        GC% rounded to two decimal places, or 0.0 for an empty sequence.
    """
    if not sequence:
        return 0.0
    seq_upper = sequence.upper()
    gc = seq_upper.count("G") + seq_upper.count("C")
    return round(gc / len(seq_upper) * 100, 2)


def calculate_n50(lengths: List[int]) -> int:
    """Return the N50 value for *lengths* (list of sequence lengths).

    N50 is the length *L* such that contigs of length ≥ L cover at least
    50 % of the total assembly.

    Parameters
    ----------
    lengths:
        List of contig/scaffold lengths (integers > 0).

    Returns
    -------
    int
        N50 value, or 0 when *lengths* is empty.
    """
    if not lengths:
        return 0
    sorted_lengths = sorted(lengths, reverse=True)
    total = sum(sorted_lengths)
    cumulative = 0
    for length in sorted_lengths:
        cumulative += length
        if cumulative >= total / 2:
            return length
    return 0


def analyze_genome_fasta(fasta_path: str) -> Dict:
    """Parse a raw genome FASTA file and return basic assembly statistics.

    Parameters
    ----------
    fasta_path:
        Absolute path to the ``*genomic.fna`` file.

    Returns
    -------
    dict
        Keys: ``genome_size``, ``num_sequences``, ``gc_content``, ``n50``,
        ``longest_seq``, ``shortest_seq``.
        All values are 0 / 0.0 on error.
    """
    stats: Dict = {
        "genome_size": 0,
        "num_sequences": 0,
        "gc_content": 0.0,
        "n50": 0,
        "longest_seq": 0,
        "shortest_seq": 0,
    }
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
        if not records:
            logger.warning("No records found in genome FASTA: %s", fasta_path)
            return stats

        lengths = [len(r.seq) for r in records]
        full_seq = "".join(str(r.seq) for r in records)

        stats["genome_size"] = sum(lengths)
        stats["num_sequences"] = len(records)
        stats["gc_content"] = calculate_gc_content(full_seq)
        stats["n50"] = calculate_n50(lengths)
        stats["longest_seq"] = max(lengths)
        stats["shortest_seq"] = min(lengths)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse genome FASTA %s: %s", fasta_path, exc)
    return stats


def analyze_cds_fasta(fasta_path: str) -> Dict:
    """Parse a CDS FASTA file (``cds_from_genomic.fna``) and return statistics.

    Parameters
    ----------
    fasta_path:
        Absolute path to the CDS FASTA file.

    Returns
    -------
    dict
        Keys: ``num_genes``, ``avg_cds_length``, ``max_cds_length``,
        ``min_cds_length``, ``cds_lengths`` (list of ints).
    """
    stats: Dict = {
        "num_genes": 0,
        "avg_cds_length": 0.0,
        "max_cds_length": 0,
        "min_cds_length": 0,
        "cds_lengths": [],
    }
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
        if not records:
            logger.warning("No CDS records found in: %s", fasta_path)
            return stats

        lengths = [len(r.seq) for r in records]
        stats["num_genes"] = len(records)
        stats["avg_cds_length"] = round(float(np.mean(lengths)), 2)
        stats["max_cds_length"] = max(lengths)
        stats["min_cds_length"] = min(lengths)
        stats["cds_lengths"] = lengths
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse CDS FASTA %s: %s", fasta_path, exc)
    return stats


# ---------------------------------------------------------------------------
# GFF parser (pure-Python, no external GFF library required)
# ---------------------------------------------------------------------------

def parse_gff(gff_path: str) -> Dict:
    """Parse a GFF3 annotation file and return feature counts.

    Only non-comment, non-``##FASTA`` section lines are considered.

    Parameters
    ----------
    gff_path:
        Absolute path to the ``genomic.gff`` file.

    Returns
    -------
    dict
        Keys: ``num_genes``, ``num_exons``, ``num_cds_features``,
        ``num_introns`` (estimated as max(0, exons - genes)).
    """
    stats: Dict = {
        "num_genes": 0,
        "num_exons": 0,
        "num_cds_features": 0,
        "num_introns": 0,
    }
    try:
        in_fasta_section = False
        with open(gff_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("##FASTA"):
                    in_fasta_section = True
                    continue
                if in_fasta_section:
                    continue
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                feature_type = parts[2].lower()
                if feature_type == "gene":
                    stats["num_genes"] += 1
                elif feature_type == "exon":
                    stats["num_exons"] += 1
                elif feature_type == "cds":
                    stats["num_cds_features"] += 1

        # Estimate intron count: for a gene with N exons there are N-1 introns.
        # Approximated globally as max(0, total_exons - total_genes).
        stats["num_introns"] = max(0, stats["num_exons"] - stats["num_genes"])
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse GFF %s: %s", gff_path, exc)
    return stats


# ---------------------------------------------------------------------------
# Folder scanning
# ---------------------------------------------------------------------------

def find_genome_dirs(root: str) -> List[str]:
    """Return a sorted list of sub-folder paths that look like genome packages.

    A valid genome folder must contain **all three** of:
      - ``cds_from_genomic.fna``
      - ``genomic.gff``
      - At least one file matching ``*genomic.fna`` that is **not**
        ``cds_from_genomic.fna``.

    Parameters
    ----------
    root:
        Path to the root directory containing genome sub-folders.

    Returns
    -------
    list of str
        Absolute paths to valid genome sub-folders, sorted alphabetically.
    """
    genome_dirs: List[str] = []
    try:
        entries = sorted(os.listdir(root))
    except OSError as exc:
        logger.error("Cannot list root directory %s: %s", root, exc)
        return genome_dirs

    for entry in entries:
        subdir = os.path.join(root, entry)
        if not os.path.isdir(subdir):
            continue

        cds_file = os.path.join(subdir, "cds_from_genomic.fna")
        gff_file = os.path.join(subdir, "genomic.gff")
        raw_fna_files = [
            f for f in glob.glob(os.path.join(subdir, "*genomic.fna"))
            if os.path.basename(f) != "cds_from_genomic.fna"
        ]

        if os.path.isfile(cds_file) and os.path.isfile(gff_file) and raw_fna_files:
            genome_dirs.append(subdir)
        else:
            missing = []
            if not os.path.isfile(cds_file):
                missing.append("cds_from_genomic.fna")
            if not os.path.isfile(gff_file):
                missing.append("genomic.gff")
            if not raw_fna_files:
                missing.append("*genomic.fna (raw genome)")
            logger.warning(
                "Skipping %s – missing: %s", subdir, ", ".join(missing)
            )
    return genome_dirs


def get_raw_genome_fna(genome_dir: str) -> Optional[str]:
    """Return the path to the raw genome ``*genomic.fna`` file.

    Excludes ``cds_from_genomic.fna``.

    Parameters
    ----------
    genome_dir:
        Path to a genome sub-folder.

    Returns
    -------
    str or None
        Path to the raw genome FASTA, or ``None`` if not found.
    """
    candidates = [
        f for f in glob.glob(os.path.join(genome_dir, "*genomic.fna"))
        if os.path.basename(f) != "cds_from_genomic.fna"
    ]
    if not candidates:
        return None
    # Prefer the longest file name (most specific) when multiple matches exist.
    return max(candidates, key=lambda p: len(os.path.basename(p)))


# ---------------------------------------------------------------------------
# Per-genome analysis orchestrator
# ---------------------------------------------------------------------------

def analyze_genome(genome_dir: str) -> Dict:
    """Run the full analysis pipeline for one genome directory.

    Parameters
    ----------
    genome_dir:
        Path to a valid genome sub-folder.

    Returns
    -------
    dict
        Merged statistics from all three input files plus ``genome_id``.
    """
    genome_id = os.path.basename(genome_dir)
    logger.info("Analyzing %s …", genome_id)

    raw_fna = get_raw_genome_fna(genome_dir)
    cds_fna = os.path.join(genome_dir, "cds_from_genomic.fna")
    gff_file = os.path.join(genome_dir, "genomic.gff")

    genome_stats = analyze_genome_fasta(raw_fna) if raw_fna else {}
    cds_stats = analyze_cds_fasta(cds_fna)
    gff_stats = parse_gff(gff_file)

    return {
        "genome_id": genome_id,
        # Genome assembly metrics
        "genome_size_bp": genome_stats.get("genome_size", 0),
        "num_sequences": genome_stats.get("num_sequences", 0),
        "gc_content": genome_stats.get("gc_content", 0.0),
        "n50": genome_stats.get("n50", 0),
        "longest_seq_bp": genome_stats.get("longest_seq", 0),
        "shortest_seq_bp": genome_stats.get("shortest_seq", 0),
        # CDS metrics
        "num_cds": cds_stats.get("num_genes", 0),
        "avg_cds_length_bp": cds_stats.get("avg_cds_length", 0.0),
        "max_cds_length_bp": cds_stats.get("max_cds_length", 0),
        "min_cds_length_bp": cds_stats.get("min_cds_length", 0),
        # GFF annotation metrics
        "num_genes_gff": gff_stats.get("num_genes", 0),
        "num_exons": gff_stats.get("num_exons", 0),
        "num_cds_features": gff_stats.get("num_cds_features", 0),
        "num_introns": gff_stats.get("num_introns", 0),
        # Keep raw CDS lengths list for histogram (not written to CSV)
        "_cds_lengths": cds_stats.get("cds_lengths", []),
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: plt.Figure) -> str:
    """Encode a matplotlib figure as a base64 PNG data URI string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f"data:image/png;base64,{encoded}"


def plot_cds_length_histogram(
    records: List[Dict], output_path: str
) -> plt.Figure:
    """Draw a CDS-length distribution histogram (all genomes overlaid).

    Parameters
    ----------
    records:
        List of per-genome analysis dicts (must contain ``_cds_lengths`` and
        ``genome_id``).
    output_path:
        File path where the PNG will be saved.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False
    for rec in records:
        lengths = rec.get("_cds_lengths", [])
        if lengths:
            ax.hist(
                lengths,
                bins=50,
                alpha=0.6,
                label=rec["genome_id"],
                edgecolor="none",
            )
            has_data = True

    ax.set_xlabel("CDS Length (bp)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("CDS Length Distribution", fontsize=14)
    if has_data:
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    try:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("Saved CDS histogram → %s", output_path)
    except OSError as exc:
        logger.error("Could not save histogram: %s", exc)
    return fig


def plot_gc_content_bar(records: List[Dict], output_path: str) -> plt.Figure:
    """Draw a horizontal bar chart of GC % for each genome.

    Parameters
    ----------
    records:
        List of per-genome analysis dicts.
    output_path:
        File path where the PNG will be saved.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = pd.DataFrame(
        {
            "genome_id": [r["genome_id"] for r in records],
            "gc_content": [r["gc_content"] for r in records],
        }
    )
    df = df.sort_values("gc_content", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4 + 1)))
    colors = sns.color_palette("viridis", len(df))
    ax.barh(df["genome_id"], df["gc_content"], color=colors)
    ax.set_xlabel("GC Content (%)", fontsize=12)
    ax.set_title("GC Content per Genome", fontsize=14)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    try:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("Saved GC content chart → %s", output_path)
    except OSError as exc:
        logger.error("Could not save GC chart: %s", exc)
    return fig


def plot_genome_size_vs_genes(
    records: List[Dict], output_path: str
) -> plt.Figure:
    """Draw a scatter plot of genome size (Mbp) vs gene count from GFF.

    Parameters
    ----------
    records:
        List of per-genome analysis dicts.
    output_path:
        File path where the PNG will be saved.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = pd.DataFrame(
        {
            "genome_id": [r["genome_id"] for r in records],
            "genome_size_mbp": [r["genome_size_bp"] / 1e6 for r in records],
            "num_genes_gff": [r["num_genes_gff"] for r in records],
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df["genome_size_mbp"],
        df["num_genes_gff"],
        c=range(len(df)),
        cmap="tab10",
        s=80,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.5,
    )

    # Annotate each point with its genome ID (truncated to 20 chars).
    for _, row in df.iterrows():
        label = row["genome_id"][:20]
        ax.annotate(
            label,
            (row["genome_size_mbp"], row["num_genes_gff"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )

    ax.set_xlabel("Genome Size (Mbp)", fontsize=12)
    ax.set_ylabel("Number of Genes (GFF)", fontsize=12)
    ax.set_title("Genome Size vs Gene Count", fontsize=14)
    fig.tight_layout()
    try:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("Saved scatter plot → %s", output_path)
    except OSError as exc:
        logger.error("Could not save scatter plot: %s", exc)
    return fig


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = [
    "genome_id",
    "genome_size_bp",
    "num_sequences",
    "gc_content",
    "n50",
    "longest_seq_bp",
    "shortest_seq_bp",
    "num_cds",
    "avg_cds_length_bp",
    "max_cds_length_bp",
    "min_cds_length_bp",
    "num_genes_gff",
    "num_exons",
    "num_cds_features",
    "num_introns",
]


def save_summary_csv(records: List[Dict], output_path: str) -> pd.DataFrame:
    """Save the summary statistics to a CSV file.

    Parameters
    ----------
    records:
        List of per-genome analysis dicts.
    output_path:
        Destination CSV file path.

    Returns
    -------
    pandas.DataFrame
        The DataFrame written to disk.
    """
    df = pd.DataFrame(records)[SUMMARY_COLUMNS]
    try:
        df.to_csv(output_path, index=False)
        logger.info("Saved summary CSV → %s", output_path)
    except OSError as exc:
        logger.error("Could not write CSV %s: %s", output_path, exc)
    return df


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Genome Analysis Report</title>
  <style>
    body  {{ font-family: Arial, sans-serif; margin: 2rem; color: #222; }}
    h1   {{ color: #2c5282; }}
    h2   {{ color: #2b6cb0; border-bottom: 2px solid #bee3f8; padding-bottom: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
    th, td {{ border: 1px solid #cbd5e0; padding: 6px 10px; text-align: left; }}
    th   {{ background: #ebf8ff; }}
    tr:nth-child(even) {{ background: #f7fafc; }}
    .figure-row {{ display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center; }}
    .figure-row img {{ max-width: 100%; height: auto; border: 1px solid #e2e8f0;
                       border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,.08); }}
    footer {{ margin-top: 2rem; font-size: 0.75rem; color: #718096; }}
  </style>
</head>
<body>
  <h1>🧬 Genome Analysis Report</h1>
  <p>Generated on <strong>{date}</strong> &nbsp;|&nbsp;
     Analyzed <strong>{n_genomes}</strong> genome(s)</p>

  <h2>Summary Table</h2>
  {table_html}

  <h2>Figures</h2>
  <div class="figure-row">
    <figure>
      <figcaption><strong>CDS Length Distribution</strong></figcaption>
      <img src="{hist_src}" alt="CDS length histogram">
    </figure>
    <figure>
      <figcaption><strong>GC Content per Genome</strong></figcaption>
      <img src="{gc_src}" alt="GC content bar chart">
    </figure>
    <figure>
      <figcaption><strong>Genome Size vs Gene Count</strong></figcaption>
      <img src="{scatter_src}" alt="Genome size vs gene count scatter">
    </figure>
  </div>

  <footer>
    Produced by <em>genome_analysis.py</em>.
    Dependencies: BioPython, pandas, matplotlib, seaborn.
  </footer>
</body>
</html>
"""


def generate_html_report(
    df: pd.DataFrame,
    hist_fig: plt.Figure,
    gc_fig: plt.Figure,
    scatter_fig: plt.Figure,
    output_path: str,
) -> None:
    """Write a self-contained HTML report.

    All figures are embedded as base64 data URIs so the report is a single
    portable file.

    Parameters
    ----------
    df:
        Summary DataFrame produced by :func:`save_summary_csv`.
    hist_fig:
        CDS-length histogram figure.
    gc_fig:
        GC-content bar chart figure.
    scatter_fig:
        Genome-size vs gene-count scatter figure.
    output_path:
        Destination HTML file path.
    """
    from datetime import datetime

    table_html = df.to_html(index=False, border=0, classes="summary-table")

    html = _HTML_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_genomes=len(df),
        table_html=table_html,
        hist_src=_fig_to_base64(hist_fig),
        gc_src=_fig_to_base64(gc_fig),
        scatter_src=_fig_to_base64(scatter_fig),
    )
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("Saved HTML report → %s", output_path)
    except OSError as exc:
        logger.error("Could not write HTML report %s: %s", output_path, exc)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(root_dir: str, output_dir: str) -> None:
    """Execute the full analysis pipeline.

    Parameters
    ----------
    root_dir:
        Root directory containing genome sub-folders.
    output_dir:
        Directory where output files (CSV, PNGs, HTML) are written.
    """
    root_dir = os.path.abspath(root_dir)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(root_dir):
        logger.error("Root directory not found: %s", root_dir)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Root dir  : %s", root_dir)
    logger.info("Output dir: %s", output_dir)

    # 1. Find genome folders
    genome_dirs = find_genome_dirs(root_dir)
    if not genome_dirs:
        logger.error(
            "No valid genome directories found under %s. "
            "Each sub-folder must contain cds_from_genomic.fna, "
            "genomic.gff, and at least one *genomic.fna file.",
            root_dir,
        )
        sys.exit(1)
    logger.info("Found %d genome folder(s).", len(genome_dirs))

    # 2. Analyze each genome
    records: List[Dict] = []
    for gdir in genome_dirs:
        try:
            records.append(analyze_genome(gdir))
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error analysing %s: %s", gdir, exc)

    if not records:
        logger.error("No genome could be analyzed successfully.")
        sys.exit(1)

    # 3. Save CSV
    csv_path = os.path.join(output_dir, "genome_summary.csv")
    df = save_summary_csv(records, csv_path)

    # 4. Plots
    hist_path = os.path.join(output_dir, "cds_length_histogram.png")
    gc_path = os.path.join(output_dir, "gc_content_bar.png")
    scatter_path = os.path.join(output_dir, "genome_size_vs_genes.png")

    hist_fig = plot_cds_length_histogram(records, hist_path)
    gc_fig = plot_gc_content_bar(records, gc_path)
    scatter_fig = plot_genome_size_vs_genes(records, scatter_path)

    # 5. HTML report
    html_path = os.path.join(output_dir, "report.html")
    generate_html_report(df, hist_fig, gc_fig, scatter_fig, html_path)

    # Close all figures to free memory
    plt.close("all")

    logger.info("Pipeline complete. Outputs in: %s", output_dir)
    logger.info("  %-30s %s", "Summary CSV:", csv_path)
    logger.info("  %-30s %s", "CDS histogram:", hist_path)
    logger.info("  %-30s %s", "GC bar chart:", gc_path)
    logger.info("  %-30s %s", "Scatter plot:", scatter_path)
    logger.info("  %-30s %s", "HTML report:", html_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Genomic data analysis and visualization pipeline.\n\n"
            "Scans a root directory for genome sub-folders (each must contain "
            "cds_from_genomic.fna, genomic.gff, and a *genomic.fna file), "
            "computes assembly and annotation statistics, and writes a CSV "
            "summary, PNG figures, and an HTML report to the output directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root_dir",
        help="Root directory containing genome sub-folders.",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default=None,
        help=(
            "Directory where outputs are written. "
            "Defaults to <root_dir>/analysis_output."
        ),
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.abspath(args.root_dir), "analysis_output"
    )
    run_pipeline(args.root_dir, output_dir)


if __name__ == "__main__":
    main()
