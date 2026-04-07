"""
tests/test_genome_analysis.py
==============================
Unit tests for genome_analysis.py.

Tests are deliberately lightweight – they use small synthetic data so no real
NCBI files are needed.  A temporary directory tree that mirrors the expected
folder structure is created for integration-style tests.
"""

import os
import sys
import textwrap
import tempfile
import unittest

# Make sure the parent package is importable when running from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genome_analysis as ga


# ---------------------------------------------------------------------------
# Helper: build a minimal but realistic fake genome directory tree
# ---------------------------------------------------------------------------

FAKE_GENOME_FNA = textwrap.dedent("""\
    >seq1 fake chromosome 1
    ATGCATGCATGCGGGGCCCCATGCATGCATGC
    >seq2 fake chromosome 2
    ATGCATGCATGCATGCATGCATGCATGCATGC
""")

FAKE_CDS_FNA = textwrap.dedent("""\
    >lcl|NC_000001.1_cds_1 gene=gene1
    ATGAAATTTGGG
    >lcl|NC_000001.1_cds_2 gene=gene2
    ATGCCCAAATTTGGGCCC
    >lcl|NC_000001.1_cds_3 gene=gene3
    ATG
""")

FAKE_GFF = textwrap.dedent("""\
    ##gff-version 3
    ##sequence-region seq1 1 100
    seq1\tRefSeq\tgene\t1\t50\t.\t+\t.\tID=gene1
    seq1\tRefSeq\texon\t1\t30\t.\t+\t.\tParent=gene1
    seq1\tRefSeq\texon\t35\t50\t.\t+\t.\tParent=gene1
    seq1\tRefSeq\tCDS\t1\t50\t.\t+\t0\tParent=gene1
    seq1\tRefSeq\tgene\t60\t90\t.\t+\t.\tID=gene2
    seq1\tRefSeq\texon\t60\t90\t.\t+\t.\tParent=gene2
    seq1\tRefSeq\tCDS\t60\t90\t.\t+\t0\tParent=gene2
""")


def _make_genome_dir(root: str, genome_id: str = "GCF_FAKE_001") -> str:
    """Create a minimal valid genome sub-folder under *root*."""
    gdir = os.path.join(root, genome_id)
    os.makedirs(gdir, exist_ok=True)

    with open(os.path.join(gdir, "cds_from_genomic.fna"), "w") as fh:
        fh.write(FAKE_CDS_FNA)
    with open(os.path.join(gdir, "genomic.gff"), "w") as fh:
        fh.write(FAKE_GFF)
    with open(
        os.path.join(gdir, f"{genome_id}_ASM1v1_genomic.fna"), "w"
    ) as fh:
        fh.write(FAKE_GENOME_FNA)

    return gdir


# ---------------------------------------------------------------------------
# Unit tests for individual helper functions
# ---------------------------------------------------------------------------

class TestGcContent(unittest.TestCase):
    def test_all_gc(self):
        self.assertEqual(ga.calculate_gc_content("GCGCGC"), 100.0)

    def test_all_at(self):
        self.assertEqual(ga.calculate_gc_content("ATATAT"), 0.0)

    def test_mixed(self):
        # ATGC → 2 GC / 4 = 50%
        self.assertEqual(ga.calculate_gc_content("ATGC"), 50.0)

    def test_lowercase(self):
        self.assertEqual(ga.calculate_gc_content("atgc"), 50.0)

    def test_empty(self):
        self.assertEqual(ga.calculate_gc_content(""), 0.0)

    def test_with_n(self):
        # ATGCN → 2 GC / 5 = 40%
        self.assertEqual(ga.calculate_gc_content("ATGCN"), 40.0)


class TestN50(unittest.TestCase):
    def test_simple(self):
        # Lengths [100, 50, 30, 10] → total 190, half = 95 → N50 = 100
        self.assertEqual(ga.calculate_n50([100, 50, 30, 10]), 100)

    def test_single(self):
        self.assertEqual(ga.calculate_n50([42]), 42)

    def test_empty(self):
        self.assertEqual(ga.calculate_n50([]), 0)

    def test_equal_lengths(self):
        # 4 × 10 = 40, half = 20, cumulative hits 20 after 2 items → N50 = 10
        self.assertEqual(ga.calculate_n50([10, 10, 10, 10]), 10)


class TestAnalyzeGenomeFasta(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fasta_path = os.path.join(self.tmpdir, "test_genomic.fna")
        with open(self.fasta_path, "w") as fh:
            fh.write(FAKE_GENOME_FNA)

    def test_basic_stats(self):
        stats = ga.analyze_genome_fasta(self.fasta_path)
        self.assertEqual(stats["num_sequences"], 2)
        self.assertGreater(stats["genome_size"], 0)
        self.assertGreater(stats["gc_content"], 0)
        self.assertGreater(stats["n50"], 0)

    def test_missing_file(self):
        stats = ga.analyze_genome_fasta("/nonexistent/path.fna")
        self.assertEqual(stats["genome_size"], 0)


class TestAnalyzeCdsFasta(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cds_path = os.path.join(self.tmpdir, "cds_from_genomic.fna")
        with open(self.cds_path, "w") as fh:
            fh.write(FAKE_CDS_FNA)

    def test_gene_count(self):
        stats = ga.analyze_cds_fasta(self.cds_path)
        self.assertEqual(stats["num_genes"], 3)

    def test_lengths(self):
        stats = ga.analyze_cds_fasta(self.cds_path)
        # Lengths: 12, 18, 3
        self.assertEqual(stats["max_cds_length"], 18)
        self.assertEqual(stats["min_cds_length"], 3)
        self.assertAlmostEqual(stats["avg_cds_length"], (12 + 18 + 3) / 3, places=1)

    def test_missing_file(self):
        stats = ga.analyze_cds_fasta("/nonexistent/cds.fna")
        self.assertEqual(stats["num_genes"], 0)


class TestParseGff(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.gff_path = os.path.join(self.tmpdir, "genomic.gff")
        with open(self.gff_path, "w") as fh:
            fh.write(FAKE_GFF)

    def test_counts(self):
        stats = ga.parse_gff(self.gff_path)
        self.assertEqual(stats["num_genes"], 2)
        self.assertEqual(stats["num_exons"], 3)
        self.assertEqual(stats["num_cds_features"], 2)
        # introns = max(0, 3 exons - 2 genes) = 1
        self.assertEqual(stats["num_introns"], 1)

    def test_missing_file(self):
        stats = ga.parse_gff("/nonexistent/genomic.gff")
        self.assertEqual(stats["num_genes"], 0)

    def test_fasta_section_ignored(self):
        gff_with_fasta = FAKE_GFF + "##FASTA\n>seq1\nATGC\n"
        gff_path2 = os.path.join(self.tmpdir, "genomic_with_fasta.gff")
        with open(gff_path2, "w") as fh:
            fh.write(gff_with_fasta)
        stats = ga.parse_gff(gff_path2)
        # Counts should be same as without the FASTA section
        self.assertEqual(stats["num_genes"], 2)


# ---------------------------------------------------------------------------
# Integration tests using the fake directory tree
# ---------------------------------------------------------------------------

class TestFindGenomeDirs(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_finds_valid_dir(self):
        _make_genome_dir(self.root, "GCF_VALID_001")
        dirs = ga.find_genome_dirs(self.root)
        self.assertEqual(len(dirs), 1)
        self.assertTrue(dirs[0].endswith("GCF_VALID_001"))

    def test_skips_incomplete_dir(self):
        # Create a folder missing the CDS file
        bad = os.path.join(self.root, "GCF_BAD_001")
        os.makedirs(bad)
        with open(os.path.join(bad, "genomic.gff"), "w") as fh:
            fh.write(FAKE_GFF)
        with open(os.path.join(bad, "GCF_BAD_001_genomic.fna"), "w") as fh:
            fh.write(FAKE_GENOME_FNA)
        # No cds_from_genomic.fna → should be skipped
        dirs = ga.find_genome_dirs(self.root)
        self.assertEqual(dirs, [])

    def test_multiple_dirs(self):
        for i in range(3):
            _make_genome_dir(self.root, f"GCF_MULTI_{i:03d}")
        dirs = ga.find_genome_dirs(self.root)
        self.assertEqual(len(dirs), 3)

    def test_nonexistent_root(self):
        dirs = ga.find_genome_dirs("/nonexistent/root")
        self.assertEqual(dirs, [])


class TestAnalyzeGenome(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.gdir = _make_genome_dir(self.root, "GCF_TEST_001")

    def test_returns_dict_with_required_keys(self):
        result = ga.analyze_genome(self.gdir)
        for key in (
            "genome_id",
            "genome_size_bp",
            "gc_content",
            "n50",
            "num_cds",
            "num_genes_gff",
            "num_exons",
        ):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_genome_id_matches_folder(self):
        result = ga.analyze_genome(self.gdir)
        self.assertEqual(result["genome_id"], "GCF_TEST_001")

    def test_cds_count(self):
        result = ga.analyze_genome(self.gdir)
        self.assertEqual(result["num_cds"], 3)

    def test_gene_count_from_gff(self):
        result = ga.analyze_genome(self.gdir)
        self.assertEqual(result["num_genes_gff"], 2)


class TestPipelineOutput(unittest.TestCase):
    """End-to-end test: run_pipeline produces expected output files."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.out_dir = tempfile.mkdtemp()
        _make_genome_dir(self.root, "GCF_E2E_001")
        _make_genome_dir(self.root, "GCF_E2E_002")

    def test_output_files_created(self):
        ga.run_pipeline(self.root, self.out_dir)
        expected = [
            "genome_summary.csv",
            "cds_length_histogram.png",
            "gc_content_bar.png",
            "genome_size_vs_genes.png",
            "report.html",
        ]
        for fname in expected:
            path = os.path.join(self.out_dir, fname)
            self.assertTrue(os.path.isfile(path), f"Missing output file: {fname}")

    def test_csv_has_correct_columns(self):
        ga.run_pipeline(self.root, self.out_dir)
        import pandas as pd
        df = pd.read_csv(os.path.join(self.out_dir, "genome_summary.csv"))
        for col in ga.SUMMARY_COLUMNS:
            self.assertIn(col, df.columns, f"CSV missing column: {col}")

    def test_csv_row_count(self):
        ga.run_pipeline(self.root, self.out_dir)
        import pandas as pd
        df = pd.read_csv(os.path.join(self.out_dir, "genome_summary.csv"))
        self.assertEqual(len(df), 2)

    def test_html_contains_table(self):
        ga.run_pipeline(self.root, self.out_dir)
        with open(os.path.join(self.out_dir, "report.html"), encoding="utf-8") as fh:
            html = fh.read()
        self.assertIn("<table", html)
        self.assertIn("data:image/png;base64,", html)


if __name__ == "__main__":
    unittest.main()
