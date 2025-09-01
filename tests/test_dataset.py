from pathlib import Path
import shutil
import unittest
from collections import Counter

from datasets import load_dataset
import jsonlines

from romansh_mt_eval.benchmarking.constants import VARIETIES


class RomanshWMT24DatasetTestCase(unittest.TestCase):
    """
    Test that the dataset uploaded to Hugging Face can be loaded and has the expected properties.
    """

    def setUp(self):
        self.dataset = {variety: load_dataset("ZurichNLP/wmt24pp-rm", f"de_DE-{variety}") for variety in VARIETIES.keys()}

    def test_dataset_subsets(self):
        for variety in VARIETIES.keys():
            self.assertIn(variety, self.dataset.keys())
            self.assertEqual(len(self.dataset[variety]["test"]), 998)

    def test_bad_source_counts(self):
        for variety in VARIETIES.keys():
            bad_source_count = sum(1 for item in self.dataset[variety]["test"] if item.get("is_bad_source", False))
            self.assertEqual(bad_source_count, 38, f"Expected 38 rows with is_bad_source=True for {variety}, but found {bad_source_count}")

    def test_domain_counts(self):
        for variety in VARIETIES.keys():
            domain_counts = Counter()
            for item in self.dataset[variety]["test"]:
                domain = item.get("domain")
                if domain:
                    domain_counts[domain] += 1
            self.assertEqual(domain_counts["literary"], 206, f"Expected 206 items with domain=literary for {variety}, but found {domain_counts['literary']}")
            self.assertEqual(domain_counts["news"], 149, f"Expected 149 items with domain=news for {variety}, but found {domain_counts['news']}")
            self.assertEqual(domain_counts["social"], 531, f"Expected 531 items with domain=social for {variety}, but found {domain_counts['social']}")
            self.assertEqual(domain_counts["speech"], 111, f"Expected 111 items with domain=speech for {variety}, but found {domain_counts['speech']}")

    def test_no_empty_targets(self):
        for variety in VARIETIES.keys():
            for item in self.dataset[variety]["test"]:
                target = item.get("target")
                self.assertIsNotNone(target, f"Target is None for an item in {variety}")
                self.assertNotEqual(target, "", f"Target is empty for an item in {variety}")


class BuildDatasetTestCase(unittest.TestCase):
    """
    Test that building the dataset from the Excel files produces the same dataset as the one uploaded to Hugging Face.
    """

    @classmethod
    def setUpClass(cls):
        cls.out_dir = Path(__file__).parent.parent / "creation_workflow" / "dataset"
        if cls.out_dir.exists():
            # Move temporarily
            cls.out_dir.rename(cls.out_dir.with_suffix(".bak"))

    @classmethod
    def tearDownClass(cls):
        if cls.out_dir.is_dir():
            shutil.rmtree(cls.out_dir)
        if cls.out_dir.with_suffix(".bak").exists():
            cls.out_dir.rename(cls.out_dir)

    def setUp(self):
        self.hf_dataset = {variety: load_dataset("ZurichNLP/wmt24pp-rm", f"de_DE-{variety}") for variety in VARIETIES.keys()}

    def test_build_dataset(self):
        from romansh_mt_eval.creation_workflow.build_dataset import main

        main()
        self.assertTrue(self.out_dir.exists())
        self.assertTrue((self.out_dir / "en-de_DE.jsonl").exists())
        for variety in VARIETIES.keys():
            self.assertTrue((self.out_dir / f"de_DE-{variety}.jsonl").exists())
            with jsonlines.open(self.out_dir / f"de_DE-{variety}.jsonl") as f:
                for line, row in zip(f, self.hf_dataset[variety]["test"]):
                    self.assertEqual(line, dict(row))
