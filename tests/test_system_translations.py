from unittest import TestCase

from romansh_mt_eval.benchmarking.constants import VARIETIES
from romansh_mt_eval.benchmarking.system_translations import load_translaturia_translations
from romansh_mt_eval.benchmarking.system_translations import load_supertext_translations
from romansh_mt_eval.benchmarking.system_translations import load_madlad_translations_direct
from romansh_mt_eval.benchmarking.system_translations import load_madlad_translations_pivot
from romansh_mt_eval.benchmarking.system_translations import load_llm_translations


class SystemTranslationsTestCase(TestCase):

    def test_load_madlad_translations_direct(self):
        translations = load_madlad_translations_direct()

        # Check that all varieties are present
        for variety in VARIETIES:
            self.assertIn(variety, translations)

            # Check system name and variety
            self.assertEqual("madlad400-10b-mt_direct", translations[variety].sys_name)
            self.assertEqual(variety, translations[variety].variety)

            # Check that translations exist in both directions
            self.assertIsNotNone(translations[variety].translations_rm_to_de)
            self.assertIsNotNone(translations[variety].translations_de_to_rm)
            self.assertEqual(998, len(translations[variety].translations_rm_to_de))
            self.assertEqual(998, len(translations[variety].translations_de_to_rm))

        # Check that all varieties have the same DE->RM translations (copied from Rumantsch Grischun)
        for variety in VARIETIES:
            self.assertEqual(
                translations["rm-rumgr"].translations_de_to_rm,
                translations[variety].translations_de_to_rm
            )

    def test_load_madlad_translations_pivot(self):
        translations = load_madlad_translations_pivot()

        # Check that all varieties are present
        for variety in VARIETIES:
            self.assertIn(variety, translations)

            # Check system name and variety
            self.assertEqual("madlad400-10b-mt_pivot_en", translations[variety].sys_name)
            self.assertEqual(variety, translations[variety].variety)

            # Check that translations exist in both directions
            self.assertIsNotNone(translations[variety].translations_rm_to_de)
            self.assertIsNotNone(translations[variety].translations_de_to_rm)
            self.assertEqual(998, len(translations[variety].translations_rm_to_de))
            self.assertEqual(998, len(translations[variety].translations_de_to_rm))

        # Check that all varieties have the same DE->RM translations (copied from Rumantsch Grischun)
        for variety in VARIETIES:
            self.assertEqual(
                translations["rm-rumgr"].translations_de_to_rm,
                translations[variety].translations_de_to_rm
            )

    def test_load_translaturia_translations(self):
        translations = load_translaturia_translations()
        self.assertIn("rm-rumgr", translations)
        self.assertEqual(998, len(translations["rm-rumgr"].translations_de_to_rm))
        self.assertEqual("translaturia", translations["rm-rumgr"].sys_name)
        self.assertEqual("rm-rumgr", translations["rm-rumgr"].variety)

    def test_load_supertext_translations(self):
        translations = load_supertext_translations()
        
        # Check that all varieties are present
        for variety in VARIETIES:
            self.assertIn(variety, translations)
            
            # Check system name and variety
            self.assertEqual("supertext", translations[variety].sys_name)
            self.assertEqual(variety, translations[variety].variety)
            
            # Check that translations exist in both directions
            self.assertIsNotNone(translations[variety].translations_rm_to_de)
            self.assertIsNotNone(translations[variety].translations_de_to_rm)
            self.assertEqual(998 - 38, len(translations[variety].translations_rm_to_de))  # 998 total - 38 bad sources
            self.assertEqual(998 - 38, len(translations[variety].translations_de_to_rm))
            
            # Check that skips_bad_sources is True
            self.assertTrue(translations[variety].skips_bad_sources)
            
        # Check that all varieties have the same DE->RM translations (copied from Rumantsch Grischun)
        for variety in VARIETIES:
            self.assertEqual(
                translations["rm-rumgr"].translations_de_to_rm,
                translations[variety].translations_de_to_rm
            )

    def test_load_llm_translations(self):
        for llm in ["Llama-3.3-70b", "Gemini-2.5-Flash", "GPT-4o"]:
            translations = load_llm_translations(llm)
            for variety in VARIETIES:
                self.assertIn(variety, translations)
                self.assertEqual(llm, translations[variety].sys_name)
                self.assertEqual(variety, translations[variety].variety)
                self.assertEqual(998, len(translations[variety].translations_rm_to_de))
                self.assertEqual(998, len(translations[variety].translations_de_to_rm))
