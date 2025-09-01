import unittest

from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation, SystemTranslations


class RomanshWMT24EvaluationTestCase(unittest.TestCase):

    def setUp(self):
        self.evaluation = RomanshWMT24Evaluation()

    def test_evaluate(self):
        system_translations = SystemTranslations(
            sys_name="test",
            variety="rm-rumgr",
            translations_rm_to_de=["test"] * 998,
            translations_de_to_rm=["test"] * 998,
        )
        result = self.evaluation.evaluate(system_translations, "chrf")
        print(result)

    def test_evaluate_translaturia(self):
        from romansh_mt_eval.benchmarking.system_translations import load_translaturia_translations
        translations = load_translaturia_translations()
        for variety, system_translations in translations.items():
            result = self.evaluation.evaluate(system_translations, "chrf")
            print(result)

    def test_evaluate_supertext(self):
        from romansh_mt_eval.benchmarking.system_translations import load_supertext_translations
        translations = load_supertext_translations()
        for variety, system_translations in translations.items():
            result = self.evaluation.evaluate(system_translations, "chrf")
            print(result)

    def test_evaluate_madlad(self):
        from romansh_mt_eval.benchmarking.system_translations import load_madlad_translations_direct, load_madlad_translations_pivot
        translations = load_madlad_translations_direct()
        for variety, system_translations in translations.items():
            result = self.evaluation.evaluate(system_translations, "chrf")
            print(result)
        translations = load_madlad_translations_pivot()
        for variety, system_translations in translations.items():
            result = self.evaluation.evaluate(system_translations, "chrf")
            print(result)

    def test_evaluate_llm(self):
        from romansh_mt_eval.benchmarking.system_translations import load_llm_translations
        translations = load_llm_translations("Gemini-2.5-Flash")
        for variety, system_translations in translations.items():
            result = self.evaluation.evaluate(system_translations, "chrf")
            print(result)
