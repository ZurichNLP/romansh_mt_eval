from typing import Literal

from romansh_mt_eval.benchmarking.evaluation import RomanshWMT24Evaluation, SystemResult, SystemTranslations
from romansh_mt_eval.benchmarking.system_translations import (
    load_translaturia_translations,
    load_supertext_translations,
    load_madlad_translations_direct,
    load_madlad_translations_pivot,
    load_llm_translations
)


def get_all_system_translations() -> dict[str, dict[str, SystemTranslations]]:
    all_translations = {}

    madlad_direct_translations = load_madlad_translations_direct()
    all_translations["madlad400-10b-mt_direct"] = {}
    for variety, system_translations in madlad_direct_translations.items():
        all_translations["madlad400-10b-mt_direct"][variety] = system_translations

    madlad_pivot_translations = load_madlad_translations_pivot()
    all_translations["madlad400-10b-mt_pivot_en"] = {}
    for variety, system_translations in madlad_pivot_translations.items():
        all_translations["madlad400-10b-mt_pivot_en"][variety] = system_translations

    translaturia_translations = load_translaturia_translations()
    all_translations["translaturia"] = {}
    for variety, system_translations in translaturia_translations.items():
        all_translations["translaturia"][variety] = system_translations

    supertext_translations = load_supertext_translations()
    all_translations["supertext"] = {}
    for variety, system_translations in supertext_translations.items():
        all_translations["supertext"][variety] = system_translations

    llm_models = ["Llama-3.3-70b", "Gemini-2.5-Flash", "GPT-4o"]
    for llm in llm_models:
        all_translations[llm] = {}
        llm_translations = load_llm_translations(llm)
        for variety, system_translations in llm_translations.items():
            all_translations[llm][variety] = system_translations

    return all_translations


def get_all_system_results(metric: Literal["bleu", "chrf", "xcomet-xl"]) -> dict[str, dict[str, SystemResult]]:
    """
    Get evaluation results for all systems using the specified metric.

    Returns a dict of dicts: system_name -> variety -> SystemResult
    """
    evaluation = RomanshWMT24Evaluation()
    all_translations = get_all_system_translations()
    results = {}
    for system_name in all_translations.keys():
        results[system_name] = {}
        for variety, system_translations in all_translations[system_name].items():
            results[system_name][variety] = evaluation.evaluate(system_translations, metric)
    return results


if __name__ == "__main__":
    results = get_all_system_results("chrf")
    import pprint
    pprint.pprint(results)
