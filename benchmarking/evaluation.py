import copy
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from datasets import load_dataset
from sacrebleu import CHRF, BLEU

from romansh_mt_eval.benchmarking.constants import VARIETIES, DOMAINS
from romansh_mt_eval.benchmarking.comet_client import Comet



@dataclass
class SystemTranslations:
    sys_name: Union[Literal["ref"], str]
    variety: Literal["rm-rumgr", "rm-sursilv", "rm-sutsilv", "rm-surmiran", "rm-puter", "rm-vallader"]
    translations_rm_to_de: list[str]
    translations_de_to_rm: list[str]
    skips_bad_sources: bool = False  # Set to true if the translations do not include segments with is_bad_source=True


@dataclass
class SystemLanguagePairResult:
    lang_pair: Literal["rm-de", "de-rm"]
    sys_name: str
    variety: Literal["rm-rumgr", "rm-sursilv", "rm-sutsilv", "rm-surmiran", "rm-puter", "rm-vallader"]
    metric: Literal["bleu", "chrf", "xcomet-xl"]
    domain_results: dict[str, float]
    micro_avg: float

    @property
    def macro_avg(self) -> float:
        return np.mean([self.domain_results[domain] for domain in DOMAINS if self.domain_results[domain] is not None]).item()

@dataclass
class SystemResult:
    sys_name: str
    variety: Literal["rm-rumgr", "rm-sursilv", "rm-sutsilv", "rm-surmiran", "rm-puter", "rm-vallader"]
    # sys_translations: SystemTranslations
    metric: Literal["bleu", "chrf", "xcomet-xl"]
    scores_rm_to_de: SystemLanguagePairResult
    scores_de_to_rm: SystemLanguagePairResult


class RomanshWMT24Evaluation:

    dataset_name = "ZurichNLP/wmt24pp-rm"

    def __init__(self):
        self.dataset = {
            variety: load_dataset(self.dataset_name, f"de_DE-{variety}")
            for variety in VARIETIES.keys()
        }
        # Postprocess the references (remove newlines, normalize quotes)
        for variety in VARIETIES.keys():
            self.dataset[variety]["test"] = self.dataset[variety]["test"].map(
                lambda x: {
                    "source": self.postprocess(x["source"]),
                    "target": self.postprocess(x["target"]),
                }
            )

    def evaluate(self,
                 system_translations: SystemTranslations,
                 metric: Literal["bleu", "chrf", "xcomet-xl"],
                 ) -> SystemResult:
        assert system_translations.variety in VARIETIES.keys(), f"Variety {system_translations.variety} not recognized."
        assert len(system_translations.translations_rm_to_de) == len(system_translations.translations_de_to_rm)
        if system_translations.skips_bad_sources:
            assert len(system_translations.translations_rm_to_de) == len(self.dataset[system_translations.variety]["test"].filter(lambda x: not x["is_bad_source"]))
        else:
            assert len(system_translations.translations_rm_to_de) == len(self.dataset[system_translations.variety]["test"])

        # Postprocess the system translations (remove newlines, normalize quotes)
        system_translations.translations_rm_to_de = [self.postprocess(t) for t in system_translations.translations_rm_to_de]
        system_translations.translations_de_to_rm = [self.postprocess(t) for t in system_translations.translations_de_to_rm]

        if metric == "chrf":
            sacrebleu_metric = CHRF()
        elif metric == "bleu":
            sacrebleu_metric = BLEU()
        elif metric == "xcomet-xl":
            comet_metric = Comet()
        else:
            raise ValueError(f"Metric {metric} not recognized. Choose from 'bleu', 'chrf', 'xcomet-xl'.")

        dataset = copy.deepcopy(self.dataset[system_translations.variety])
        # Add the system translations as a new column to the dataset
        if system_translations.skips_bad_sources:
            # First filter out bad sources, then add columns
            dataset["test"] = dataset["test"].filter(lambda x: not x["is_bad_source"])
            dataset["test"] = dataset["test"].add_column("sys_rm_to_de", system_translations.translations_rm_to_de)
            dataset["test"] = dataset["test"].add_column("sys_de_to_rm", system_translations.translations_de_to_rm)
        else:
            # First add columns, then filter
            dataset["test"] = dataset["test"].add_column("sys_rm_to_de", system_translations.translations_rm_to_de)
            dataset["test"] = dataset["test"].add_column("sys_de_to_rm", system_translations.translations_de_to_rm)
            dataset["test"] = dataset["test"].filter(lambda x: not x["is_bad_source"])

        system_result = SystemResult(
            sys_name=system_translations.sys_name,
            variety=system_translations.variety,
            # sys_translations=system_translations,
            metric=metric,
            scores_rm_to_de=SystemLanguagePairResult(
                lang_pair="rm-de",
                sys_name=system_translations.sys_name,
                variety=system_translations.variety,
                metric=metric,
                domain_results={}, # will be filled in below
                micro_avg=0.0,  # will be filled in below
            ),
            scores_de_to_rm=SystemLanguagePairResult(
                lang_pair="de-rm",
                sys_name=system_translations.sys_name,
                variety=system_translations.variety,
                metric=metric,
                domain_results={}, # will be filled in below
                micro_avg=0.0, # will be filled in below
            ),
        )
        for lang_pair in ["rm-de", "de-rm"]:
            for domain in DOMAINS:
                subset = dataset["test"].filter(lambda x: x["domain"] == domain and not x["is_bad_source"])
                if lang_pair == "rm-de":
                    sys_translations = subset["sys_rm_to_de"]
                    references = subset["source"]
                elif lang_pair == "de-rm":
                    sys_translations = subset["sys_de_to_rm"]
                    references = subset["target"]

                if metric in ["bleu", "chrf"]:
                    score = sacrebleu_metric.corpus_score(sys_translations, [references]).score
                elif metric == "xcomet-xl":
                    if lang_pair == "rm-de":
                        score = 100 * comet_metric.corpus_score([None] * len(subset), sys_translations, references)
                    elif lang_pair == "de-rm":
                        score = None  # COMET does not support DE->RM

                if lang_pair == "rm-de":
                    system_result.scores_rm_to_de.domain_results[domain] = score
                elif lang_pair == "de-rm":
                    system_result.scores_de_to_rm.domain_results[domain] = score

            # All domains together (micro average)
            subset = dataset["test"].filter(lambda x: not x["is_bad_source"])
            if lang_pair == "rm-de":
                sys_translations = subset["sys_rm_to_de"]
                references = subset["source"]
            elif lang_pair == "de-rm":
                sys_translations = subset["sys_de_to_rm"]
                references = subset["target"]

            if metric in ["bleu", "chrf"]:
                score = sacrebleu_metric.corpus_score(sys_translations, [references]).score
            elif metric == "xcomet-xl":
                if lang_pair == "rm-de":
                    score = 100 * comet_metric.corpus_score([None] * len(subset), sys_translations, references)
                elif lang_pair == "de-rm":
                    score = None  # COMET does not support DE->RM

            if lang_pair == "rm-de":
                system_result.scores_rm_to_de.micro_avg = score
            elif lang_pair == "de-rm":
                system_result.scores_de_to_rm.micro_avg = score

        return system_result

    @staticmethod
    def postprocess(s: str) -> str:
        """
        Postprocess translations and references by normalizing the quotes and removing newlines.
        """
        return (s.replace("“", '"').replace("”", '"').replace("«", '"')
                .replace("»", '"').replace("\n", " ").strip())
