import os
import spacy
import json
import torch
import gc
import argparse

from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm

from .madlad_translator import MadladTranslator
from .pipeline import Pipeline

MODEL_NAMES = [
    "google/madlad400-3b-mt",
    "google/madlad400-7b-mt",
    "google/madlad400-7b-mt-bt",
    "google/madlad400-10b-mt",
]
BEAM_SIZE = 5
LANG_DE = "de"
LANG_RM = "rm"
LANG_EN = "en"
LANG_IT = "it"
SPACY_SENTENCE_MODELS = {
    "de": "de_dep_news_trf",
    "en": "en_core_web_trf",
    "it": "it_core_news_lg",
}


def load_data(translation_dataset, config_name):
    if translation_dataset.endswith(".jsonl"):
        from datasets import Dataset
        import pandas as pd
        ds = pd.read_json(translation_dataset, lines=True)
        return Dataset.from_pandas(ds)
    else:
        ds = load_dataset(translation_dataset, config_name, trust_remote_code=True)
        return ds["train"]


def store_dataset(ds_elem, save_dir, variety_name, split):
    os.makedirs(save_dir, exist_ok=True)
    variety_save_path = os.path.join(save_dir, variety_name)
    os.makedirs(variety_save_path, exist_ok=True)
    split_save_path = os.path.join(variety_save_path, f"{split}.jsonl")
    with open(split_save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ds_elem) + "\n")


def make_elem(old_elem, source, target, lp, translation_time):
    new = old_elem.copy()
    new.pop("original_target", None)
    new["source"] = source
    new["target"] = target
    new["lp"] = lp
    new["translation_time"] = translation_time
    return new


def translate_dataset(
    dataset,
    translator: MadladTranslator,
    nlp_models: dict[str, spacy.language.Language],
    pipelines: list[Pipeline],
    model_name: str,
    source_column: str,
):
    save_dir = f"madlad_v1/{model_name}_translations_v1"
    os.makedirs(save_dir, exist_ok=True)
    for idx, elem in tqdm(list(enumerate(dataset, 1))):
        segmented: dict[str, list[str]] = {}
        for src_lang in {p.steps[0][0] for p in pipelines}:
            text_in = elem[source_column]
            if src_lang in nlp_models.keys():
                doc = nlp_models[src_lang](text_in)
            else: # default to Italian - e.g. for Romansh
                doc = nlp_models["it"](text_in)
        segmented[src_lang] = [s.text.strip() for s in doc.sents]
        ts = datetime.now().isoformat()
        print(f"Index {idx}: Translating pipelines {[p.name for p in pipelines]}")
        cache: dict[tuple[str,str], str] = {}
        for p in pipelines:
            # Prepare cache and initial texts
            src_lang = p.steps[0][0]
            prev_texts = segmented[src_lang]
            record_src_texts = []
            # If record_src_step == 0, record originals
            if p.record_src_step == 0:
                record_src_texts = prev_texts.copy()
            # Apply each step in batch
            for step_idx, (_, tgt_lang) in enumerate(p.steps):
                # Identify which inputs need fresh translation
                keys = [(txt, tgt_lang) for txt in prev_texts]
                new_inputs = []
                for txt, lang in keys:
                    if (txt, lang) not in cache:
                        new_inputs.append(txt)
                # Batch translate new inputs
                if new_inputs:
                    BATCH_SIZE = 5  # Adjust batch size as needed
                    for i in range(0, len(new_inputs), BATCH_SIZE):
                        batch = new_inputs[i:i+BATCH_SIZE]
                        batch_outputs = translator.translate_batch(batch, tgt_lang, num_beams=BEAM_SIZE)
                        for inp, out in zip(batch, batch_outputs):
                            cache[(inp, tgt_lang)] = out
                # Gather outputs in order
                step_outputs = [cache[(txt, tgt_lang)] for txt in prev_texts]
                # If this step matches record_src_step, capture sources
                if step_idx == p.record_src_step - 1:
                    record_src_texts = step_outputs.copy()
                prev_texts = step_outputs
            # Final targets
            final_tgts = prev_texts
            # Combine and store using helper functions
            full_src = " ".join(record_src_texts)
            full_tgt = " ".join(final_tgts)
            new_elem = make_elem(elem, full_src, full_tgt, p.name, ts)
            store_dataset(new_elem, save_dir, p.name, "train")
        # Memory cleanup
        if idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    print(f"Saved batched translations to {save_dir}")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:False"
    )
    parser = argparse.ArgumentParser(
        description="Translate dataset in one run: specify one or more src:tgt or src:pivot:tgt specs"
    )
    parser.add_argument(
        "-T", "--trans",
        action="append",
        required=True,
        help="Translation specification in the format src:tgt or src:pivot:tgt"
    )
    parser.add_argument(
        "--dataset",
        default="google/wmt24pp",
        help="Huggingface dataset name (default: google/wmt24pp)"
    )
    parser.add_argument(
        "--config",
        default="en-de_DE",
        help="Dataset configuration name (default: en-de_DE)"
    )
    parser.add_argument(
        "--source_column",
        default="target",
        help="The column in the dataset that contains the source text to be translated."
    )
    parser.add_argument(
        "--model",
        default="google/madlad400-3b-mt",
        help="The name of the model to use for translation (default: google/madlad400-3b-mt)"
    )
    args = parser.parse_args()
    # parse specs → list of (src, tgt, pivot)
    specs: list[tuple[str,str,str|None]] = []
    for spec in args.trans:
        parts = spec.split(":")
        if len(parts) == 2:
            full_src, tgt = parts
            pivot = None
        elif len(parts) == 3:
            full_src, pivot, tgt = parts
        else:
            parser.error(f"invalid --trans spec: {spec}")
        model_src = "rm" if full_src.startswith("rm-") else full_src
        specs.append((full_src, model_src, tgt, pivot))
    # --- Load Spacy Model ---
    nlp_models = {
        "en": spacy.load(SPACY_SENTENCE_MODELS["en"]),
        "de": spacy.load(SPACY_SENTENCE_MODELS["de"]),
        "it": spacy.load(SPACY_SENTENCE_MODELS["it"]),
    }
    # --- Load Data ---
    dataset = load_data(args.dataset, args.config)
    FULL_MODEL_NAME = args.model
    print(f"\n Translating using model: {FULL_MODEL_NAME}")
    # --- Configuration ---
    MODEL_NAME = FULL_MODEL_NAME.split("/", 1)[-1]
    pipelines = []
    for variety_src, model_src, tgt, pivot in specs:
        pipelines.extend(Pipeline.build_pipeline(MODEL_NAME, model_src, tgt, pivot, src_variety_code=variety_src, variety_src=variety_src))
    # --- Load Model ---
    translator = MadladTranslator(FULL_MODEL_NAME)
    # --- Process Data ---
    translate_dataset(dataset, translator, nlp_models, pipelines, MODEL_NAME, args.source_column)
    # --- Save Results ---
    print(f"Translations for {MODEL_NAME} completed.")
    # --- Empty Memory --- (potentially can be removed if memory is not an issue)
    del translator
    gc.collect()
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
