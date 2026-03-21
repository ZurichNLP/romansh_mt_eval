# Machine Translation Evaluation for Romansh

Code for the papers
- ["Expanding the WMT24++ Benchmark with Rumantsch Grischun, Sursilvan, Sutsilvan, Surmiran, Puter, and Vallader"](https://aclanthology.org/2025.wmt-1.79/)
- "Translation Asymmetry in LLMs as a Data Augmentation Factor: A Case Study for 6 Romansh Language Varieties"

The benchmark data are located at https://huggingface.co/datasets/ZurichNLP/wmt24pp-rm and the human quality ratings are located at https://huggingface.co/datasets/ZurichNLP/romansh-mt-evaluation – this repo contains the code to reproduce the experiments in the papers.

## Reproducing the results in "Expanding the WMT24++ Benchmark with Rumantsch Grischun, Sursilvan, Sutsilvan, Surmiran, Puter, and Vallader"

### Installation
- `pip install -r requirements.txt`

### Table 2 – Language classifier confusion matrix
- Download the language classifier (900MB size) from https://drive.switch.ch/index.php/s/5WM7aL2Nlo2wNDq
- `python -m scripts.benchmark_paper.language_classifier_confusion_matrix /path/to/model.bin`

### Table 3 – Cross-variety ChrF
`python -m scripts.benchmark_paper.cross_variety_scores`

### Table 4 – RM→DE: ChrF/xCOMET
We queried xCOMET-XL via a custom API – adapt the code to run the metric locally if needed.

`python -m scripts.benchmark_paper.results_rm_to_de`

### Table 5 – DE→RM: ChrF
`python -m scripts.benchmark_paper.results_de_to_rm`

### Figure 3 – Target variety adherence
`python -m scripts.benchmark_paper.target_variety_adherence`

### Table 6 – Dataset statistics
`python -m scripts.benchmark_paper.dataset_stats`

### Tables 7–12 – RM→DE: per-domain ChrF/xCOMET
We queried xCOMET-XL via a custom API – adapt the code to run the metric locally if needed.

`python -m scripts.benchmark_paper.results_rm_to_de_detailed`

### Tables 13–18 – DE→RM: per-domain ChrF
`python -m scripts.benchmark_paper.results_de_to_rm_detailed`

xCOMET-XL is queried via the existing API client where applicable; run ChrF-only or adapt for local COMET if needed.

## Reproducing the results in "Translation Asymmetry in LLMs as a Data Augmentation Factor: A Case Study for 6 Romansh Language Varieties"

### Installation
- `pip install -r requirements.txt`

### Splitting WMT24++ into validation and test sets
- `python -m scripts.mt_paper.split_wmt24pp_dataset`
- This creates `benchmarking/wmt24pp_split.json`
- The first half of documents per domain is used as the validation split, and the remaining half is used as the test split

### Table 1 – Automatic evaluation results
We queried xCOMET-XL via a custom API - adapt the code to run the metric locally if needed.

`python -m scripts.mt_paper.results_automatic`

### Table 2 – Human evaluation results
`python human_evaluation/scripts/create_results_table.py`

### Figure 2b – Target variety adherence
`python -m scripts.mt_paper.target_variety_adherence_dual`

### Table 4 – Validation results for back-translation from Romansh
We queried xCOMET-XL via a custom API - adapt the code to run the metric locally if needed.

`python -m scripts.mt_paper.results_validation_backtranslation`

### Appendix L – Detailed automatic evaluation results
We queried xCOMET-XL via a custom API - adapt the code to run the metric locally if needed.

`python -m scripts.mt_paper.results_detailed_automatic`

`python -m scripts.mt_paper.target_variety_adherence_all_systems`

### Appendix M – Detailed human evaluation results

`python human_evaluation/scripts/create_detailed_results_table.py`

`python human_evaluation/scripts/pairwise_system_comparison.py`

### Appendix N – Human evaluation statistics

`python human_evaluation/scripts/create_human_evaluation_statistics_table.py`

`python human_evaluation/scripts/create_extended_inter_annotator_agreement.py`

## Collecting system translations

The collected system translations are stored in `systems/`.

### Installation

- `pip install -r requirements.txt`
- PyTorch needed for MADLAD-400

### MADLAD-400
```bash
for variety in rm-rumgr rm-sursilv rm-sutsilv rm-surmiran rm-puter rm-vallader; do
    python -m systems.madlad.run_translate "${variety}:de" google/madlad400-10b-mt <output_file>  # direct
    python -m systems.madlad.run_translate "${variety}:en:de" google/madlad400-10b-mt <output_file>  # pivoting via English
done
```

### Translatur-ia
`python -m systems.translaturia.collect_translations`

### Supertext
Translations were collected manually via the web interface of Supertext.

- Uploaded files are in `systems/supertext/uploaded_files/`
- Files returned by the web interface are in `system_translations/wmt25_oldi/supertext/outputs/`

### LLMs
See instructions in `./wmt-collect-translations/README.md`

## Reproducing the benchmark data creation workflow

### Creating the blank Excel files given to the translators
`python -m creation_workflow.create_worksheets`

### Creating the benchmark dataset from the Excel files submitted by the translators
The submitted Excel files are in `creation_workflow/completed_xlsx/`

`python -m creation_workflow.build_dataset`

## Tests

```bash
python -m unittest discover -s romansh_mt_eval/tests
python -m unittest discover -s romansh_mt_eval/human_evaluation/tests
```

## Citation

```bibtex
@inproceedings{vamvas-et-al-2025-expanding,
    title = "Expanding the {WMT}24++ Benchmark with {R}umantsch {G}rischun, {S}ursilvan, {S}utsilvan, {S}urmiran, {P}uter, and {V}allader",
    author = "Vamvas, Jannis  and  P{\'e}rez Prat, Ignacio  and  Soliva, Not  and  Baltermia-Guetg, Sandra  and  Beeli, Andrina  and  Beeli, Simona  and  Capeder, Madlaina  and  Decurtins, Laura  and  Gregori, Gian Peder  and  Hobi, Flavia  and  Holderegger, Gabriela  and  Lazzarini, Arina  and  Lazzarini, Viviana  and  Rosselli, Walter  and  Vital, Bettina  and  Rutkiewicz, Anna  and  Sennrich, Rico",
    editor = "Haddow, Barry  and  Kocmi, Tom  and  Koehn, Philipp  and  Monz, Christof",
    booktitle = "Proceedings of the Tenth Conference on Machine Translation",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.wmt-1.79/",
    pages = "1028--1047",
    ISBN = "979-8-89176-341-8",
}
```
