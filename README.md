# Machine Translation Evaluation for Romansh

Code for the paper ["Expanding the WMT24++ Benchmark with Rumantsch Grischun, Sursilvan, Sutsilvan, Surmiran, Puter, and Vallader"](https://arxiv.org/abs/2509.03148)

The benchmark data are located at https://huggingface.co/datasets/ZurichNLP/wmt24pp-rm – this repo contains the code to reproduce the experiments in the paper.

## Reproducing the results in the paper

### Installation
- `pip install -r requirements.txt`

### Table 2 – Language classifier confusion matrix
- Download the language classifier (900MB size) from https://drive.switch.ch/index.php/s/5WM7aL2Nlo2wNDq
- `python -m scripts.language_classifier_confusion_matrix /path/to/model.bin`

### Table 3 – Cross-variety ChrF
`python -m scripts.cross_variety_scores`

### Table 4 – RM→DE: ChrF/xCOMET
We queried xCOMET-XL via a custom API – adapt the code to run the metric locally if needed.

`python -m scripts.results_rm_to_de`

### Table 5 – DE→RM: ChrF
`python -m scripts.results_de_to_rm`

### Figure 3 – Target variety adherence
`python -m scripts.target_variety_adherence`

### Table 6 – Dataset statistics
`python -m scripts.dataset_stats`

### Tables 7–12 – RM→DE: per-domain ChrF/xCOMET
We queried xCOMET-XL via a custom API – adapt the code to run the metric locally if needed.

`python -m scripts.results_rm_to_de_detailed`

### Tables 13–18 – DE→RM: per-domain ChrF
`python -m scripts.results_de_to_rm_detailed`

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

- Uploaded files are in systems/supertext/uploaded_files/
- Files returned by the web interface are in systems/supertext/outputs/

### LLMs
See instructions in `./wmt-collect-translations/README.md`

## Reproducing the data creation workflow

### Creating the blank Excel files given to the translators
`python -m creation_workflow.create_worksheets`

### Creating the benchmark dataset from the Excel files submitted by the translators
The submitted Excel files are in `creation_workflow/completed_xlsx/`

`python -m creation_workflow.build_dataset`

## Tests

`python -m unittest discover tests`

## Citation

```bibtex
@inproceedings{vamvas-et-al-2025-expanding,
  title        = {Expanding the {WMT}24++ Benchmark with {R}umantsch {G}rischun, {S}ursilvan, {S}utsilvan, {S}urmiran, {P}uter, and {V}allader},
  author       = {Vamvas, Jannis and P\'{e}rez Prat, Ignacio and Soliva, Not Battesta and Baltermia-Guetg, Sandra and Beeli, Andrina and Beeli, Simona and Capeder, Madlaina and Decurtins, Laura and Gregori, Gian Peder and Hobi, Flavia and Holderegger, Gabriela and Lazzarini, Arina and Lazzarini, Viviana and Rosselli, Walter and Vital, Bettina and Rutkiewicz, Anna and Sennrich, Rico},
  year={2025},
  eprint={2509.03148},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2509.03148},
}
```
