# TRIAD replication package

Everything in the paper is computed by the scripts in `src/` from public artefacts cloned into `data/`.

## Inputs (clone into data/)
- data/experiments  : github.com/SWE-bench/experiments (evaluation/verified/*/results/results.json, per_instance_details.json, metadata.yaml), HEAD 40f164d
- data/paichecker   : github.com/manyiResearch/PAIChecker (labelled_data/), HEAD 0f0a2ae
- data/patchdiff    : github.com/ZJU-CTAG/PatchDiff (dataset + three tools' predictions), HEAD 0a8ed60
- data/utboost      : github.com/uiuc-kang-lab/UTBoost (assets/useful_scripts/augTest.json), HEAD 1b84ea3

## Run order
python3 src/build.py      # base table + 500 x 175 response matrix
python3 src/fix_base.py   # restore task fields truncated in the PAIChecker CSV (required)
python3 src/features.py   # static gold-patch and test-patch features, external labels
python3 src/footprint.py  # passing agent patches vs gold file footprint
python3 src/a2.py         # axes, overlap, item statistics, 2PL, regression  -> out/res_a2.json
python3 src/a3.py         # frontier, strength trends, re-ranking, cost      -> out/res_a3.json
python3 src/a4.py         # PAIChecker re-ranking replication + nulls, triage -> out/res_a4.json
python3 src/a5.py         # robustness, power, reliability, Multilingual   -> out/res_a5.json
python3 src/figs2.py      # Figures 1-9 of the paper (out/F*.png)

## Outputs
- out/triad_defect_cards.csv : one row per instance, every flag and statistic
- out/triad_clean_369.txt    : instances with no flag on any axis
- out/res_a*.json            : every number reported in the paper
- out/F1..F9 *.png
Python 3.13, numpy, pandas, scipy, scikit-learn, statsmodels, pyarrow, matplotlib, pyyaml.
