# ReAM: Reliability-Aware Address Matching

Reference implementation of **ReAM**, a reliability-aware fusion model for cross-source geospatial entity matching. ReAM separates evidence **availability** from candidate-pair-specific **reliability** and fuses independent text, spatial, and neighborhood decisions according to the estimated probability that each branch decision is correct.

## Method implemented here

The implementation uses the following configuration:

- three evidence branches: text, spatial, and neighborhood;
- `bert-base-uncased`, maximum sequence length 128, 128-dimensional branch representations;
- spatial features based on great-circle distance, local density, nearest-alternative margin, and category-specific spatial scale;
- neighborhood radius 500 m, at most 15 neighbors per side;
- character CNN for neighbor names with kernel widths 2/3/4/5, 32 channels each, 32-dimensional character embeddings;
- 16 radial basis functions for relative distance;
- bidirectional soft neighborhood correspondence with default `beta=0.2` and `T=1.0`;
- two-layer reliability heads with hidden size 64, GELU, dropout 0.1, and sigmoid output;
- availability-aware reliability normalization over available branches;
- five branch-only warm-up epochs followed by joint training;
- branch correctness threshold 0.5 and epoch-wise correctness-target refresh;
- final objective `L_match + 0.3 L_branch + 0.2 L_rel`;
- AdamW, text-encoder learning rate `2e-5`, other-module learning rate `1e-4`, weight decay 0.01;
- batch size 64, at most 25 epochs, patience 5, BF16 on supported CUDA hardware;
- controlled text masking, coordinate shift, neighborhood deletion, and attribute conflict;
- ECE (15 bins), Brier score, risk-coverage curve, AURC, and branch reliability calibration;
- reliability, attention, confidence, static-fusion, no-quality, no-reliability-supervision, and no-degradation variants.

## Reference experiment environment

The code runs in the following reference environment:

| Component | Reference configuration |
|---|---|
| Operating system | Ubuntu 22.04 LTS (64-bit) |
| Python | 3.10.14 |
| PyTorch | 2.3.1 |
| CUDA toolkit/runtime | 12.1 |
| Transformers | 4.41.2 |
| NumPy | 1.26.4 |
| scikit-learn | 1.5.0 |
| PyYAML | 6.0.1 |
| tqdm | 4.66.4 |
| GPU | 1 x NVIDIA RTX A6000, 48 GB |
| Mixed precision | BF16 |

For an exact Python dependency set, use `requirements-lock.txt`. A Conda-compatible environment specification is provided in `environment.yml`. A recent NVIDIA driver supporting CUDA 12.1 is required for GPU execution. CPU execution is supported for preprocessing and tests, but full model training is designed for CUDA.

```bash
conda env create -f environment.yml
conda activate ream
```

Alternatively:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

A CUDA-enabled PyTorch installation is recommended for training.

## Data

The experiments use the public **Geo-ER** benchmark introduced by Balsebre et al. (WWW 2022). The official repository provides the eight OSM-Foursquare / OSM-Yelp tasks for Singapore, Edinburgh, Toronto, and Pittsburgh with fixed train/validation/test splits:

https://github.com/PasqualeTurin/Geo-ER

The benchmark is not redistributed in this repository; keep the original files outside it (for example under `data/`). The model code uses a canonical JSONL representation so that all three branches and their quality descriptors are explicit.

### Canonical JSONL schema

Each line is one candidate pair:

```json
{
  "id": "sin-osm_fsq-000001",
  "city": "singapore",
  "source_pair": "osm_fsq",
  "label": 1,
  "left": {
    "name": "Example Place",
    "address": "12 Main Street",
    "postal_code": "12345",
    "category": "cafe",
    "lat": 1.3000,
    "lon": 103.8000,
    "neighbors": [
      {"name": "Neighbor A", "category": "restaurant", "distance_m": 83.2}
    ]
  },
  "right": {
    "name": "Example Place",
    "address": "12 Main St",
    "postal_code": "12345",
    "category": "cafe",
    "lat": 1.3001,
    "lon": 103.8001,
    "neighbors": []
  },
  "nearest_candidate_margin_m": 41.7,
  "category_key": "cafe"
}
```

The optional `nearest_candidate_margin_m` and `category_key` fields support the spatial quality descriptors. `fit_statistics` derives typical spatial scales and normalization statistics from the rows it is given, so running it on training rows keeps validation and test data out of those statistics. In leave-one-city-out experiments it is run on the three training cities.

### Converting Geo-ER pair files

Geo-ER serializes a pair as `<entity1>\t<entity2>\t<label>`. Convert a split with:

```bash
python scripts/prepare_geoer.py pairs \
  --input data/raw/train.txt \
  --output data/prepared/train.jsonl \
  --city singapore \
  --source-pair osm_fsq
```

If neighborhood JSON is available in the format documented by the Geo-ER repository, attach it with:

```bash
python scripts/prepare_geoer.py neighbors \
  --pairs data/prepared/train.jsonl \
  --neighbors data/raw/train_neighbors.json \
  --output data/prepared/train_with_neighbors.jsonl
```

## Training

```bash
python scripts/train.py \
  --train data/prepared/train.jsonl \
  --valid data/prepared/valid.jsonl \
  --output outputs/sin_osm_fsq
```

The best validation-F1 checkpoint is written to `outputs/.../best.pt`. The final decision threshold is selected from the validation set and stored in the checkpoint.

## Evaluation

```bash
python scripts/evaluate.py \
  --data data/prepared/test.jsonl \
  --checkpoint outputs/sin_osm_fsq/best.pt \
  --output outputs/sin_osm_fsq/test_metrics.json
```

The output contains final F1, ECE, Brier score, AURC, branch calibration bins, final probabilities, branch probabilities, reliability values, and fusion weights.

## Controlled degradation

Generate the single-evidence test conditions used for robustness curves:

```bash
python scripts/degradation_eval.py --data data/prepared/test.jsonl --kind text_mask --output-dir data/degraded/text
python scripts/degradation_eval.py --data data/prepared/test.jsonl --kind coordinate_shift --output-dir data/degraded/spatial
python scripts/degradation_eval.py --data data/prepared/test.jsonl --kind neighborhood_delete --output-dir data/degraded/neighborhood
python scripts/degradation_eval.py --data data/prepared/test.jsonl --kind attribute_conflict --output-dir data/degraded/conflict
```

The compound condition uses 30% text masking and a 100 m coordinate shift:

```bash
python scripts/compound_eval.py \
  --data data/prepared/test.jsonl \
  --text-mask 0.30 \
  --coordinate-shift-m 100 \
  --output data/degraded/compound.jsonl
```

Controlled test degradation changes only the target evidence state. Candidate pairs and labels remain fixed.

## Ablations

```bash
bash scripts/run_ablation.sh data/prepared/train.jsonl data/prepared/valid.jsonl outputs/ablation
```

The script trains:

- full ReAM;
- static averaging without the reliability module;
- Attention+CD;
- Confidence+CD;
- ReAM without reliability supervision;
- ReAM without quality features;
- ReAM without controlled-degradation training.

## Leave-one-city-out evaluation

Arrange prepared data as:

```text
data/prepared/
  osm_fsq/{singapore,edinburgh,toronto,pittsburgh}/{train,valid,test}.jsonl
  osm_yelp/{singapore,edinburgh,toronto,pittsburgh}/{train,valid,test}.jsonl
```

Then run:

```bash
bash scripts/run_cross_city.sh data/prepared outputs/cross_city
```

Each fold pools training and validation data from the other three cities and uses only the held-out city's test splits for evaluation.

## Experimental protocol

### Eight benchmark tasks

The main evaluation uses the fixed Geo-ER splits for the following eight tasks:

- Singapore: OSM-Foursquare and OSM-Yelp;
- Edinburgh: OSM-Foursquare and OSM-Yelp;
- Toronto: OSM-Foursquare and OSM-Yelp;
- Pittsburgh: OSM-Foursquare and OSM-Yelp.

Each main-task result is reported from five runs. The data split is fixed; only model initialization, mini-batch order, and degradation random state change between runs. Seeds are `13, 29, 47, 71, 101`.

### Training schedule

- Maximum epochs: 25.
- Branch-only warm-up: first 5 epochs.
- Early stopping: validation F1, patience 5.
- Batch size: 64.
- Text encoder learning rate: `2e-5`.
- Remaining modules learning rate: `1e-4`.
- AdamW weight decay: `0.01`.
- Dropout: `0.1`.
- Objective: `L_match + 0.3 L_branch + 0.2 L_rel`.
- Branch correctness threshold: `0.5`.
- Final matching threshold: selected on validation F1 only.
- Mixed precision: BF16 on supported CUDA hardware.

### Controlled-degradation mixture during training

The training sampler uses 50% original-evidence samples and 50% degraded samples. Within the degraded half, 70% receive one degradation and 30% receive a two-evidence compound degradation, yielding the overall 50% / 35% / 15% original / single / compound proportions.

Single-evidence degradation ranges are:

- text masking probability: 0.10-0.40;
- coordinate shift: 20-200 m;
- neighborhood deletion ratio: 0.20-0.60;
- attribute conflict: donor-based replacement from the same city, source pair, and split while excluding the true match.

The compound condition is 30% text masking plus a 100 m coordinate shift.

### Metrics

F1 is the primary matching metric. Probability quality and selective prediction are evaluated with 15-bin ECE, Brier score, risk-coverage curves, and AURC. Branch reliability calibration uses 15 equal-width reliability bins.

## Reproducibility notes

- The official Geo-ER candidate sets and fixed splits are used unchanged.
- Five repeated runs differ only in model initialization, mini-batch order, and degradation RNG seed.
- No positive/negative resampling or class weighting is applied by ReAM.
- Branch correctness uses a fixed 0.5 threshold; the final matching threshold is selected independently on validation F1.
- Outside controlled robustness evaluation, inference does not inject degradation.
- Attribute-conflict donors are drawn from the current data partition and never from the true matching entity. `ControlledDegrader` accepts a `donor_lookup` hook; `PartitionDonorIndex` implements the partition-local donor index.
- Spatial scale and feature-standardization statistics are learned from the training partition only.

## Tests

```bash
pip install pytest
pytest -q
```

## Citation

If you use the Geo-ER benchmark, cite the original Geo-ER paper and repository.

## Repository layout

```text
configs/                 model/training configuration
scripts/                 data preparation, training, evaluation, robustness, ablation, cross-city entry points
src/ream/                ReAM model and core library
tests/                   unit tests for degradation and metrics
requirements.txt         minimum dependency ranges
requirements-lock.txt    pinned release environment
environment.yml          Conda-compatible reference environment
README.md                 English documentation
README_CN.md              Chinese documentation
```

## Expected outputs

Training writes a best checkpoint selected by validation F1. Evaluation JSON files contain the final threshold, F1, ECE, Brier score, AURC, final probabilities, branch probabilities, estimated reliabilities, normalized fusion weights, and branch-calibration bins. Robustness scripts emit one result file per degradation intensity, so robustness curves are reconstructed from model outputs without changing candidate sets or labels.
