# ThermoMPNN FP

`thermompnn_fp` is an FP-style reproduction of the ThermoMPNN paper pipeline, covering preprocessing, training, and inference while keeping the code under `src/thermompnn_fp`.

## Implemented modules

- `config.py`: YAML config loading
- `featurize.py`: PDB parsing and graph feature construction
- `proteinmpnn_backbone.py`: original ProteinMPNN-style frozen feature extractor
- `head.py`: light-attention + MLP stability head
- `pipeline.py`: mutation scoring pipeline
- `datasets.py`: Megascale / FireProt datasets
- `preprocessing.py`: raw CSV curation
- `splits.py`: MMseqs2 split utilities
- `training.py`: training and evaluation loop
- `inference.py`: mutation prediction helpers
- `cli.py`: command line entrypoint

## Important note

The backbone now wraps the original ProteinMPNN graph feature extractor and encoder/decoder layout, and by default points at `ThermoMPNN/vanilla_model_weights/v_48_020.pt` via `configs/local_paths.yaml`.

If no local FP checkpoint is found, inference automatically falls back to `ThermoMPNN/models/thermoMPNN_default.pt` and converts the original ThermoMPNN checkpoint layout into the current FP model layout at load time.

The vendored FireProt training CSV under `ThermoMPNN/data_all/training/fireprot_train.csv` can be used directly with the current loader. The Megascale CSV in that folder still has a different schema, so the default `megascale_curated_csv` remains the local curated file path.

## Configs

- `configs/train_default.yaml`
- `configs/local_paths.yaml`

## Typical workflow

```bash
thermompnn-fp prepare-data --config configs/train_default.yaml --paths configs/local_paths.yaml
thermompnn-fp train --config configs/train_default.yaml --paths configs/local_paths.yaml
thermompnn-fp predict --config configs/train_default.yaml --paths configs/local_paths.yaml --pdb path/to/model.pdb --mutation A23V
```

## References

- Paper: <https://doi.org/10.1073/pnas.2314853121>
- Official repository: <https://github.com/Kuhlman-Lab/ThermoMPNN>
