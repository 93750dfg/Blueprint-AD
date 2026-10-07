# Blueprint-AD: Pose-Aligned Object-Centric Spatial Reference

Code release for **Blueprint-AD**.

This repository provides the EfficientAD, CFlow-AD, and PatchCore implementations used in our experiments.

We provide precomputed Blueprint-AD intermediate artifacts for a subset of categories under `runs/`. These categories can be run directly with the released Blueprint-AD integration code.

Part of the Blueprint-AD preprocessing code is not included in the current release and will be added in a later update.

The paths in the commands below are examples. Please replace them with your local dataset paths.

## Running EfficientAD

Baseline:

```bash
python efficientad/baseline/efficientad.py -d mvtec_ad -s cable -a "E:\data\MVtec" -i "E:\data\dtd\images"
```

Blueprint-AD:

```bash
python efficientad/blueprint/efficientad.py -d mvtec_ad -s cable -a "E:\data\MVtec" -i "E:\data\dtd\images"
```

## Running CFlow-AD

Baseline:

```bash
python cflow/baseline/run.py --dataset mvtec --classes cable --mvtec_ad_path "E:\data\MVtec"
```

Blueprint-AD:

```bash
python cflow/blueprint/run.py --dataset mvtec --classes cable --mvtec_ad_path "E:\data\MVtec"
```

## Running PatchCore

Baseline:

```bash
python patchcore/baseline/run.py --dataset mvtec --classes cable --mvtec_ad_path "E:\data\MVtec"
```

Blueprint-AD:

```bash
python patchcore/blueprint/run.py --dataset mvtec --classes cable --mvtec_ad_path "E:\data\MVtec"
```

More documentation and code will be added in later updates.
