# FIB Modular Pipeline

Notebook-free Python project for:
- **Internal training/evaluation**
- **External validation**
- **Marker comparator analysis**: FIB-4, NFS, M2BPGi
- **Dual-zone plots**
- **Threshold export / reuse**
- **Excel / JSON artifact saving**

## Structure

```text
fib_modular_project/
├── main.py
├── pyproject.toml
├── configs/
│   ├── internal.yaml
│   └── external.yaml
├── models/
│   ├── __init__.py
│   └── builders.py
├── pipelines/
│   ├── __init__.py
│   ├── internal.py
│   └── external.py
└── utils/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── dca.py
    ├── diagnostics.py
    ├── io_utils.py
    ├── markers.py
    ├── metrics.py
    ├── plots.py
    ├── shap_utils.py
    └── thresholding.py
```

## Quick start

```bash
python main.py internal --config configs/internal.yaml
python main.py external --config configs/external.yaml
```

## Notes

- `main.py` is the single entrypoint.
- Internal pipeline exports model bundles and marker cutoff summaries.
- External pipeline reuses internal bundles and internal marker cutoffs.
- FIB-4 and NFS use fixed cutoffs by default.
- M2BPGi uses internal-derived cutoffs by default.
