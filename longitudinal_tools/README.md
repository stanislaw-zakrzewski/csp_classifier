# Longitudinal Tools Package

Zestaw narzędzi i modułów dla **Eksperymentu 10** w projekcie `csp_classifier`, służący do długofalowej ewaluacji wielosesyjnej na autorskim zbiorze danych `longitudal_dataset`.

## Struktura Modułu

- `longitudinal_dataset_loader.py`: Ładowanie i pre-processing sygnałów EEG z plików EDF (`*_DRY.edf` oraz `*_WET.edf`). Automatycznie wykrywa wszystkie dostępne sesje w `longitudal_dataset/`, normalizuje nazewnictwo kanałów (np. `CZ` -> `Cz`), przeprowadza filtrowanie pasmowe (8–30 Hz), resampling do 128 Hz oraz epokowanie.
- `train_longitudinal_models.py`: Trening modeli od zera (*scratch*) oraz fine-tuning pre-trenowanych modeli klastrowych GNN na $k$ sesjach treningowych ($k \in \{1, \dots, N-1\}$). Obserwuje 6 podejści:
  1. CSP-LDA
  2. tGSP-Cov (COV-TGSP + LR)
  3. EEGNet (Longitudinal Scratch)
  4. ATCNet (Longitudinal Scratch)
  5. EEGNet Cluster Pretrained + Fine-Tuned (Head-Only)
  6. ATCNet Cluster Pretrained + Fine-Tuned (Head-Only)
- `simulate_longitudinal_adaptive.py`: Przeprowadzanie symulacji statycznej (Zero-Shot) oraz adaptacji online (*Head-Only Adaptation*, micro-batch $I=8$, buffer $M=64$) na sesjach testowych $(k+1..N)$.
- `evaluate_longitudinal_benchmarks.py`: Agregacja wyników, kalkulacja metryk uogólniania $k$-sesyjnego, porównanie elektrod DRY vs WET oraz generowanie wykresów trajektorii adaptacji w 10 binarowych kwantylach czasowych (Bin 0–9).

## Struktura Danych `longitudal_dataset`

Zbiór umieszczony jest w folderze `longitudal_dataset/` i posiada podkatalogi sesji:
```
longitudal_dataset/
├── session1/
│   ├── 2026-08-06T10-49-11_DRY.edf
│   └── 2026-08-06T11-08-49_WET.edf
├── session2/
│   ├── 2026-08-07T10-33-04_DRY.edf
│   └── 2026-08-07T10-51-45_WET.edf
└── sessionN/ ...
```

## Sposób Uruchomienia

Główny potok eksperymentalny uruchamia się z poziomu głównego katalogu projektu:
```bash
python run_exp10_longitudinal_pipeline.py
```
