# HANDOFF — reprendre le projet (Phases 3, 4, 5)

Ce dépôt contient tout le **code** (Phases 1 & 2 terminées). Les données et
les artefacts dérivés sont **privés** et se transmettent hors GitHub
(OneDrive / clé USB).

## 1. Ce que tu dois recevoir

| Élément | Où | Usage |
|---|---|---|
| Ce dépôt | `git clone https://github.com/tfiras11/sinert-energy-benchmark.git` | tout le code |
| `sinert_handoff_output.zip` | OneDrive / USB (~38 Mo) | artefacts dérivés : données nettoyées, fenêtres ML, résultats benchmark partiels, figures, paramètres de comportement |
| Dossiers de données bruts (`<Mois><Année>/` de juin 2022 à janvier 2026) | OneDrive / USB (**privé**) | nécessaire seulement si tu dois relancer la Phase 1 |
| `contrat_mt.pdf`, `Facture Steg 09-2026.pdf`, `em24dinds.pdf`, cahier des charges | OneDrive / USB (**privé**) | constantes tarifaires + vérité terrain de la Phase 3 |

## 2. Installation

```bash
git clone https://github.com/tfiras11/sinert-energy-benchmark.git
cd sinert-energy-benchmark
pip install -r requirements.txt        # Python ≥ 3.12
```

Dézipper `sinert_handoff_output.zip` → cela crée le dossier `output/`
à côté du code (le CLI attend `output/` par défaut).

Si tu dois relancer la Phase 1 (données bruts) :
```bash
python -I run_phase1.py --data-dir "C:\chemin\vers\dataset" --out-dir output
python -I run_phase1_analysis.py --out-dir output
```
Le chemin des données se règle aussi dans `config.py` (`DEFAULT_DATA_DIR`).
Ne jamais committer `output/` ni les données (le `.gitignore` s'en charge).

## 3. État d'avancement

- [x] **Phase 1** complète : `output/` contient déjà tout
      (`cleaned_15min_imputed.parquet`, `structured/features_15min.parquet`,
      `structured/windows_daily.npz`, `behavior/`, `figures/`)
- [x] **Phase 2** : protocole + baselines (naive_96, naive_672) + RandomForest
      entraînés → `output/benchmark/benchmark_results.csv` (3 lignes) et
      `predictions/`
- [ ] **Phase 2 (suite)** — sur **Google Colab** (machine locale limitée en RAM) :
      uploader `output/benchmark/windows_scaled.npz` + `phase2/colab_benchmark.py`,
      puis `!python colab_benchmark.py --models ml,sarimax` puis
      `--models dl --epochs 60` (runtime GPU). Rapporter
      `colab_results.csv` + `colab_predictions.npz` dans `output/benchmark/`
      et lancer `python -I phase2/merge_colab.py` → table des 13 modèles.
- [ ] **Phase 3** — modèle tarifaire STEG (à créer : `tariff.py` + `run_phase3.py`)
      : postes horaires du contrat MT, cos φ (bonus 0,5 %/centième > 0,90 ;
      pénalités cumulatives < 0,80), prime de puissance, TVA 19 %, surtaxe.
      Entrées : `output/cleaned_15min_imputed.parquet` (facture réelle) +
      `output/benchmark/predictions/*.npz` (facture prédite par modèle).
      Vérité terrain : la facture réelle fournie.
- [ ] **Phase 4** — classement multi-critères (`run_phase4.py`) :
      `benchmark_results.csv` + sortie Phase 3 (ΔFacture TND et %).
- [ ] **Phase 5** — dashboard SCADA (Streamlit) : modules historique /
      arène de benchmarking / financier / alertes, à construire sur
      `output/` (voir le tableau des entrées dans le README).

## 4. Règles importantes (défendables en revue)

- Données privées : **jamais** de commit de données ou de `output/` sur GitHub.
- Août 2025 est **imputé par motif saisonnier** (énergie conservée exactement
  depuis l'écart de compteur) et **flaggé** (`is_imputed`,
  `sample_imputed`) — citer les métriques finales test avec/sans ces échantillons.
- Juin 2022 exclu (`ANALYSIS_START = "2022-07-01"` dans `config.py`).
- Test = nov. 2025 → janv. 2026 : 100 % mesures réelles.
- Le split, les scalers et les métriques sont identiques pour tous les
  modèles (`phase2/data_protocol.py`, `phase2/benchmark.py`) — ne pas y toucher
  sans refaire tourner tous les modèles.
