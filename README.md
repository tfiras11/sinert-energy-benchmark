# SInERT — Benchmark IA pour la prédiction énergétique de SUP'COM

Pipeline complet de **Phase 1 (ingestion → nettoyage → analyse → structuration)**
et **Phase 2 (benchmark multi-modèles)** du projet
*« Évaluation approfondie, Benchmark multi-critères et Modélisation IA pour la
prédiction énergétique et l'estimation de la facturation de SUP'COM »*
(SUP'COM — INDP3-SysTIC, projet SInERT).

> ⚠️ **Données privées** — ce dépôt ne contient que le code. Les relevés des
> compteurs (VMU-C / Carlo Gavazzi EM24), les factures STEG et tout fichier
> dérivé (`output/`) sont exclus du versionnage (voir `.gitignore`).

## Contexte

Le campus de SUP'COM est équipé d'une infrastructure de télémesure (projet
SInERT) : compteurs intelligents Carlo Gavazzi EM24 + concentrateur VMU-C EM,
transmettant via Modbus les grandeurs électriques échantillonnées toutes les
**15 minutes**. Le pipeline transforme ces exports bruts en un jeu de données
prêt pour l'apprentissage, puis compare **13 modèles d'IA** répartis en 6
familles pour la prévision J+1 à pas de 15 minutes.

## Données attendues (non fournies)

```
dataset/                       # racine des données (privée)
├── <Mois><Année>/             # un dossier par mois (juin 2022 → janvier 2026)
│   └── YYYY-MM-DD_YYYY-MM-DD_Daily Report_Main AC meter.csv(.xlsx)
├── contrat_mt.pdf             # contrat STEG moyenne tension (référence tarifs)
└── Facture Steg *.pdf         # facture réelle (vérité terrain Phase 3)
```

Chaque export contient 193 lignes × 1501 colonnes : un bloc `EM_VIRTUAL`
(compteur principal virtuel) + 13 sous-compteurs (TGBT_TR1/TR2, Tours A/B/C,
clim, onduleurs, PV…). Le chemin des données se passe via `--data-dir`.

## Phase 1 — pipeline de données

```bash
python -I run_phase1.py                  # ingestion + nettoyage (~2 min)
python -I run_phase1_analysis.py         # figures + paramètres de comportement + structuration
```

1. **Ingestion** — ~704 exports (CSV + XLSX), en-têtes normalisés, horodatage
   `DD-MM-YYYY HH:MM:SS`, suppression de la ligne vide `24:00:00`,
   concaténation en une table large indexée par timestamp.
2. **Nettoyage** — grille stricte 15 min, compteurs cumulatifs → énergie par
   pas (première différence ; étalement uniforme sur les trous ≤ 2 j),
   détection de resets, imputation selon la durée : ≤ 1 h linéaire, ≤ 2 j
   saisonnière naïve (t−672 → t−96), > 2 j laissée NaN + flag. Règle de
   valeurs défectueuses (crêtes ≈ 0 avec P_avg normal → NaN + imputation).
   PF dérivé `P/√(P²+Q²)` (les colonnes PF brutes sont vides).
3. **Analyse** — totaux d'énergie (jour/semaine/mois/an), creux & pointes
   (heure, durée), heures d'activation, variabilité, profils par jour de
   semaine / saison tarifaire STEG / période académique, 10 figures.
4. **Structuration** — table 11 variables (4 physiques + 7 calendaires
   cycliques) + fenêtres glissantes 672 pas → cibles 96 pas
   (ΔEa, ΔEr, Pmax, Smax). Août 2025 (interruption) imputé par motif
   saisonnier avec conservation EXACTE de l'énergie lue au compteur, et
   **flaggé** (`is_imputed`).

## Phase 2 — benchmark

**Tâche commune à tous les modèles :** prévision J+1 — entrée = semaine
écoulée (672 × 11), sortie = jour suivant (96 × 4 : ΔEa, ΔEr, Pmax, Smax).
Split chronologique : train ≤ 2025-05-31 (1 059 j) · val 06-31/10/2025 (153 j)
· test 11/2025-01/2026 (92 j, 100 % données réelles). Scalers ajustés sur
le train uniquement.

| Famille | Modèles | Exécution |
|---|---|---|
| Baseline | persistance saisonnière t−96, t−672 | local |
| Statistique | SARIMAX (2,0,2) + exog calendaires (Fourier), walk-forward | Colab |
| Machine Learning | RandomForest, XGBoost, LightGBM, SVR (PCA) | RF local, reste Colab |
| RNN | LSTM, Bi-LSTM, GRU | Colab (PyTorch) |
| CNN / Résiduel | CNN-1D, TCN / ResNet-1D (blocs résiduels + BatchNorm) | Colab |
| Transformers / Hybride | Vanilla Transformer, CNN-LSTM | Colab |

Métriques (Phase 4 du cahier de charges) : MAE, RMSE, R² sur les profils ;
MAPE sur les énergies journalières ; **PDE** (erreur de pointe) ; n_paramètres
et temps d'entraînement. Les prédictions val/test de chaque modèle sont
sauvegardées (`predictions/<modèle>.npz`) pour la Phase 3 (ΔFacture STEG).

### Workflow Colab (modèles lourds — machine locale limitée en RAM)

```bash
# 1) générer le fichier de données scaled
python -I run_phase2.py            # crée output/benchmark/windows_scaled.npz
# 2) uploader windows_scaled.npz + phase2/colab_benchmark.py sur Colab, puis :
#    !python colab_benchmark.py --models ml,sarimax
#    !python colab_benchmark.py --models dl --epochs 60
# 3) rapporter colab_results.csv + colab_predictions.npz dans output/benchmark/
python -I phase2/merge_colab.py    # fusionne -> benchmark_results.csv complet
```

### Résultats (test, en cours — DL sur Colab)

| Modèle | RMSE ΔEa (kWh) | MAPE ΔEa journalier | PDE Pmax |
|---|---|---|---|
| naive_96 (baseline) | 2.84 | 12.2 % | 18.1 % |
| naive_672 (baseline) | 3.38 | 15.5 % | 21.7 % |
| RandomForest | **2.12** | **9.0 %** | **17.0 %** |

## Structure du dépôt

```
├── config.py                # chemins, blocs compteurs, variables, règles
├── ingestion.py             # Phase 1.1
├── cleaning.py              # Phase 1.2 (+ imputation longue durée par motif)
├── behavior.py              # Phase 1.4 — paramètres de comportement
├── visualize.py             # Phase 1.3 — figures
├── structure.py             # Phase 1.5 — 11 variables + fenêtres 672
├── run_phase1.py            # CLI ingestion + nettoyage
├── run_phase1_analysis.py   # CLI analyse + structuration
├── run_phase2.py            # CLI benchmark local (baselines, RF, SARIMAX)
├── phase2/
│   ├── data_protocol.py     # splits, scalers, export Colab
│   ├── baselines.py         # persistances t-96 / t-672
│   ├── sarimax_model.py     # SARIMAX walk-forward
│   ├── ml_models.py         # RandomForest (local)
│   ├── benchmark.py         # moteur de métriques + registre des prédictions
│   ├── colab_benchmark.py   # XGB, LGBM, SVR, SARIMAX + 7 DL (à lancer sur Colab)
│   └── merge_colab.py       # fusion des résultats Colab
└── docs/PHASE1_PHASE2_details.md   # documentation détaillée du pipeline
```

## Environnement

Python ≥ 3.12 — `pip install -r requirements.txt`
(pandas, numpy, pyarrow, matplotlib, seaborn, openpyxl, scikit-learn,
xgboost, lightgbm, statsmodels, torch).

`python -I` (mode isolé) est utilisé pour garantir qu'aucun module n'est
jamais importé depuis les dossiers de données.

## Feuille de route

- [x] Phase 1 — ingestion, nettoyage, imputation, comportement, structuration
- [x] Phase 2 — protocole, baselines, RF (local) ; le reste sur Colab
- [ ] Phase 2 — résultats DL (Colab) + fusion du benchmark complet
- [ ] Phase 3 — modélisation tarifaire STEG & ΔFacture (TND)
- [ ] Phase 4 — évaluation multi-critères consolidée
- [ ] Phase 5 — tableau de bord SCADA interactif (Streamlit/Dash)
