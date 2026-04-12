# Hailo 8L — Benchmark d'inférence de modèles temporels

Ce repo évalue la latence d'inférence de différentes architectures de réseaux de neurones temporels sur un **Raspberry Pi équipé d'un accélérateur Hailo 8L**, en comparaison avec une exécution CPU pure (ONNX Runtime).

---

## Prérequis selon l'environnement

| Tâche | Environnement requis |
|---|---|
| Génération des modèles ONNX | Ubuntu + Hailo SDK (`hailo_venv`) |
| Compilation ONNX → HEF | Ubuntu + Hailo DFC (`hailo_sdk_client`) |
| Benchmark CPU + Hailo | Raspberry Pi + Hailo 8L (PCIe/M.2) + HailoRT |

> Les fichiers `.hef` compilés sont versionnés dans le repo — il n'est donc pas nécessaire de recompiler pour lancer les benchmarks.

---

## Structure du repo

```
hailo/
├── models/
│   ├── mlp/           # Multi-Layer Perceptron
│   │   ├── export.py      # Génère les ONNX (3 tailles)
│   │   ├── compile.py     # ONNX -> HAR -> HEF (Hailo DFC)
│   │   ├── onnx/          # Modèles ONNX exportés
│   │   ├── har/           # Fichiers HAR (parsé + quantifié int8)
│   │   └── hef/           # Fichiers HEF compilés pour Hailo 8L
│   │
│   ├── tcn/           # Temporal Convolutional Network
│   │   ├── export.py      # Génère les ONNX (Conv2D NCHW pour Hailo)
│   │   ├── compile.py     # ONNX -> HAR -> HEF
│   │   ├── onnx/
│   │   ├── har/
│   │   └── hef/
│   │
│   ├── lstm/          # Long Short-Term Memory (CPU uniquement)
│   │   ├── export.py      # Génère les ONNX
│   │   └── onnx/
│   │
│   └── tft/           # Temporal Fusion Transformer (CPU uniquement)
│       ├── parse_tft.py   # Tente le parsing ONNX -> HAR avec Hailo DFC
│       └── tft_duree_fin_cluster0.onnx
│
└── benchmark/
    ├── run_all.py         # Lance tous les benchmarks et affiche un tableau récapitulatif
    ├── run_cpu.py         # Benchmark CPU pour un modèle ONNX individuel
    ├── run_hailo.py       # Benchmark Hailo pour un fichier HEF individuel
    └── run_tft_cpu.py     # Benchmark CPU spécifique au TFT (entrées multi-tenseurs)
```

---

## Architectures et configurations

### MLP — Multi-Layer Perceptron
Architecture : `Linear → ReLU → ... → Linear`  
Input 1D, trois tailles disponibles :

| Modèle | input_dim | hidden_dim | output_dim | couches |
|---|---|---|---|---|
| mlp_small | 64 | 128 | 16 | 3 |
| mlp_medium | 128 | 256 | 32 | 4 |
| mlp_large | 256 | 512 | 64 | 5 |

Compatible Hailo 8L : **oui**

---

### TCN — Temporal Convolutional Network
Architecture : `Conv1D (via Conv2D NCHW) → ReLU + résidu → GlobalAvgPool → Linear`  
Input shape : `[1, in_channels, 1, seq_len]`

| Modèle | in_channels | hidden | seq_len | couches | kernel |
|---|---|---|---|---|---|
| tcn_small | 32 | 64 | 64 | 4 | 3 |
| tcn_medium | 64 | 128 | 64 | 6 | 3 |
| tcn_large | 64 | 256 | 128 | 8 | 3 |

> Conv1D est représenté en Conv2D avec kernel `[1, K]` — format natif Hailo.

Compatible Hailo 8L : **oui**

---

### LSTM — Long Short-Term Memory
Architecture : `Transpose → LSTM (1-2 couches) → dernier état caché → Linear`

| Modèle | input_size | hidden_size | seq_len | couches |
|---|---|---|---|---|
| lstm_small | 32 | 64 | 32 | 1 |
| lstm_medium | 64 | 128 | 64 | 1 |
| lstm_large | 64 | 256 | 64 | 2 |

Compatible Hailo 8L : **non** — le compilateur Hailo déroule les LSTM dans le temps, créant une chaîne séquentielle incompatible avec l'architecture parallèle du Hailo. CPU uniquement.

---

### TFT — Temporal Fusion Transformer
Modèle réel (`tft_duree_fin_cluster0.onnx`) avec encodeur/décodeur multi-entrées (séries continues, catégorielles, embeddings).

Compatible Hailo 8L : **non** — architecture trop complexe (attention, LSTM interne). CPU uniquement.

---

## Workflow : générer et compiler les modèles (Ubuntu)

### 1. Exporter en ONNX

```bash
cd models/mlp
python export.py        # génère onnx/mlp_{small,medium,large}.onnx

cd ../tcn
python export.py        # génère onnx/tcn_{small,medium,large}.onnx

cd ../lstm
python export.py        # génère onnx/lstm_{small,medium,large}.onnx
```

### 2. Compiler en HEF (nécessite le Hailo DFC)

```bash
source ~/hailo_venv/bin/activate

cd models/mlp
python compile.py --all    # ou --model mlp_small

cd ../tcn
python compile.py --all
```

Le pipeline DFC se déroule en 3 étapes automatiques :
1. Parsing ONNX → HAR (`.har`)
2. Calibration int8 avec données aléatoires → HAR quantifié (`_quantized.har`)
3. Compilation → HEF (`.hef`)

---

## Lancer les benchmarks (Raspberry Pi + Hailo 8L)

### Tout lancer d'un coup

```bash
python benchmark/run_all.py
python benchmark/run_all.py --runs 500
python benchmark/run_all.py --no-hailo   # CPU uniquement (sans matériel Hailo)
```

Affiche un tableau récapitulatif avec latence CPU, latence Hailo et speedup :

```
==============================================================
Model                CPU (ms)   Hailo (ms)      Speedup
--------------------------------------------------------------
  [mlp]
  mlp_small             X.XXX        X.XXX         X.Xx
  mlp_medium            X.XXX        X.XXX         X.Xx
  ...
  [lstm (cpu only)]
  lstm_small            X.XXX            —         cpu only
==============================================================
```

### Benchmark individuel

```bash
# CPU
python benchmark/run_cpu.py --onnx models/mlp/onnx/mlp_small.onnx --runs 200

# Hailo
python benchmark/run_hailo.py --hef models/mlp/hef/mlp_small.hef --runs 200

# TFT (CPU uniquement)
python benchmark/run_tft_cpu.py --runs 200
```

---

## Dépendances

**Ubuntu (génération + compilation) :**
```
onnx==1.16.0
numpy
hailo_sdk_client   # via Hailo DFC (hailo_venv)
```

**Raspberry Pi (benchmark) :**
```
onnxruntime
numpy
hailo-platform     # inclus dans HailoRT
```
