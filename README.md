# Siamese Nueral Network For Near Duplicate Detection in Web App Model Inference

## 🛠️ Prerequisites

## 🚀 Get Started

Follow these steps to create and activate the Conda environment to set up the project.


### 1) Clone the repository

```bash
git clone https://github.com/ast-fortiss-tum/near-duplicate-detection-siamese-networks.git
cd near-duplicate-detection-siamese-networks
```

### 2) Create the Conda environment

The `environment.yml` already pins compatible versions (Python 3.11, NumPy < 2, SciPy 1.10.1, etc.) and contains the line `- -e .` so your code is installed automatically.

```bash
conda env create --name snn-ndd -f environment.yml
```

*What this does*

1. Creates a new Conda env called **`snn-ndd`**.
2. Installs all Conda‑managed packages (PyTorch, transformers, gensim, …).
3. Runs `pip install -e .` inside the env, wiring up your local source tree as a package named **`ndd`**.

### 3) Activate the environment

```bash
conda activate snn-ndd
```


## 📂 Repository Structure

```
├── resources/                # Downloaded baseline datasets, doms, baseline runners (jar) and doc2vec embedding models
├── scripts/                  # Python scripts for running evaluations
│   ├── rq1/
│   │   ├── within-app-classification/
│   │   │   ├── bert_contrastive_classification.py
│   │   │   ├── doc2vec_contrastive_classification.py
│   │   │   └── markuplm_contrastive_classification.py
│   │   ├── across-app-classification/
│   │   │   └── [...same naming as above...]
│   │   └── baseline-classification/
│   │       └── run_baseline.py
│   ├── rq2/                   # RQ2 evaluation scripts
│   ├── rq3/                   # RQ3 evaluation scripts
│   └── rq4/                   # RQ4 evaluation scripts
├── results/                  # Generated experiment results
│   ├── rq1/
│   ├── rq2/
│   ├── rq3/
│   └── rq4/
├── baseline/                 # Baseline results
├── models/                   # Saved trained models
├── embedding/                # Cached embedding files
└── README.md                 
```


## 🖥️ Generate Reported Results


### 1) RQ1: Near-Duplicate Detection

#### a) SNN Evaluation

1. **Download resources**
   Get `resources.zip` from:

   ```
   https://syncandshare.lrz.de/getlink/fiY4KcSXCwSLW8g8TVmnep/resources.zip
   ```

   *Password: `snn-ndd`*

2. **Unpack and move the resources folder to the base project**

   ```bash
   unzip resources.zip -d resources
   ```

   You should now have folders:

   ```
   dataset
   scripts
   resources/
   ├── baseline-dataset/
   ├── baseline-runner/
   ├── doms/
   └── embedding-models/
   ```

3. **Run the evaluation**

   ```bash
   python scripts/rq1/<evaluation_setting>-app-classification/<embedding_type>_contrastive_classification.py
   ```

   * `<evaluation_setting>`:

     * `within`
     * `across`
   * `<embedding_type>`:

     * `bert` (select or adjust the variant inside the script)
     * `doc2vec`
     * `markuplm`

4. **Outputs**

   * Experiment results → `results/rq1/`
   * Trained models → `models/`
   * Cached embeddings → `embedding/`

> **Re-runs** will automatically reuse any existing models or embeddings; nothing is re-trained if already present.




## ⚙️ Run New Experiment
