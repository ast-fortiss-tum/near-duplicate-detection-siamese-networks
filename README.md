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

The `environment.yml` already pins compatible versions.

```bash
conda env create --name snn-ndd -f environment.yml
```

*What this does*

1. Creates a new Conda env called **`snn-ndd`**.
2. Installs all Conda‑managed packages.

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

        * `bert` (adjust the variant inside the script for ModernBERT)
        * `doc2vec`
        * `markuplm`

   4. **Outputs**

      * Experiment results → `results/rq1/`
      * Trained models → `models/`
      * Cached embeddings → `embedding/`

> **Re-runs** will automatically reuse any existing models or embeddings; nothing is re-trained if already present.


#### b) FragGen Evaluation

   Run the evaluation from the project base directory

   ```
   python scripts/rq1/baseline-classification/within_app_fraggen.py
   ```


#### c) Other Baseline Methods - (WEBEMBED, RTED, PDIFF) Evaluation

   Run the evaluation from the project base directory

   ```
   python scripts/rq1/baseline-classification/<evaluation_setting>_app_baseline.py
   ```

   * `<evaluation_setting>`:

     * `within`
     * `across`

### 2) RQ2: Model Quality

#### a) SNN Evaluation

Run below command from the project base directory. Results will be saved in `results/rq2/`

   ```
   python scripts/rq2/model-quality-snn.py
   ```

#### b) FRAGGEN Evaluation

Run below command from the project base directory. Results will be saved in `results/rq2/`

   ```
   python scripts/rq2/fraggen.py
   ```

#### c) WEBEMBED Evaluation

Run below commands from the project base directory one after the other. The first scripts generates intermediate results files in `resources/csv_results_table/`. The second script generates the RQ2 results and save in `results/rq2/`

   ```
   python scripts/rq2/a-webembed.py
   python scripts/rq2/b-webembed.py
   ```

#### d) Other Baseline (RTED and PDIFF) Evaluation

Run below commands from the project base directory one after the other. The first scripts generates intermediate results files in `resources/csv_results_table/`. The second script generates the RQ2 results and save in `results/rq2/`
   ```
   python scripts/rq2/a-other-baselines.py
   python scripts/rq2/b-other-baselines.py
   ```


## ⚙️ Run New Experiment
