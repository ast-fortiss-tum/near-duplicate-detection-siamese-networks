# Siamese Nueral Network (SSN) For Near Duplicate Detection in Web App Model Inference

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

### 3) RQ3: Code Coverage

To integrate the state abstraction function during crawling, you want to follow the implementation of the state vertex and adjust the endpoint for your Flask app. When the crawler needs to evaluate  two states are, it can simply make a POST request to the endpoint with the necessary data, which allows the Flask app to function as the State Abstraction Function (SAF) and return the results you're looking for. 
Run below command to up flask app. 
   ```
   python scripts/rq3/saf-snn.py # SNN methods
   python scripts/rq3/saf-other-baseline.py # WEBEMBED methods
   python scripts/rq3/saf-other-baseline.py # RTED and PDIFF
   ```
Use this  [example](https://github.com/crawljax/crawljax/blob/master/examples/src/main/java/com/crawljax/examples/FragCrawlExample.java) for directly evaluating the FRAGGEN method without external SAF.

Code coverage varies depending on the application type. For PHP applications, server-side code coverage is measured, while JavaScript applications focus on client-side coverage. For crawling, we used Crawljax (5.2.4-SNAPSHOT). But it has compatibility considerations with certain versions of the Chrome driver. Alternative browsers can be used (Firefox for example) if specified in the crawl configuration. For evaluating FraGen is only possible with chrome hence we had to downgrade Chrome ( 114.0.5735.90)

---
In PHP applications, code coverage measurement is conducted using the Xdebug 2.2.42 extension alongside the php-code-coverage 2.2.33 library. Both crawling and code coverage measurement occur simultaneously. It is important to start code coverage before starting the crawl and to stop it after crawling completion to generate the coverage report. 

Refer [PHP Code Coverage README](scripts/rq3/PHP-Based-Coverage-README.md) for more details.

---
For JavaScript applications, the process operates separately. Crawljax was executed for Dante, which generates corresponding Selenium test cases. Once crawling is complete, Dante can utilize the generated test case files to produce JUnit test cases. For measuring coverage in JavaScript applications, cdp4j 3.0.81, a Java implementation of Chrome DevTools, was used.

Refer [JavaScript Code Coverage README](scripts/rq3/Javascript-Based-Coverage-README.md) for more details.

## ⚙️ Run New Experiment
