# XCRG Path Training Workflow

This repository contains a Snakemake-based workflow for training models as part of the XCRG path pipeline. It also includes a Conda environment specification for reproducible setup.

---

## Environment Setup

To ensure all dependencies are correctly installed, set up the Conda environment using the provided YAML file.

### Step 1: Create the Environment

```bash
conda env create -f envs/xcrg_path_training.yml
```

### Step 2: Activate the Environment

```bash
conda activate xcrg_path_training
```

>  Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed before proceeding.

---
### Step 3: Update the KG2 Version


Modify KG2 version to the version you want to use [in this section](https://github.com/kvnthomas98/xcrg_path_training_workflow/blob/main/config.yaml#L2)

---
## Running the Workflow

Once the environment is activated, you can execute the Snakemake workflow.

### Basic Run

```bash
snakemake --cores <number_of_cores>
```

Replace `<number_of_cores>` with the number of CPU cores you'd like to use. For example:

```bash
snakemake --cores 1
```

### Dry Run (Preview)

To preview the steps without executing them:

```bash
snakemake -n
```

---

### Final Output

In the `results/` folder all the necessary files for utilizing the model will be present.

`results/node_ids.npy` contains node indices mapping.

`results/node2vec_embeddings.npy` contains node2vec embedding.

`results/graph.pkl` provides the KG2 graph in a networkx form.

`results/ffnn_node2vec.pth` provides weights for the models.

---

