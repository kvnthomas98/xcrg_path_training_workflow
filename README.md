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

> 💡 Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed before proceeding.

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
