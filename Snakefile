# Snakefile

configfile: "config.yaml"

# 1. Final target
rule all:
    input:
        "results/ffnn_node2vec.pth"

# 2. Download TFLink
rule download_tf_link_db:
    output:
        "inputs/TFLink_Homo_sapiens_interactions_All_simpleFormat_v1.0.tsv.gz"
    shell:
        """
        mkdir -p inputs
        wget -O {output} {config[tf_link_db]}
        """

# 3. Download KG2 nodes
rule download_kg2_nodes:
    output:
        "inputs/kg2_nodes.jsonl.gz"
    shell:
        """
        mkdir -p inputs
        wget -O {output} {config[kg2][nodes]}
        """

# 4. Download KG2 edges
rule download_kg2_edges:
    output:
        "inputs/kg2_edges.jsonl.gz"
    shell:
        """
        mkdir -p inputs
        wget -O {output} {config[kg2][edges]}
        """

# 5. Download KG2 node synonymizer
rule download_node_synonymizer:
    output:
        "inputs/node_synonymizer.sqlite"
    shell:
        """
        mkdir -p inputs
        wget -O {output} {config[kg2][synonymizer]}
        """
# 6. Generate Training Data        
rule generate_training_data:
    input:
        tf_link_db = "inputs/TFLink_Homo_sapiens_interactions_All_simpleFormat_v1.0.tsv.gz"
    output:
        "results/training_data.tsv"
    log:
        "logs/generate_training_data.log"
    shell:
        """
        mkdir -p results logs
        python scripts/generate_training_data.py {input.tf_link_db} {output} &> {log}
        """
rule canonicalize_training_data:
    input:
        input_file="results/training_data.tsv",
        sqlite_db="inputs/node_synonymizer.sqlite"
    output:
        output_file="results/canonicalized_training_data.tsv"
    shell:
        """
        python scripts/canonicalize_training_data.py {input.input_file} {input.sqlite_db} {output.output_file}
        """

rule train_model:
    input:
        training_data="results/canonicalized_training_data.tsv",
        kg2_nodes="inputs/kg2_nodes.jsonl.gz",
        kg2_edges="inputs/kg2_edges.jsonl.gz"
    output:
        node_ids="results/node_ids.npy",
        node_embeddings="results/node2vec_embeddings.npy",
        graph_pickle="results/graph.pkl",
        trained_model="results/ffnn_node2vec.pth",
        node2vec_model="results/node2vec.zip"
    shell:
        """
        python scripts/ffnn_node2vec.py {input.training_data} {input.kg2_nodes} {input.kg2_edges} results/
        """


