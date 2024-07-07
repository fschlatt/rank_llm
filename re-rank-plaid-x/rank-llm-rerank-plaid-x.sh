#!/usr/bin/env bash

INPUT_RUN=$(tira-cli download --dataset $1 --approach reneuir-2024/reneuir-baselines/plaid-x-retrieval)

/prepare-rerank-file-from-plaid-x.py --output /tmp/ --input-dataset $1 --input-run ${INPUT_RUN}/run.txt --top-k 100 

zcat /tmp/rerank.jsonl.gz |head -10

python3 /workspace/rank_llm/scripts/run_rank_llm.py \
	--model_path=castorini/rank_zephyr_7b_v1_full \
	--top_k_candidates=100 \
	--dataset=/tmp/rerank.jsonl.gz \
	--prompt_mode=rank_GPT \
	--context_size=4096 \
	--variable_passages \
	--retrieval_method=unspecified \
	--output_dir=$2

