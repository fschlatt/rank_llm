#!/usr/bin/env bash

tira-cli download --dataset $1 --approach reneuir-2024/reneuir-baselines/plaid-x-retrieval

/prepare-rerank-file-from-plaid-x.py --output /tmp/rerank.jsonl.gz --input-dataset $1

python3 /workspace/rank_llm/scripts/run_rank_llm.py \
	--model_path=castorini/rank_zephyr_7b_v1_full \
	--top_k_candidates=100 \
	--dataset=tmp/rerank.jsonl.gz \
	--prompt_mode=rank_GPT \
	--context_size=4096 \
	--variable_passages \
	--retrieval_method=unspecified \
	--output_dir=$2

