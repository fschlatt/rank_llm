# docker build -t fschlatt/rank_llm:0.0.1 -f docker/Dockerfile .
FROM huggingface/transformers-pytorch-gpu:4.35.2

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y git wget curl build-essential htop openjdk-21-jdk

RUN pip install \
    ftfy \
    fastchat \
    einops \
    accelerate \
    pyserini \
    python-dotenv \
    openai \
    pip install tiktoken \
    && pip cache purge

COPY ./src/rank_llm /workspace/rank_llm

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

ENTRYPOINT [ "python3", \
    "/workspace/rank_llm/scripts/run_rank_llm.py", \
    "--model_path=castorini/rank_zephyr_7b_v1_full", \
    "--top_k_candidates=100", \
    "--dataset=$inputDataset", \
    "--prompt_mode=rank_GPT", \
    "--context_size=4096", \
    "--variable_passages", \
    "--retrieval_method=unspecified", \
    "--output_dir=$outputDir" \
    ]
