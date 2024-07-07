#docker build -t plaid-x-rank-zephyr -f re-rank-plaid-x/Dockerfile  . 
FROM fschlatt/rank_llm:0.0.1

RUN pip3 install --no-deps ir-datasets
ADD src/rank_llm/scripts/run_rank_llm.py /workspace/rank_llm/scripts/run_rank_llm.py

ADD re-rank-plaid-x/rank-llm-rerank-plaid-x.sh re-rank-plaid-x/rank-llm-rerank-fusion-t5.sh re-rank-plaid-x/prepare-rerank-file-from-plaid-x.py /

ENTRYPOINT [ "/rank-llm-rerank-fusion-t5.sh", "$inputDataset", "$outputDir" ]
