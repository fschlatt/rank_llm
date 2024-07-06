#docker build -t plaid-x-rank-zephyr -f re-rank-plaid-x/Dockerfile  . 
FROM fschlatt/rank_llm:0.0.1

ADD re-rank-plaid-x/rank-llm-rerank-plaid-x.sh re-rank-plaid-x/prepare-rerank-file-from-plaid-x.py /

ENTRYPOINT [ "/rank-llm-rerank-plaid-x.sh", "$inputDataset", "$outputDir" ]
