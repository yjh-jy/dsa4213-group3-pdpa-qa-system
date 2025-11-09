# artefacts/

This subdirectory holds the generated artifacts used by the PDPA QA system. The repository currently contains these subdirectories:

- bm25_index/  
  - Artifacts for the lexical (BM25) retriever.

- dense_retriever/  
  - Artifacts for the dense retriever (vector indexes and model files).

- cross_encoder/  
  - Artifacts for the cross-encoder used for re-ranking.

- ltr_reranker/  
  - Artifacts for the learning-to-rank reranker.


## Important note about large model files (dense retriever)
- The dense retriever includes a model file named model.safetensors that is too large to commit directly to the repository.
- To keep the repository manageable, the file was split into 90 MB chunks before adding to storage. Use the following commands to reassemble the file:

```bash
# navigate
cd artefacts/dense_retriever/model

# reassemble
cat model.safetensors.part.* > model.safetensors
```
