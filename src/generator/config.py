import os
import re
import torch

HF_GENERATOR_MODEL = os.getenv("GEN_MODEL", "Qwen/Qwen2.5-3B-Instruct")  
### general SLM: Qwen/Qwen2.5-3B-Instruct: 
#                           Size: 6GB
#                           Download time: 2.51 mins 
#                           Retrieve from Checkpoint Shards Time: 6s
#                deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B: 
#                           Size: 3.55GB
#                           Download time: 1.33 mins 
#                           Retrieve from Checkpoint Shards Time: 2s
###
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.mps.is_available() else torch.float32
CITE_RE = re.compile(r"\[?PDPA\s*s\.\s*[0-9A-Za-z]+(?:\(\d+\))*(?:\([a-z]\))?\]?", flags=re.IGNORECASE)
TOPK_RETRIEVER = 10
K_GEN = 3
RANDOM_SEED = 42

TAU_RETR = float(os.getenv("TAU_RETR", 0.0318))  # retrieval confidence threshold, optimized on validation set
TAU_RETR_MARGIN = float(os.getenv("TAU_RETR_MARGIN", 0.0)) # 
TAU_RERANK = float(os.getenv("TAU_RERANK", 1.0))  # reranker confidence threshold, cannot be negative
TAU_RERANK_MARGIN = float(os.getenv("TAU_RERANK_MARGIN", 0.0))

TAU_CITE = float(os.getenv("CITE_COVERAGE", 0.0)) # citation coverage threshold (not applicable in our usecase but kept for completeness)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 350)) # max token for SLM to generate
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
