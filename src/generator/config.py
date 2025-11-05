import os
import re
import torch

### general SLMs: Qwen/Qwen2.5-3B-Instruct: 
#                           Size: 6GB
#                           Download time: 2.51 mins 
#                           Retrieve from Checkpoint Shards Time: 6s
#                Qwen3-4B: 
#                           Size: 9GB
#                           Download time: 3.27mins
#                           Retrieve from Checkpoint Shards Time: 9s
#                deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B: 
#                           Size: 3.55GB
#                           Download time: 1.33 mins 
#                           Retrieve from Checkpoint Shards Time: 2s
#

HF_GENERATOR_MODEL = os.getenv("GEN_MODEL", "Qwen/Qwen3-4B")   

# General SLM related knobs
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.mps.is_available() else torch.float32

# Qwen 2.5 related sampling parameters 
TEMPERATURE_QWEN2_5 = 0.2
MAX_NEW_TOKENS_QWEN2_5 = 350 # max token for SLM to generate

# Qwen 3 related sampling parameters
ENABLE_REASONING = True

MAX_NEW_TOKENS_QWEN3_THINKING = 500 # higher than non thinking to accomodate for the thinking budget 
TEMPERATURE_QWEN3_THINKING = 0.6
TOP_P_QWEN3_THINKING = 0.95
TOP_K_QWEN3_THINKING = 20

MAX_NEW_TOKENS_QWEN3_NON_THINKING = 350
TEMPERATURE_QWEN3_NON_THINKING = 0.2
TOP_P_QWEN3_NON_THINKING = 0.8
TOP_K_QWEN3_NON_THINKING = 20

# Retriever/Reranking related knobs
CITE_RE = re.compile(r"\[?PDPA\s*s\.\s*[0-9A-Za-z]+(?:\(\d+\))*(?:\([a-z]\))?\]?", flags=re.IGNORECASE)
TOPK_RETRIEVER = 10
K_GEN = 3
RANDOM_SEED = 42

# Abstention related knobs
TAU_RETR = float(os.getenv("TAU_RETR", 0.031))  # retrieval confidence threshold, optimized on validation set
TAU_RETR_MARGIN = float(os.getenv("TAU_RETR_MARGIN", 0.0)) # 
TAU_RERANK = float(os.getenv("TAU_RERANK", 1.0))  # reranker confidence threshold, cannot be negative
TAU_RERANK_MARGIN = float(os.getenv("TAU_RERANK_MARGIN", 0.0))
TAU_CITE = float(os.getenv("CITE_COVERAGE", 0.0)) # citation coverage threshold (not applicable in our usecase but kept for completeness)

