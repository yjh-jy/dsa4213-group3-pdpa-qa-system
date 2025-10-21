# dsa4263-group3-pdpa-qa-system

## PDPA-Grounded Question Answering via Citation-Constrained RAG with Abstention

**Authors:** Group 3 (Ashley Toh Ke Wei, Sybella Chloe Enriquez Tan, Choy Qi Hui, Yoong Jun Han)

## Introduction

Privacy policy documents are notoriously long, legally dense, and difficult for the general public to understand (Tan, 2022). As a result, users struggle to extract precise answers, which undermines transparency and informed consent.

This project aims to apply Natural Language Processing (NLP) techniques to make privacy policies more accessible by enabling users to query these documents in natural language. Specifically, we focus on the Singapore Personal Data Protection Act (PDPA) (PDPC, 2023), the principal legislation governing the collection, use, and disclosure of personal data by private sector organisations in Singapore. This focus aligns with strong public concern: in a 2016 KPMG survey, 32% of Singaporeans reported being “extremely concerned” about companies’ use of their personal data, with even more expressing broader worry. 

Therefore, we propose a retrieval-augmented QA system that answers PDPA-related questions in plain language with exact section citations and abstains when evidence is insufficient.

## Datasets

- Singapore PDPA Document: Official statute text chunked by Section (with Part/Division/Section IDs, titles, character offsets). This will be used as the primary knowledge base for the RAG system and source to create PDPA QAs.
- PDPA QAs (synthesized): approx. 800 LLM-generated QAs based on the Singapore PDPA document, using PolicyQA as  a seed for question styles. Human validation will be used for quality control (manual filtering, consistency checks). In addition, a golden set of 30 manually crafted QAs will be used as guiding examples for synthesis.
- PolicyQA Dataset: 714 questions across 115 website privacy policies. This will provide a secondary benchmark QA format for synthesis.

## Methods/Models

### Benchmark Model: Prompt-only LLM

A general GPT-type model (e.g., FLAN-T5-base) grounded with the PDPA document context and is prompted to answer only from provided text, or output “No sufficient basis.”

### Proposed System: Retrieval Augmented Generation (RAG) with Abstention 

- Retrieval: Choose among BM25 (lexical), Dense (E5/MiniLM-style embeddings; cosine similarity), Hybrid (e.g., reciprocal rank fusion or fixed α·BM25 + (1–α)·Dense)
- Reranking: Choose among Bi-encoder (assuming lexical/hybrid retrieval), Cross-encoder, or none
- Generation with citation constraints: Generative encoder-decoder model (e.g., FLAN-T5, BART, LLaMA-2, Mistral-7B)  with citation-constrained decoding that must output section IDs used for grounding
- Plain-language prompting: generative models are prompted to rephrase retrieved PDPA content into simple, non-legal terms 
- Calibration & abstention: Use confidence/entropy thresholds (and/or max rerank score) to decide between answer vs “No sufficient basis.”

## Ablation Studies

As mentioned, we will do a series of ablation studies to build our system

- Encoder-decoder model impact: Comparing different underlying encoder-decoder model used (not for reranking)
- Retriever architecture impact: BM25-only vs embedding-only vs hybrid retriever, to quantify gains from combining lexical and semantic signals.
- Reranker contribution and architecture: Compare None, Bi-encoder (assuming lexical/hybrid retrieval), and Cross-encoder

## Evaluation

### Quantitative Metrics

- Retrieval: Recall, MRR, NDCG, latency
- Reranking: Precision, NDCG, HitRate, latency
- Answering: EM, F1; Support-F1 (token F1 on cited gold span)
- Generative Quality: ROUGE-L for overlap with ground-truth answers. (Synthesized PDPA QAs)
- Citations & Faithfulness: Citation hit-rate, hallucination rate, coverage-accuracy curves, Expected Calibration Error (ECE)

### Qualitative Metrics (Human raters)

#### RAG system in isolation

We will construct a set of 30 challenging PDPA test questions (distinct from the 30 gold examples used to guide synthesis) and collect double-blind human ratings of the RAG system’s answers on correctness, clarity (plain-language quality), and faithfulness to cited sections using a 5-point scale. We will report inter-rater agreement (e.g., Cohen’s κ or Krippendorff’s α) and provide a concise error taxonomy covering retrieval miss, span mismatch, multi-section reasoning errors, and ambiguity.

#### Additional Qualitative Comparison (RAG vs. benchmark)

Using the same 30-question set, we will run a separate double-blind, side-by-side comparison in which raters view anonymized pairs of answers produced by the RAG system and the benchmark model. For each question, raters indicate which answer is superior on correctness, clarity, and citation faithfulness, and provide brief justifications, to assess whether RAG’s quantitative gains translate into perceptible improvements for end users.

## References

- Ahmad, et al. (Nov, 2020). PolicyQA: A Reading Comprehension Dataset for Privacy Policies. Retrieved from Association for Computational Linguistics Anthology: https://doi.org/10.18653/v1/2020.findings-emnlp.66 
- KPMG. (7 Nov, 2016). Crossing the line: Staying on the right side of consumer privacy [Report]. KPMG Corporate Office. https://assets.kpmg.com/content/dam/kpmg/sg/pdf/2016/11/KPMG-Cyber-Security-Privacy-Report.pdf
- PDPC (3 Nov, 2023). PDPA Overview. Retrieved from PDPC: https://www.pdpc.gov.sg/overview-of-pdpa/the-legislation/personal-data-protection-act
- Singapore Statutes Online. (2012). Personal Data Protection Act 2012. Retrieved from Singapore Statutes Online: https://sso.agc.gov.sg/Act/PDPA2012
- Tan, B. (6 Oct, 2022). Commentary: If data privacy is so important, why do we click 'agree' on user agreements without reading? Retrieved from CNA: https://www.channelnewsasia.com/commentary/privacy-agreement-data-breach-policy-too-long-2987651
