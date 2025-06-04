---
title: "[논문리뷰] s3: You Don't Need That Much Data to Train a Search Agent via RL"
last_modified_at: 2025-06-04
categories:
  - 
tags:
  - 
excerpt: "s3: You Don't Need That Much Data to Train a Search Agent via RL"
use_math: true
classes: wide
---


> arXiv 2025. [[Paper](https://arxiv.org/abs/2505.14146)]  
> Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng Sun, Jiawei Han  
> University of Illinois Urbana-Champaign, Emory University  
> 20 May 2025  

**s3: You Don't Need That Much Data to Train a Search Agent via RL**

Retrieval-augmented generation (RAG) systems empower large language models (LLMs) to access external knowledge during inference. Recent advances have enabled LLMs to act as search agents via reinforcement learning (RL), improving information acquisition through multi-turn interactions with retrieval engines. However, existing approaches either optimize retrieval using search-only metrics (e.g., NDCG) that ignore downstream utility or fine-tune the entire LLM to jointly reason and retrieve—entangling retrieval with generation and limiting the real search utility and compatibility with frozen or proprietary models. In this work, we propose **s3**, a lightweight, model-agnostic framework that decouples the searcher from the generator and trains the searcher using a Gain Beyond RAG reward: the improvement in generation accuracy over naive RAG. s3 requires only 2.4k training samples to outperform baselines trained on over 70x more data, consistently delivering stronger downstream performance across six general QA and five medical QA benchmarks.

Subjects: Artificial Intelligence (cs.AI); Computation and Language (cs.CL)  
Cite as: arXiv:2505.14146 [cs.AI]  
DOI: [10.48550/arXiv.2505.14146](https://doi.org/10.48550/arXiv.2505.14146)

