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



![review-image](https://moonlight-paper-snapshot.s3.ap-northeast-2.amazonaws.com/arxiv/s3-you-dont-need-that-much-data-to-train-a-search-agent-via-rl-0.png)

## 서론 

- Classic RAG / 고정된 검색 방법 사용 / 검색 품질과 생성 성능이 분리됨
- Pre-RL-Zero / LLM이 추론 중 적극적 참여 (Active RAG) / 제로샷 프롬프트에 의존, 학습 가능한 컴포넌트 부족

위에 방법들이 있고 여기서 다뤄질 내용은 아래와 같다

- RL-Zero 강화 학습(RL) 도입은 에이전트 기반 메서드를 가능하게 했다.

- 이전 접근 방식은 검색 전용 지표(예: NDCG)를 사용하여 검색을 최적화했으며, 이는 하류 유틸리티를 무시했다.

- 다른 방법들은 LLM 전체를 미세 조정하여 추론과 검색을 공동으로 수행하게 함으로써 검색과 생성을 얽히게 했다.

- 이는 실제 검색 유틸리티와 고정되거나 독점적인 모델과의 호환성을 제한했다.

- 검색 결과를 사용한 추론과 생성을 공동으로 훈련하는 접근 방식(예: Search-R1)은 검색과 생성의 긴밀한 얽힘으로 인해 진정한 검색 개선을 분리하기 어렵게 만들었다.

- 또한, Exact Match(EM)와 같은 보상 신호는 의미론적으로 올바르게 표현된 답변에 보상하지 못하는 취약점을 가지고 있다

- GBR은 다음 공식으로 계산된다: GBR(Q) = Acc(G(Q, Ds3), A) - Acc(G(Q, DRAG), A) 


## s3 프레임워크의 성능 평가 및 결과 

- 평가는 생성 정확도(Generation Accuracy)를 주 측정 지표로 사용하며, 기존의 Exact Match보다 의미 정확도를 더 잘 포착한다

- 의료 도메인 QA 성능에서, s3는 결합된 Wikipedia+PubMed+Textbook 코퍼스를 사용했을 때 76.6%의 가장 높은 평균 정확도를 달성하며 모든 검색 증강 기준선을 능가했다

- s3의 의료 QA에서의 제로샷 성공은 일반 QA 데이터에서만 훈련되었음에도 불구하고, 강화 학습으로 학습된 검색 스킬이 생성 튜닝 접근 방식보다 더 신뢰할 수 있게 일반화됨을 시사한다

- 결론적으로, 검색을 생성과 분리하고 검색기만 최적화함으로써 s3는 효율성과 일반화 모두에서 상당한 이득을 얻으며, RAG 시스템 개선을 위한 확장 가능한 경로를 제공한다
 


## s3 프레임워크의 한계 및 향후 연구 방향

- GenAcc와 같은 생성 기반 보상의 사용은 보상 신호를 계산하기 위해 훈련 중에 LLM 추론이 필요하다

- 생성 기반 보상(GenAcc 등)은 훈련 중 LLM 추론이 필요해 계산 비용이 더 든다.