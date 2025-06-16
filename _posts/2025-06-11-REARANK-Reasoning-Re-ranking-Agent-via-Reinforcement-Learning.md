---
title: "[논문리뷰] REARANK: Reasoning Re-ranking Agent via Reinforcement Learning [WIP]"
last_modified_at: 2025-06-11
categories:
  - 
tags:
  - 
excerpt: "REARANK: Reasoning Re-ranking Agent via Reinforcement Learning"
use_math: true
classes: wide
---

> [arXiv] 2025. [[Paper](https://arxiv.org/abs/2505.20046)] [[Github](링크 없음)]  
> Le Zhang, Bo Wang, Xipeng Qiu, Siva Reddy, Aishwarya Agrawal  
> McGill University, Mila, Fudan University  
> 2025년 5월 26일  




개발 과정에서 흥미로운 논문이 있어 소개해볼까 한다. 바로 RERANK에 관한 연구인데, 이 논문은 추론 능력을 갖춘 에이전트를 학습시키는 데 강화학습(RL) 접근 방식이 필수적임을 보여준다. 특히 강화학습 기반의 listwise reasoning reranking agent를 통해 대규모 언어 모델(LLM)의 추론 능력을 효과적으로 향상시킬 수 있는 방법을 제안하고 있다.


- REARANK의 주요 목표는 정보 검색(Information Retrieval, IR) 시스템의 두 번째 단계인 재랭킹 성능을 개선
-


## Abstract

REARANK는 대규모 언어 모델(LLM) 기반의 listwise reasoning reranking agent입니다. REARANK는 재순위화(reranking) 전에 명시적으로 추론을 수행하여 성능과 해석 가능성을 크게 향상시킵니다. 강화학습과 데이터 증강을 활용하여 REARANK는 인기 있는 정보 검색 벤치마크에서 기준 모델 대비 상당한 개선을 달성했으며, 특히 179개의 주석 샘플만으로도 이러한 성과를 보였습니다. Qwen2.5-7B를 기반으로 구축된 REARANK-7B는 도메인 내 및 도메인 외 벤치마크에서 GPT-4와 비교할 만한 성능을 보여주며, 추론 집약적인 BRIGHT 벤치마크에서는 GPT-4를 능가하기도 합니다. 이러한 결과는 우리 접근법의 효과를 강조하고 강화학습이 재순위화에서 LLM의 추론 능력을 어떻게 향상시킬 수 있는지를 보여줍니다.

## 1. Introduction

정보 검색(Information Retrieval) 시스템에서 재순위화(reranking)는 초기 검색 결과의 순서를 개선하여 사용자에게 더 관련성 높은 결과를 제공하는 중요한 단계입니다. 기존의 재순위화 방법들은 주로 점수 기반 접근법을 사용했지만, 이는 복잡한 추론이 필요한 쿼리에서 한계를 보였습니다.

본 논문에서는 이러한 문제를 해결하기 위해 REARANK를 제안합니다. REARANK의 주요 특징은 다음과 같습니다:

- **명시적 추론**: 재순위화 전에 각 문서의 관련성에 대해 명시적으로 추론
- **강화학습 기반 학습**: 소량의 주석 데이터로도 효과적인 학습 가능
- **해석 가능성**: 추론 과정을 통해 재순위화 결정에 대한 설명 제공

## 2. Methodology

### 2.1 REARANK Architecture

REARANK는 다음과 같은 구조로 구성됩니다:

1. **Reasoning Module**: 쿼리와 문서 간의 관련성을 분석하고 추론
2. **Ranking Module**: 추론 결과를 바탕으로 문서들의 순위 결정
3. **Reinforcement Learning Module**: 성능 피드백을 통한 모델 개선

### 2.2 Training Process

REARANK의 학습 과정은 다음 단계로 진행됩니다:

1. **초기 학습**: 소량의 주석 데이터로 기본 추론 능력 학습
2. **데이터 증강**: 자동 생성된 추론 체인으로 학습 데이터 확장
3. **강화학습**: 재순위화 성능을 보상으로 하는 강화학습 적용

## 3. Experimental Results

### 3.1 Benchmark Performance

REARANK-7B는 다양한 정보 검색 벤치마크에서 우수한 성능을 보였습니다:

- **MS MARCO**: 기존 방법 대비 15% 성능 향상
- **TREC-DL**: GPT-4와 비교할 만한 성능
- **BRIGHT**: GPT-4를 능가하는 성능

### 3.2 Few-shot Learning

특히 주목할 점은 REARANK가 단 179개의 주석 샘플만으로도 뛰어난 성능을 달성했다는 것입니다. 이는 강화학습과 데이터 증강 기법의 효과를 보여줍니다.

## 4. Analysis and Discussion

### 4.1 Reasoning Quality

REARANK가 생성하는 추론 체인을 분석한 결과, 다음과 같은 특징을 발견했습니다:

- 쿼리의 핵심 의도 파악
- 문서 내용의 관련성 분석
- 비교적 추론을 통한 순위 결정

### 4.2 Interpretability

명시적 추론 과정을 통해 REARANK는 기존 블랙박스 모델들과 달리 재순위화 결정에 대한 명확한 설명을 제공할 수 있습니다.

## 5. Conclusion

REARANK는 강화학습을 활용한 추론 기반 재순위화 에이전트로서, 소량의 데이터로도 GPT-4 수준의 성능을 달성했습니다. 특히 추론이 필요한 복잡한 쿼리에서 뛰어난 성능을 보여주며, 해석 가능한 재순위화 시스템의 가능성을 제시했습니다.

향후 연구에서는 더 다양한 도메인에서의 적용과 추론 품질의 추가적인 개선이 기대됩니다.
