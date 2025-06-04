---
title: "[논문리뷰] EfficientLLM: Efficiency in Large Language Models"
last_modified_at: 2025-05-27
categories:
  - 
tags:
  - 
excerpt: "EfficientLLM: Efficiency in Large Language Models"
use_math: true
classes: wide
---


> arXiv 2025. [[Paper](https://arxiv.org/abs/2505.13840)]  
> EfficientLLM Team  
> 28 May 2025  

![Refer to caption](https://arxiv.org/html/2505.13840v1/x2.png)

## Metrics
![[CleanShot 2025-05-27 at 13.53.14@2x.png | 300]]


**논문 정리**

## 1. 개요
이 논문은 "LLM을 어떻게 하면 빠르고 효율적으로 만들 수 있을까?"라는 질문에 답하기 위해 작성되었습니다. 대규모 언어 모델의 효율성을 높이기 위한 다양한 기법들을 체계적으로 분석하고 평가합니다.

## 2. 아키텍처 및 사전 학습 최적화

### 2.1 효율적인 어텐션 메커니즘
- **MQA (Multi-Query Attention)**: 제한된 디바이스에서 최적의 메모리-지연시간 성능 (모바일 기기에 최적화)
- **MLA (Multihead-Latent Attention)**: 품질이 중요한 작업에서 가장 낮은 perplexity (성능 중심)
- **GQA (Grouped-Query Attention)**: 균형 잡힌 성능 제공
- **NSA**: 토큰당 에너지 소비가 가장 적음 (에너지 효율성 중심)

### 2.2 Mixture-of-Experts (MoE)
- **장점**: 계산 속도 향상 및 정확성 증대
- **단점**: VRAM 사용량 40% 증가

## 3. 양자화 (Quantization)
- **INT4 양자화**: 메모리/에너지 효율성 3.9배 향상
- **성능 저하**: 3-5%의 정확도 감소

## 4. Parameter-Efficient Fine-Tuning (PEFT) 방법들

### 4.1 주요 PEFT 기법
- **LoRA (Low-Rank Adaptation)**
- **RSLoRA**: 14B 이상 모델에서만 LoRA보다 우수
- **DoRA**
- **LoHa (Low-Rank Hadamard Product)**
- **LoKR (Low-Rank Kronecker Product)**
- **GLoRA (Generalized LoRA)**

### 4.2 성능 비교
- **1-3B 모델**: LoRA-plus가 최고 성능 (DoRA, LoRA도 우수)
- **14B+ 모델**: RSLoRA가 LoRA보다 우수

## 5. Mamba 아키텍처
- Mamba + 양자화 조합 시 언어 성능이 트랜스포머보다 떨어짐

## 6. 권장사항

### 6.1 용도별 최적 구성
- **최고 성능**: MLA attention을 사용한 표준 LLM
- **모바일 환경**: MQA or Mamba + INT4 양자화

### 6.2 적용 범위
- 텍스트 생성 모델
- Vision-Language Model (VLM)
- 기타 다양한 모델 아키텍처

## 7. 결론
성능, 속도, 메모리, 에너지 효율성 간의 최적 균형점을 찾는 것이 핵심입니다. 각 사용 사례에 따라 적절한 기법을 선택하여 효율적인 LLM을 구축할 수 있습니다.
