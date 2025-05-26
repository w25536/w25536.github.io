---
title: "Transformers: AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE 논문 분석"
last_modified_at: 2025-01-24
categories:
  - 부트캠프
tags:
  - python
  - 파이썬
  - torchtext
  - pytorch
  - 파이토치
  - 전처리
  - data science
  - 데이터 분석
  - 딥러닝
  - 딥러닝 자격증
  - 머신러닝
  - 빅데이터
  - Vision Transformer
  - ViT
  - Transformer
  - 컴퓨터 비전
excerpt: "Vision Transformer (ViT) 논문 분석: AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE 상세 정리"
use_math: true
classes: wide
---

## 논문 개요

**참고 자료**: <https://iy322.tistory.com/66>

본 포스트는 Vision Transformer (ViT)의 핵심 논문 "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"에 대한 상세한 분석을 다룹니다.

## Vision Transformer (ViT) 계산 과정

### 1. 입력 이미지 처리 (Z_0)

#### 1.1 이미지 패치화 (Image Patching)
- 입력 이미지 (H × W × C)를 N개의 패치로 분할합니다[1](https://hyundoil.tistory.com/334)
- 각 패치의 크기는 (P × P × C)입니다[2](https://devocean.sk.com/blog/techBoardDetail.do?ID=166868&boardType=techBlog)
- **예시**: 224×224×3 이미지를 16×16 패치로 나누면 14×14=196개의 패치가 생성됩니다[5](https://velog.io/@leehyuna/Vision-TransformerViT)

#### 1.2 패치 임베딩 (Patch Embedding)
- 각 패치를 1차원 벡터로 평탄화(flatten)합니다[1](https://hyundoil.tistory.com/334)
- 평탄화된 벡터에 선형 변환을 적용하여 D 차원의 임베딩 벡터로 변환합니다[2](https://devocean.sk.com/blog/techBoardDetail.do?ID=166868&boardType=techBlog)
- **예시**: 16×16×3=768 차원의 벡터를 D 차원으로 변환합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT)

#### 1.3 위치 임베딩 추가 (Positional Encoding)
- 학습 가능한 1D 위치 임베딩을 각 패치 임베딩에 더합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT)
- 위치 임베딩의 크기는 (N+1) × D입니다 (클래스 토큰 포함)[5](https://velog.io/@leehyuna/Vision-TransformerViT)
- 실제로 16×16으로 나누어 flatten시킨 후 임베딩 벡터가 되고, 정수로 positional encoding을 진행합니다

#### 1.4 클래스 토큰 추가 (Class Token)
- 학습 가능한 클래스 토큰을 패치 임베딩 시퀀스의 맨 앞에 추가합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT)
- **클래스 토큰이 사용되는 이유**: 분류 task이므로 정답 정보를 함께 제공하기 위함입니다
- 0번째 위치에는 클래스 토큰이 배치됩니다

### 2. Transformer 인코더 처리

#### 2.1 레이어 정규화 (Layer Normalization)
- 입력 시퀀스에 레이어 정규화를 적용합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT)

#### 2.2 멀티헤드 셀프 어텐션 (Multi-Head Self-Attention, MSA)
- 정규화된 입력을 Query, Key, Value로 변환합니다[2](https://devocean.sk.com/blog/techBoardDetail.do?ID=166868&boardType=techBlog)
- 각 헤드에서 어텐션 스코어를 계산: `A = softmax(QK^T / √d)`[2](https://devocean.sk.com/blog/techBoardDetail.do?ID=166868&boardType=techBlog)
- 어텐션 스코어와 Value를 곱하여 컨텍스트 벡터를 생성합니다[2](https://devocean.sk.com/blog/techBoardDetail.do?ID=166868&boardType=techBlog)
- 모든 헤드의 결과를 연결하고 선형 변환을 적용합니다[2](https://devocean.sk.com/blog/techBoardDetail.do?ID=166868&boardType=techBlog)

#### 2.3 잔차 연결 (Residual Connection)
- MSA의 출력을 입력과 더합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT)

#### 2.4 레이어 정규화
- 잔차 연결의 결과에 다시 레이어 정규화를 적용합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT)

#### 2.5 MLP (Multi-Layer Perceptron)
- 정규화된 결과를 MLP에 통과시킵니다[5](https://velog.io/@leehyuna/Vision-TransformerViT)
- MLP는 일반적으로 두 개의 선형 레이어와 GELU 활성화 함수로 구성됩니다

#### 2.6 잔차 연결
- MLP의 출력을 이전 단계의 입력과 더합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT)

#### 2.7 반복 처리
- 위의 과정(2.1-2.6)을 L번 반복합니다 (L은 Transformer 블록의 수)[5](https://velog.io/@leehyuna/Vision-TransformerViT)

### 3. 분류 헤드 (Classification Head)

#### 3.1 최종 레이어 정규화
- 마지막 Transformer 블록의 출력을 정규화합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT)

#### 3.2 클래스 토큰 추출
- 정규화된 출력에서 클래스 토큰에 해당하는 벡터를 추출합니다[5](https://velog.io/@leehyuna/Vision-TransformerViT)
- **중요**: 분류를 위해서는 오직 클래스 토큰만 사용됩니다

#### 3.3 최종 분류
- 추출된 클래스 토큰 벡터를 MLP 분류기(classification head)에 통과시켜 최종 분류 결과를 얻습니다[5](https://velog.io/@leehyuna/Vision-TransformerViT)

## 핵심 개념 정리

### Downstream Task란?
- 사전 훈련된 모델을 특정 작업에 맞게 미세 조정(fine-tuning)하는 과정을 의미합니다
- ViT의 경우 ImageNet 등에서 사전 훈련 후, 특정 분류 작업에 적용하는 것이 downstream task입니다

### 클래스 토큰의 학습 방식
- 클래스 토큰은 64차원(또는 설정에 따라 다른 차원)으로 학습됩니다
- ViT에서는 이 클래스 정보가 self-attention 메커니즘을 통해 모든 패치와 상호작용하며 학습됩니다
- 최종 MLP 분류기에서는 클래스 토큰의 표현만을 사용하여 예측을 수행합니다

### BERT 모델과의 연관성
- ViT는 BERT의 구조를 이미지 도메인에 적용한 것으로 볼 수 있습니다
- 클래스 토큰의 개념 또한 BERT의 [CLS] 토큰에서 영감을 받았습니다

## 결론

ViT는 이미지를 패치 단위로 처리하고, Transformer 구조를 사용하여 전역적인 관계를 학습하며, 최종적으로 이미지 분류를 수행하는 혁신적인 모델입니다. 기존의 CNN 기반 모델과 달리 순수하게 attention 메커니즘만을 사용하여 뛰어난 성능을 달성했다는 점에서 컴퓨터 비전 분야에 큰 영향을 미쳤습니다.
