---
title: "[논문리뷰] U-Net: Convolutional Networks for Biomedical"
last_modified_at: 2025-01-20
categories:
  - 부트캠프
tags:
  - python
  - 파이썬
  - pytorch
  - 파이토치
  - 데이터
  - 분석
  - 딥러닝
  - 딥러닝
  - 자격증
  - 머신러닝
  - 빅데이터
excerpt: "U-Net: Convolutional Networks for Biomedical"
use_math: true
classes: wide
---
오늘의 논문 분석 


U-Net은 Biomedical 분야에서 이미지 분할(Image Segmentation)을 목적으로 제안된 End-to-End 방식의 Fully-Convolutional Network 기반 모델이다.

네트워크 구성의 형태(‘U’)로 인해 U-Net이라는 이름이 붙여졌다.


![](https://miro.medium.com/v2/resize:fit:700/1*qNdglJ1ORP3Gq77MmBLhHQ.png)





![](https://blog.kakaocdn.net/dn/eqF2ws/btr7hStTV9e/v7iDYX2pH94abwFIgsSmpk/img.png)



U-Net은 쉽게말해 FCN(Fully Convolutional Network)와 Up convolution 층들의 집합이다. 용도는 보통 Segmentation에 많이 쓴다. Segmentation은 뭘까?




![](https://blog.kakaocdn.net/dn/b2PMcQ/btr91zMOZsy/utTa9EY4YRllajvbpDltt0/img.png)



