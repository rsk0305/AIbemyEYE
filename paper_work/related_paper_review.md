# 관련 논문 서치 및 검토 결과

작성일: 2026-06-28.
대상 초안: `인코더-디코더 구조를 이용한 유도탄 데이터 점검 방법`.
검토 목적: 초안의 선행연구, 방법론 근거, 국내 국방 적용 사례, Reference 후보를 보강한다.

## 1. 검색 범위와 판단 기준

검색 범위는 다음 네 묶음으로 설정했다.

- Enc-Dec, Autoencoder, VAE 기반 시계열 이상탐지.
- 우주기·항공우주·산업제어 텔레메트리 이상탐지.
- Transformer 기반 최신 다변량 시계열 이상탐지.
- KIMST와 국내 국방 분야 데이터 점검·이상탐지 사례.

판단 기준은 초안과의 직접 관련성, 데이터 형태 유사성, 국방·항공우주 적용성, 초안의 어느 절에 들어갈 수 있는지, 실제 Reference 채택 우선순위이다.

## 2. 채택 우선순위 요약

| 우선순위 | 논문 | 채택 판단 | 초안 반영 위치 |
|---|---|---|---|
| 1 | Malhotra et al., LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection | 초안의 Enc-Dec 핵심 근거로 가장 직접적이다. | 서론, 3.2 모델 구조 |
| 1 | Hundman et al., Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding | 유도탄 데이터와 유사한 항공우주 텔레메트리 이상탐지 근거다. | 서론, 4.1 정상 기준선 학습 |
| 1 | Kim et al., AI-Based DAS Data Analysis for Real-Time Anomaly Detection in Missile FCS | KIMST 내 미사일 FCS 데이터와 실시간 이상탐지 사례라 국내 관련성이 매우 높다. | 서론, 4.2 점검 실행 |
| 2 | Zhang et al., MSCRED | 채널 간 관계와 다변량 진단 관점 보강에 유용하다. | 3.2 모델 구조, 6. 고찰 |
| 2 | Xu et al., Anomaly Transformer | 최신 Transformer 계열 비교 대상으로 유용하지만, 초안의 단순 구조와는 거리가 있다. | 6. 고찰 |
| 2 | Tuli et al., TranAD | Transformer 기반 진단·설명성 후보로 유용하다. | 6. 고찰 |
| 2 | Ren et al., SR-CNN | 운영형 이상탐지 서비스 사례로 실시간 점검 논리 보강에 좋다. | 4.2 점검 실행 |
| 2 | Kim et al., Weapon-system PHM using GAN and synthetic data | 국방 PHM 데이터 부족 문제를 설명하는 국내 사례로 적합하다. | 2.2 문제 정의, 5. 적용 예시 |
| 3 | Min and Jin, Depot test-equipment DB analysis SW | 모델 논문은 아니지만 시험결과 DB화와 점검 체계 필요성을 뒷받침한다. | 서론, 5. 기대 효과 |

## 3. 논문별 검토

### 3.1 Malhotra et al., LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection

- 링크: https://arxiv.org/abs/1607.00148.
- 핵심 내용: 다중 센서 시계열을 LSTM Enc-Dec 구조로 학습하고, 정상 패턴 복원 오차를 기반으로 이상을 탐지한다.
- 초안 관련성: 매우 높다. 현재 초안의 “정상 데이터로 Enc-Dec를 학습하고 복원 오차를 점검 점수로 사용한다”는 핵심 논리와 직접 연결된다.
- 보완 방향: 기존 Reference [3]으로 유지하되, 본문에서 “다중 센서 시계열에서 Enc-Dec 복원 오차 기반 이상탐지가 활용되어 왔다”는 문장을 더 명확히 써도 된다.
- 주의점: 논문 자체는 일반 다중 센서 데이터 대상이므로 유도탄 데이터 적용성은 초안에서 별도 가정과 제한을 두어야 한다.

### 3.2 Hundman et al., Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding

- 링크: https://arxiv.org/abs/1802.04431.
- 핵심 내용: NASA SMAP/MSL 우주기 텔레메트리에서 LSTM 예측 오차와 동적 임계값을 이용해 이상을 탐지한다.
- 초안 관련성: 매우 높다. 유도탄 데이터와 동일하지는 않지만 항공우주 계측·텔레메트리의 시계열 이상탐지 사례로 설득력이 크다.
- 보완 방향: 4.1절의 임계값 산정 부분에 “고정 임계값만으로 운영하지 않고 데이터 분포와 시간적 특성을 반영한 동적 기준이 필요하다”는 근거로 넣는 것이 좋다.
- 주의점: 예측 모델 기반 논문이므로 Enc-Dec 복원 오차와는 점수 생성 방식이 다르다. 초안에서는 “시계열 모델 기반 잔차”라는 상위 개념으로 연결한다.

### 3.3 Kim et al., AI-Based DAS Data Analysis for Real-Time Anomaly Detection in Missile Fire Control System

- 링크: https://www.kimst.or.kr/electronic_paper/pdf_down.kin?idx=25675.
- 핵심 내용: 미사일 FCS의 DAS 통신 로그를 대상으로 주기 메시지, 시간 편차, 메시지 손실, 지터를 시계열 모델과 규칙 기반 임계값으로 검증하는 구조를 제안한다.
- 초안 관련성: 매우 높다. KIMST 내 유도무기체계 분야이며, “AI 추론 + 규칙 기반 검증” 구조가 현재 초안과 거의 같은 운용 철학을 갖는다.
- 보완 방향: 서론과 4.2절에 국내 선행 사례로 추가하는 것을 권장한다. 특히 초안의 “모델 점수와 규칙 위반을 병렬로 산출한다”는 구조를 뒷받침한다.
- 주의점: 해당 논문은 학술대회 논문이고 FCS 통신 메시지 중심이다. 본 초안의 범위가 센서·상태·이벤트 로그 전체로 더 넓다는 차이를 밝혀야 한다.

### 3.4 Zhang et al., MSCRED

- 링크: https://arxiv.org/abs/1811.08055.
- 핵심 내용: 다변량 시계열의 채널 간 관계를 signature matrix로 표현하고 CNN/ConvLSTM 기반 재구성 오차로 이상 탐지와 진단을 수행한다.
- 초안 관련성: 높다. 유도탄 데이터 점검에서 채널 간 관계 붕괴를 다룰 때 근거로 쓸 수 있다.
- 보완 방향: 3.2절의 “복잡한 구조는 채널 간 관계를 세밀하게 반영할 수 있으나 초기 점검 체계에서는 단순 모델을 우선한다”는 문장에 예시로 유지한다.
- 주의점: 현재 초안이 단순 구조를 목표로 하므로 본문에서는 확장 후보로만 언급하는 편이 적절하다.

### 3.5 Xu et al., Anomaly Transformer

- 링크: https://openreview.net/forum?id=LzQQ89U1qm_ 또는 https://arxiv.org/abs/2110.02642.
- 핵심 내용: 시계열 이상탐지를 위해 association discrepancy를 사용하고, 이상 구간의 attention 특성을 모델링한다.
- 초안 관련성: 중간이다. 최신성은 좋지만 현재 초안의 단순 Enc-Dec 구조와 직접 연결성은 약하다.
- 보완 방향: 6장 고찰에서 “향후 복잡한 의존성 모델링이 필요하면 Transformer 계열로 확장 가능하다”는 한 문장 정도가 적절하다.
- 주의점: 초안의 핵심 Reference로 올리면 단순 구조라는 사용자 방향과 어긋날 수 있다.

### 3.6 Tuli et al., TranAD

- 링크: https://arxiv.org/abs/2201.07284.
- 핵심 내용: Transformer 기반 다변량 시계열 이상탐지 모델이며, attention과 adversarial training을 사용한다.
- 초안 관련성: 중간이다. 고도화 후보로는 좋지만 초기 점검 구조에는 과하다.
- 보완 방향: Anomaly Transformer와 함께 “향후 고도화 후보”로 묶어 고찰에 배치한다.
- 주의점: 본 초안에 너무 자세히 설명하면 논문 방향이 복잡해진다.

### 3.7 Ren et al., Time-Series Anomaly Detection Service at Microsoft

- 링크: https://arxiv.org/abs/1906.03821.
- 핵심 내용: Microsoft의 대규모 시계열 이상탐지 서비스 운영 경험과 SR-CNN 기반 모델을 다룬다.
- 초안 관련성: 중간 이상이다. 실무형 점검 체계와 운영 관점에서 도움이 된다.
- 보완 방향: 4.2절 또는 6장에 “모델 단독보다 운영 기준, 임계값, 오탐 관리가 중요하다”는 근거로 활용한다.
- 주의점: 방산 데이터와 직접 관련은 없으므로 방법론 운영 사례로만 인용한다.

### 3.8 Kim et al., GAN을 활용한 무기체계 PHM 분석 이상 탐지 방안 연구

- 링크: https://www.kimst.or.kr/electronic_paper/pdf_down.kin?idx=20294.
- 핵심 내용: 무기체계 PHM에서 고장 데이터 부족과 데이터 불균형 문제를 지적하고, 합성데이터와 GAN 활용 가능성을 검토한다.
- 초안 관련성: 높다. 현재 초안의 “라벨이 부족한 초기 점검 환경”이라는 문제 정의를 국내 국방 사례로 보강한다.
- 보완 방향: 2.2절에 고장 라벨 부족과 정상 데이터 기반 점검 필요성을 설명할 때 인용하면 좋다.
- 주의점: GAN 기반 합성데이터 논문이므로 Enc-Dec 방법론의 직접 근거로 쓰기보다는 데이터 부족 배경 근거로 쓰는 편이 맞다.

### 3.9 Min and Jin, 빅데이터 기반 창정비용 시험장비 시험결과 DB 분석 SW 개발

- 링크: https://www.kimst.or.kr/electronic_paper/pdf_down.kin?idx=24227.
- 핵심 내용: 창정비 시험장비의 시험결과 데이터를 DB화하고, 기존 단순 합부 판정에서 추이 분석과 비교 분석으로 확장하는 SW를 제안한다.
- 초안 관련성: 중간 이상이다. AI 모델 논문은 아니지만 데이터 점검 체계와 시험결과 DB 관리 필요성을 국내 사례로 보여준다.
- 보완 방향: 서론에서 “시험결과를 단순 합부 판정으로 끝내지 않고 누적 데이터 기반으로 비교·분석해야 한다”는 문제 제기에 활용한다.
- 주의점: 직접 이상탐지 알고리즘 논문은 아니므로 Reference 우선순위는 3순위다.

## 4. 초안 보완 권장안

### 4.1 Reference 보강 권장

현재 초안 Reference에 추가하거나 교체를 검토할 항목은 다음과 같다.

1. Kim, H., Kwon, K., Choi, G., Moon, J. and Jung, H., “An AI-Based DAS Data Analysis for Real-Time Anomaly Detection in Missile Fire Control System,” Proceedings of the KIMST Annual Conference, Guided Weapon System, pp. 599-600, 2025.
2. Ren, H., Xu, B., Wang, Y., Yi, C., Huang, C., Kou, X., Xing, T., Yang, M., Tong, J. and Zhang, Q., “Time-Series Anomaly Detection Service at Microsoft,” Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 3009-3017, 2019.
3. Kim, G., Yoon, Y. A., Bak, B., Kim, M., Park, K., Jo, S. and Choi, H., “A Case Study on PHM Analysis Using Synthetic Data,” Proceedings of the KIMST Annual Conference, Systems Engineering, pp. 1739-1740, 2022.
4. Min, G. and Jin, J., “Development of DB Analysis SW for Depot Test-Equipment Based on Big Data,” Proceedings of the KIMST Annual Conference, Structures and Machinery, pp. 73-74, 2025.

### 4.2 본문 보완 권장

- 서론: KIMST FCS 논문을 인용해 유도무기체계 데이터의 실시간 이상탐지 필요성을 국내 선행연구로 제시한다.
- 문제 정의: PHM/GAN 논문을 인용해 무기체계 데이터에서 고장 라벨과 비정상 데이터가 부족하다는 점을 보강한다.
- 방법론: Malhotra 논문과 Hundman 논문을 나란히 두어 복원 오차 기반 점검과 텔레메트리 잔차 기반 점검의 공통 구조를 설명한다.
- 고찰: MSCRED, Anomaly Transformer, TranAD는 확장 후보로만 다루고 본 논문은 단순 구조를 택했다는 논리를 강화한다.
- 기대 효과: 창정비 시험결과 DB 분석 논문을 인용해 “합부 판정 이후 누적 데이터 기반 추이 분석”의 필요성을 연결한다.

## 5. 최종 판단

현재 초안에는 Enc-Dec 방법론 근거는 어느 정도 있으나, 국내 유도무기체계 및 국방 시험 데이터 사례가 부족하다. 가장 먼저 추가해야 할 논문은 KIMST의 미사일 FCS DAS 실시간 이상탐지 논문이다. 그 다음으로 PHM/GAN 논문과 창정비 시험결과 DB 분석 논문을 배경 근거로 추가하면, 논문이 단순한 일반 이상탐지 방법론이 아니라 국방 데이터 점검 문제를 다루는 글로 보강된다.
