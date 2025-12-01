🧭 Multi-Rate · Multi-Type Unsupervised Sensor Anomaly Detection
Full Research & Engineering Roadmap (Markdown)
#️⃣ Overview

본 프로젝트의 목표는 다음과 같은 복합 조건에서 scalable·unsupervised anomaly detection pipeline 구축:

Multi-Rate: {2000Hz, 200Hz, 50Hz, Non-Periodic}

Multi-Type: {1word(16bit), 2word(32bit, MSB/LSB), complex(bitwise multi-field)}

Dataset dimension 변화에도 안정적으로 작동

대규모 데이터 → Fully supervised 방식 불가

Temporal 비정상(Anomaly) 탐지 필요

🏗 전체 Architecture (Stage 1 → Stage 3)
```pgsql
Raw Multi-Rate Sensors
        │
        ▼
─────────────────────────────────────
 Stage 1. Pre-Classifier (Data Type ID)
─────────────────────────────────────
        │ type label
        ▼
─────────────────────────────────────
 Stage 2. Representation Learning
   2.1 Type-Specific Network
   2.2 Rate Header Network
   2.3 Body Fusion Network (GNN/LSTM)
─────────────────────────────────────
        │ unified representation
        ▼
─────────────────────────────────────
 Stage 3. Fine-Tune (Human-in-the-loop)
─────────────────────────────────────
        │
        ▼
   Anomaly per timestep


```

🎯 Stage 1 — Pre-Classifier (Data Type Identification)
📌 목표

Multi-Type(1word/2word/complex)을 자동 식별하여
각 rate별 type 라우팅 자동화

실제 데이터는 type label이 부족하므로
random synthetic generation 기반 Self-Supervised classifier 학습

📌 해야 할 일 (To-Do)
1.1 Synthetic Multi-Type Data Generator 개선

✓ 1word, 2word(MSB/LSB), complex-bit 구조 정확히 생성

✓ Real distribution matching (amplitude, correlation, burst event, overflow 등)

✓ 각 rate별 다른 noise profile 구축

✓ Non-periodic 이벤트도 포함

1.2 Pre-Classifier Architecture 설계

입력 길이 변화 대응을 위해:

옵션 A) Adaptive Pooling 기반 CNN

1×T → adaptive_avg_pool → 1×fixed

차원이 달라도 동일한 classifier 사용 가능

Temporal structure 일부 보존

옵션 B) GNN (Graph-of-Time)

timestep을 node로 보고 multi-rate merging 가능

irregular sampling도 처리 가능

옵션 C) LSTM/Transformer + packed sequence

variable length batch 처리 가능

하지만 multi-rate alignment 필요성 ↑

=> 추천: 옵션 A (Adaptive CNN) + 옵션 C (Packed LSTM)

1.3 Loss 구성

Cross-entropy(type classification)

2word에 대해 MSB/LSB alignment loss 추가

L_msb_lsb = BCE(pred_MSB, GT_MSB) + BCE(pred_LSB, GT_LSB)

1.4 출력

type_id ∈ {1word, 2word, complex}

optional: bit-mask / MSB-LSB pair confidence

🎯 Stage 2 — Representation Learning Core

Stage 1에서 type이 결정되면 sensor는 아래 path로 들어감.

2.1 Type-Specific Network
📌 목표

Type별로 적절한 encoder 사용

Hidden size는 동일하여 downstream fusion이 가능하도록 설계

Type	입력 예시	권장 Network
1word	정수 또는 float 값	CNN / LSTM encoder
2word	MSB/LSB	bit-GNN 또는 MLP pair encoder
complex bits	multi-field / bitmask	Bit-GNN, Set Transformer
📌 해야 할 일 (To-Do)

✓ 각 type encoder를 모듈화

✓ embedding dimension 통일 (ex. 128d)

✓ Contrastive learning 도입

time-shift positive / other rate negative

overflow·burst event 별로 event-level contrastive 학습

2.2 Rate Header Network
📌 목표

각 rate encoder 출력 → 서로 다른 sampling에도 공통 공간으로 projection

비유: 서로 다른 길이의 영상을 찍었지만 동일한 descriptor로 만드는 단계.

📌 해야 할 일 (To-Do)

✓ {2kHz, 200Hz, 50Hz, Non-periodic} encoder 각각 구성

✓ Temporal pooling 전략 선택

Adaptive pooling

Learnable pooling(weighted pooling)

Temporal attention pooling

✓ Output shape

H_rate = [B, 128]   # 모든 rate 동일 차원

2.3 Body Fusion Network (Multi-Rate Integration)
📌 목표

각 rate의 representation을 통합해
“시점별 전체 sensor system representation” 생성.

📌 후보 방식
A. GNN with Rate Graph

Node = sensor rate

Edge = causal/temporal correlation

Multi-rate irregular 데이터를 자연스럽게 fusion

B. Hierarchical Temporal Fusion Transformer

low rate → upsample하여 high rate에 alignment

temporal attention으로 multi-resolution 처리

C. Mixture-of-Experts (Soft/Hard gating)

high rate → detail

low rate → trend

gating으로 dynamic fusion

📌 해야 할 일 (To-Do)

✓ Rate간 attention matrix 학습

✓ Fusion vector 출력

✓ Stage 3와 연결

2.4 Output (Unsupervised / Semi-Supervised)
📌 목표

각 시간 프레임마다 정상/비정상 판단

🎯 Stage 3 — Fine-Tuning (Human-in-the-Loop)
📌 목적

Real-world anomaly에 대해 사람이 label 수정하여
모델의 false positive/negative 밸런스를 개선

📌 해야 할 일 (To-Do)
3.1 Human feedback loop

domain expert가 time frame 별 anomaly 검증

모델 output vs human 수정

3.2 Loss 반영

Semi-supervised loss:

L = L_consistency + L_reconstruction + L_human_label

3.3 Active learning loop

불확실도가 높은 구간만 sampling하여 labeling 효율↑

📦 Module 구성 (개발할 파트)
✨ Module 1 — Multi-Rate Multi-Type Generator

random + real-statistics 기반

1word / 2word / complex bit 구조 생성

✨ Module 2 — Pre-Classifier

type classifier

MSB/LSB classifier

bitmask approximate detection

✨ Module 3 — Type-Specific Encoders
✨ Module 4 — Rate Header Networks
✨ Module 5 — Body Fusion Network

GNN 또는 Transformer 기반

✨ Module 6 — Anomaly Inference Head

reconstruction error 기반

contrastive distance 기반

event-level detection

✨ Module 7 — Fine-Tune + Feedback Trainer
🎯 최종 결과물

Pre-classifier로 rate별 type 자동 식별

Type-specific encoder로 effective compression

Rate header → Body fusion으로 system-level representation

Human-in-the-loop로 fine tune

Dataset dimension 변화에도 작동하는 scalable architecture 완성
