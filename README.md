# AIbemyEYE

Stage 0 — 전제/데이터 파이프라인 준비 (필수)

목표: 모델이 요구하는 입력 형태·증강·메타데이터 규격을 정의한다.

원시 포맷 규격

한 센서 항목: {id, raw_rate, raw_array (numpy), meta: {type? bit_start? pair?}, time_start(optional)}

파일/스톱시스템

큰 데이터는 per-scene 파일(예: 10~60s 단위), 압축은 zarr/hdf5 권장.

Preproc

정규화: 각 센서별 robust z-score (x - median)/MAD 또는 min-max(0..1) — rate별로 차등 적용.

Missing handling: 선형보간/forward fill/flag.

Augment

Contrastive용: jitter, scale, mask, cutout, bitflip(for bitfields), reorder windows (for permutation invariance experiments).

Metadata

raw32 보관 등 2word 관련 정보는 meta에 보관.

Stage 1 — Pre-classifier (data type 식별)

목적: 각 센서(또는 센서 행렬)를 보고 type ∈ {1word, 2word, complex_bits} 를 예측. 또한 2word라면 pair candidate(LSB/MSB) 가능성을 제공.

접근 (실무 권장)

Per-rate pre-classifier: 같은 raw_rate 그룹끼리 별도 pre-classifier(빠른 inference).

모델: 1D-CNN → Attention pooling → MLP classifier

이유: CNN은 variable length 허용, 파라미터 적음, 빠름.

입력: single-sensor time series (variable T)

출력:

node_type_logits (3-way softmax)

pair_score_vector (N-length if entire scene provided) — optional: if you feed whole scene, else produce embeddings for candidate pairing stage

학습

데이터: 시뮬레이터로 대규모 synthetic dataset 생성 (다양한 bit/word arrangements, msb/lsb 위치, random scattering).

loss: CrossEntropy(node_type) + (optional) contrastive loss for embedding separability

hparams (권장):

emb_dim = 128, lr = 1e-3, batch_size = 256 (sensor-level), epochs = 50

augment each sample twice for contrastive pretrain if desired, temp=0.1

Output 인터페이스
preclassifier.predict(sensor_tensor) -> {
  'type_probs': [p1,p2,p3],
  'emb': vector128  # optional
}

Stage 2 — 타입별 인코더 + Header/Body 통합

목적: 각 rate/type에 최적화된 인코더로 특징을 뽑아 공통 dimension으로 투영(헤더), 이후 시간·센서 간 상호작용(바디)으로 이상점 판단.

2.1 Type-specific encoders (Header)

구조:

For high-rate (2kHz): TCN / 1D-CNN stack (dilated) → multi-scale features → temporal-compressed representation (L_bins × D) + pooled vector v (D).

For mid/low-rate: ResNet1D / small Transformer → same outputs.

For complex bits: bit-aware encoder:

Parse 16-bit word vector per time step into bit-channels (16 dims) → small conv over time + embedding → also detect local patterns such as toggles, counters.

Output:

v_i (sensor-level vector, dim D) — for GNN/ML fusion input

T_feat_i (temporal compressed features shape L×D) — for overflow/time-event prediction & framewise scoring

2.2 Header Network (Projection)

All encoders project to a shared D (e.g. 128) via small MLP: proj(v) -> z.

This allows heterogeneous encoders to feed a common fusion module.

2.3 Body Network (Fusion)

Two main choices:

A) Cross-Attention Transformer (recommended)

Input: sequence of node-temporal-embeddings — we can represent each sensor as sequence of L tokens (temporal compressed).

Use a multi-modal cross-attention:

Per-sensor temporal tokens attend within-sensor (local) then cross-sensor.

Design: per-sensor encoder outputs L×D; create tokens sensor_id + time_bin and apply Transformer layers with sparse attention (local + cross-sensor top-k).

Output: per-time-bin anomaly scores per sensor (or aggregated per global time bin).

B) GNN over sensor-nodes + temporal tokens

Build graph nodes = sensors; edges built from similarity/physical adjacency; node features = pooled + aggregated temporal summaries.

Use message passing → update node state and optionally decode temporal predictions using the temporal compressed features and pairwise attention.

2.4 Output(s)

framewise anomaly score: score_{sensor, bin} (continuous)

binary decision: after EVT thresholding or percentile thresholding → normal/abnormal

pairing outputs: pair likelihood matrix (for 2word detection)

bitmask predictions: for bits nodes (optional)

Losses (unsupervised / weakly-supervised)

Because labels mostly absent, use self-supervised + weak labels + EVT:

Self-supervised representation losses (pretrain)

Contrastive (NT-Xent) on v or on temporal tokens (positive = augmented views of same sensor window, negatives = other sensors/windows).

CPC (Contrastive Predictive Coding): predict future compressed token given past.

Masked reconstruction for bits (predict masked bits from context).

Reconstruction loss (optional AE head)

Reconstruct temporal compressed features, use MSE. Helps modeling typical patterns.

Temporal scoring loss (semi-supervised / pseudo)

If you have small labeled anomalies (or human in loop), supervise with BCE/CE on frame level.

Pair detection and overflow losses (for 2word)

Pair contrastive: bring MSB/LSB embeddings close (if pair known from meta or synthetic).

Overflow event detection: per-pair time-bin BCE on detected rollovers (if raw32 available in synthetic training).

Regularization

Cosine-norm constraints, KL-diversity for experts in MoE, dropout.

Loss composition example (training head & body)

L = λ_c * L_contrastive + λ_rec * L_recon
    + λ_pair * L_pair + λ_overflow * L_overflow_bins


Initial recommended λ values:

λ_c = 1.0, λ_rec = 0.5, λ_pair = 0.5, λ_overflow = 1.0 (tune per dataset)

Stage 2 Practical Training Recipe

Pretrain encoders (self-supervised):

Per-rate pretrain using contrastive/CPC for 50–200 epochs.

Augmentations: jitter/mask/bitflip/time-warp

Freeze header → train fusion (body) with contrastive + reconstruction:

Optionally initialize fusion with small LR on encoder params.

If small labeled set available:

Finetune with framewise BCE on anomaly labels (weak-supervision).

Estimate anomaly score distribution on validation normal data:

Collect validation normal set (or most of data assumed normal) → compute scores → fit EVT (Generalized Pareto Distribution) to upper tail of scores or use percentile (e.g. 99.5%) as initial threshold.

Stage 3 — Thresholding, Human-in-the-loop, Continual Finetune
EVT thresholding (recommended)

For anomaly scores s_t, estimate tail behavior:

Choose high threshold u (e.g., 95th percentile of s_t on validation normal set)

Collect exceedances y = s_t - u and fit GPD to y

Choose target false alarm rate α (e.g., 1e-3/day) → invert GPD to get decision threshold s*

Practical simplification: if no EVT impl, use percentile thresholds (99.7%).

Human-in-the-loop

Show top-k highest scoring windows to an operator for labeling (active learning).

Two feedback styles:

Binary correction: operator marks window normal/abnormal → add as labeled examples for finetune.

Weak region marking: operator marks long intervals abnormal → produce weak labels for MIL (Multiple Instance Learning).

After accumulating N_feedback (e.g., 200 samples), perform finetune:

Small LR, weighted BCE with positive class upsampling, early stopping.

Continual learning / drift adaptation

Keep an online buffer of recent features/scores.

Drift detection:

Monitor embedding distribution shift via KL divergence or MMD between current window and baseline.

If drift detected, trigger unsupervised adaptation: fine-tune encoders on recent data with self-supervised objectives (contrastive + replay).

Use regularized finetune: LR low (1e-5), batchnorm momentum reset, freeze lower layers.

Stage 4 — Evaluation & Metrics

Because supervised labels are scarce, use a mixture:

A. If small labeled eval set exists:

Frame-level: Precision/Recall/F1, AUROC (score vs label), AUPR

Event-level: segment-level IoU (predicted abnormal segments vs GT segments)

Pair detection (for 2word): precision/recall/F1 on pair edges

Bitmask IoU for bits nodes

B. Unsupervised diagnostics (no labels)

Reconstruction error distribution on held-out normal set

Embedding cluster stability (k-means inertia over time)

Alarm rate vs. operator time (false alarm rate) — major production metric

C. Drift & robustness

Monitor score quantiles (median, 95th, 99.9th) over sliding windows → alert if shifting significantly.


```mermaid
flowchart TD
  A[Raw multi-rate sensors] --> B[Preprocessing & augmentation]
  B --> C[Pre-classifier per-rate] 
  C --> D1[Type-specific Encoder A '2kHz]
  C --> D2[Type-specific Encoder B '200Hz']
  C --> D3[Type-specific Encoder C '50Hz']
  D1 --> E[Header projection 'shared dim']
  D2 --> E
  D3 --> E
  E --> F[Temporal Fusion Body 'cross-attention / GNN']
  F --> G[Anomaly Scorer per time-frame]
  G --> H[EVT thresholding & alerts]
  H --> I[Human-in-the-loop labeling / feedback]
  I --> J[Continual fine-tune / drift detection]


```

파일 구조
``` bash
project/
├─ data/
│  ├─ generator.py            # realistic generator for sim data
│  ├─ dataset.py              # SceneDataset, sensor loaders
├─ models/
│  ├─ preclassifier.py        # per-rate pre-classifier model
│  ├─ encoders.py             # rate-specific encoders
│  ├─ fusion.py               # cross-attention/GNN fusion
│  ├─ heads.py                # anomaly head, overflow head, pair head
├─ train/
│  ├─ pretrain.py             # contrastive pretrain scripts
│  ├─ finetune.py             # fusion training with loss composition
│  ├─ evaluate.py             # metrics, EVT thresholding
├─ tools/
│  ├─ augment.py
│  ├─ evt.py                  # EVT fit & threshold utilities (GPD)
│  ├─ viz.py
└─ experiments/
   └─ config.yaml

```



Soft ↔ Hard gating 전환 전략 포함 MoE multi-modal pipeline을 Mermaid flow로 시각화

**heterogeneous 센서 입력, gating network, expert encoders, soft/hard gating 선택, fusion, shared embedding, downstream task, EVT threshold까지 포함됩니다.**

```mermaid
flowchart TD
    %% ================= Inputs =================
    subgraph Inputs["🟦 Multi-Modal Sensor Inputs"]
        X1["X1: Binary / 1bit"]
        X2["X2: 2-word / 16bit"]
        X3["X3: Float / Int"]
        X4["X4: Multi-bit (3~15 bits)"]
    end

    %% ================= Initial Embedding =================
    subgraph InitEmbed["🟨 Initial Embedding / Preprocessing"]
        PRE1["Binary → bitwise linear → hidden_dim"]
        PRE2["2-word → optional bit embedding → linear → hidden_dim"]
        PRE3["Float/Int → linear → hidden_dim"]
        PRE4["Multi-bit → linear → hidden_dim"]
    end

    X1 --> PRE1
    X2 --> PRE2
    X3 --> PRE3
    X4 --> PRE4

    %% ================= Gating Network =================
    subgraph Gating["🟩 Gating Network"]
        GATE["MLP → softmax → gating_prob (soft)"]
        SOFT["Soft Gating → weighted sum of all Expert outputs"]
        HARD["Hard Gating → argmax / top-k selection"]
        SWITCH["Soft ↔ Hard Switch (training phase dependent)"]
    end

    PRE1 --> GATE
    PRE2 --> GATE
    PRE3 --> GATE
    PRE4 --> GATE
    GATE --> SWITCH
    SWITCH --> SOFT
    SWITCH --> HARD

    %% ================= Expert Encoders =================
    subgraph Experts["🟧 Expert Encoders (MoE)"]
        E1["Expert 1: Binary-focused"]
        E2["Expert 2: Float/Int-focused"]
        E3["Expert 3: Multi-bit / High-cardinality"]
    end

    SOFT --> E1
    SOFT --> E2
    SOFT --> E3

    HARD --> E1
    HARD --> E2
    HARD --> E3

    %% ================= Fusion Layer =================
    subgraph Fusion["🟨 Fusion Layer / Shared Embedding"]
        CONCAT["Concat / Attention / Weighted sum"]
        Z_shared["Shared embedding z_t"]
    end

    E1 --> CONCAT
    E2 --> CONCAT
    E3 --> CONCAT
    CONCAT --> Z_shared

    %% ================= Downstream =================
    subgraph Downstream["🟪 Downstream Tasks"]
        PRETRAIN["Self-Supervised: CPC / Contrastive / Recon Loss"]
        FINETUNE["MIL / Weak Label Fine-tune"]
        HUMAN["Human Feedback / Pseudo-label update"]
        CONTINUAL["Continual Fine-tune / Drift adaptation"]
        EVT["EVT / Percentile Threshold"]
        LABEL["Normal / Abnormal Label"]
    end

    Z_shared --> PRETRAIN --> FINETUNE --> HUMAN --> CONTINUAL --> EVT --> LABEL

    %% ================= Legend / Notes =================
    classDef inputs fill:#d0ebff,stroke:#000,stroke-width:1px;
    classDef preprocessing fill:#fff3bf,stroke:#000,stroke-width:1px;
    classDef gating fill:#c3f0ca,stroke:#000,stroke-width:1px;
    classDef experts fill:#ffd6d6,stroke:#000,stroke-width:1px;
    classDef fusion fill:#fff3bf,stroke:#000,stroke-width:1px;
    classDef downstream fill:#e0c3ff,stroke:#000,stroke-width:1px;

    class Inputs inputs;
    class InitEmbed preprocessing;
    class Gating gating;
    class Experts experts;
    class Fusion fusion;
    class Downstream downstream;

```


사용 설명 (간단)

generate_multimodal_data_advanced(...) 호출로 센서 리스트와 (N x target_T) 정렬된 행렬을 얻습니다.

MoEMultiSensorDataset는 PyTorch 학습 루틴에 바로 사용될 수 있는 형태로 각 센서의 원시(raw)와 정렬(aligned) 데이터를 제공합니다.

anomaly insert는 anomaly_cfg 파라미터로 제어 가능합니다.

use_multirate=True이면 센서별로 랜덤하게 50/200/2000Hz를 사용하여 원시 데이터를 만들고, 마지막에 모두 resample로 target_T 길이로 정렬합니다.




**"main_pre_classifier.py"** 구조

```mermaid
flowchart TB
    subgraph INPUT
        S[scene: list of sensors] --> RAW[raw np.array/tensor/list, raw_rate, type, meta]
    end

    subgraph ENCODERS
        RAW --> ENC[RateEncoderTemporal<br>outputs: vec + temporal features]
    end

    subgraph GRAPH
        ENC --> X[X: N x emb]
        X --> G1[SimpleGraphLayer g1]
        G1 --> H1[H1: N x emb]
        H1 --> G2[SimpleGraphLayer g2]
        G2 --> H2[H2: N x emb]
    end

    subgraph HEADS
        H2 --> NODE[node_head: N x 3]
        H2 --> BIT[bitmask_head: N x 16]
        H2 --> PAIR[pair_feat: N x N x 2*emb]
        PAIR --> EDGE[edge_head: N x N]
        PAIR --> ORDER[order_head: N x N x 2]
        ENC --> TEMP[temps: list of L x emb]
        TEMP --> OVERFLOW[overflow temporal conv per-pair<br>conv1->ReLU->conv2->ReLU->conv_out]
    end

    subgraph OUTPUT
        NODE --> OUT[node_logits: N x 3]
        EDGE --> OUT2[edge_logits: N x N]
        ORDER --> OUT3[order_logits: N x N x 2]
        BIT --> OUT4[bitmask_logits: N x 16]
        OVERFLOW --> OUT5[overflow_logits: N x N x L]
        G2 --> ATT[attention: N x N]
    end
```


**Contrastive Learning(pretrain)**

```mermaid
flowchart TB
    subgraph PRETRAIN_Contrastive
        POOL[Collect  raw, rate from all scenes] --> BATCH[Random batch]

        BATCH --> LOOP1[For each sensor]
        LOOP1 --> RAW[Convert raw to 1D tensor]
        RAW --> AUG1[Aug1]
        RAW --> AUG2[Aug2]

        AUG1 --> ENC1[Encoder rate]
        AUG2 --> ENC2[Encoder rate]

        ENC1 --> Z1[z1 batch]
        ENC2 --> Z2[z2 batch]

        Z1 --> NTX[NT-Xent loss]
        Z2 --> NTX

        NTX --> OPT[Adam update]
    end

```



**Full fine-tue Learning**

```mermaid
flowchart TB
    subgraph Finetune
        TRAIN[Train Dataset Scenes] --> LOADER[SceneDataLoader]

        LOADER --> SCENE[Scene list]

        SCENE --> GT[Build Ground Truth numpy->tensor]
        GT --> GTD[Move GT to device]

        SCENE --> MODEL[SensorStructureModel forward]
        MODEL --> OUT[Outputs dict]

        OUT --> SAN[Sanitize outputs nan->num]
        GTD --> SANGT[Sanitize GT]

        SAN --> LOSS[Compute all losses]
        SANGT --> LOSS

        LOSS --> CHECK[Check finite]
        CHECK -->|finite| BACKWARD[Backward + Clip + Optim Step]
        CHECK -->|not finite| SKIP[Skip step]

        BACKWARD --> ACC[Accumulate loss mean]

        ACC --> END1[Print epoch train loss]

        subgraph Validation
            VALDS[Val dataset] --> VALLOADER
            VALLOADER --> VSC[V scene]
            VSC --> VGT[GT build]
            VSC --> VOUT[Model forward]
            VGT --> MET[Metrics]
            VOUT --> MET
            MET --> END2[Print validation metrics]
        end
    end

```




** Module Structure - 전체 모델 구조 **
``` mermaid
flowchart LR
    %% INPUT
    SENS[Scene sensors list] --> RATESEL[Select encoder by raw_rate]

    %% ENCODER
    RATESEL --> ENC[RateEncoderTemporal]
    ENC --> VECS[vec per sensor]
    ENC --> TEMP[tfeat per sensor L x Emb]

    %% BUILD X
    VECS --> X[N x Emb]

    %% GRAPH STACK
    X --> G1[GraphLayer1]
    G1 --> H1[N x Emb]

    H1 --> G2[GraphLayer2]
    G2 --> H2[N x Emb]
    G2 --> ATT[attention N x N]

    %% HEADS
    H2 --> NODE[node head N x 3]
    H2 --> BIT[bitmask head N x 16]

    H2 --> PAIR[pair features N x N x 2Emb]

    PAIR --> EDGE[edge head N x N]
    PAIR --> ORDER[order head N x N x 2]

    TEMP --> OVRCONV[overflow temporal conv stack]
    OVRCONV --> OVR[overflow logits N x N x L]

    %% OUTPUT
    NODE --> O1[node_logits]
    BIT --> O2[bitmask_logits]
    EDGE --> O3[edge_logits]
    ORDER --> O4[order_logits]
    OVR --> O5[overflow_logits]
    ATT --> O6[attention_matrix]

```

** Module Structure(Simple GraphLayer) **
``` mermaid
flowchart TB
    subgraph SimpleGraphLayer
        X[Input X N x Emb] --> Q[Linear -> Q]
        X --> K[Linear -> K]
        X --> V[Linear -> V]

        Q --> MATMUL1[Q x K^T / sqrt Emb ]
        K --> MATMUL1
        MATMUL1 --> SOFT[Softmax row-wise]
        SOFT --> ATT[Attention A N x N]

        ATT --> MATMUL2[A x V]
        V --> MATMUL2

        MATMUL2 --> RES[Residual + X]
        RES --> OUT_L[Linear]
        OUT_L --> REL[ReLU]
        REL --> OUT[Output H N x Emb]
    end

```

** Module Structure(Rate Encoder Temporal) **
```mermaid

flowchart TB
    subgraph RateEncoderTemporal
        X[Input 1D time-series T] --> U1[Unsqueeze to 1x1xT]
        U1 --> C1[Conv1d 1->Cc k7]
        C1 --> R1[ReLU]
        R1 --> C2[Conv1d Cc->Cc k5]
        C2 --> R2[ReLU]

        %% vector branch
        R2 --> P1[AdaptiveAvgPool1d 1]
        P1 --> S1[Squeeze]
        S1 --> FC[Linear Cc->Emb]
        FC --> Vec[Vector Emb]

        %% temporal branch
        R2 --> TP[AdaptiveAvgPool1d L bins]
        TP --> PROJ[Conv1d Cc->Emb k1]
        PROJ --> TFEAT[Squeeze and Permute L x Emb]

        Vec --> OUT1(Output vec)
        TFEAT --> OUT2(Output temporal)
    end
```
