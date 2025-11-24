# AIbemyEYE


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
