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

