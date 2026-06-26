# 프로젝트 파악 컨텍스트 노트

## 2026-06-23

- 사용자의 요청은 현재 저장소를 파악하고 핵심 구조, 실행 방법, 주요 모듈을 요약하는 것으로 해석했다.
- 현재까지 루트에서 `README.md`와 `requirements.txt`가 확인되었다.
- 코드 변경은 요청 범위 밖이므로 프로젝트 문맥 정리용 문서만 추가한다.
- 프로젝트 목표는 이기종 멀티레이트 센서 시계열의 이상 탐지, 결함 센서 국소화, 피드백 기반 적응을 구현하는 연구 프로토타입이다.
- 실제 구현은 Stage 1 학습 파이프라인, 프리클래시파이어, 멀티모달 이상 탐지 모델, 실시간 오케스트레이터, NASA와 SWaT 평가 스크립트를 포함한다.
- `scripts/stage1_generate_data.py`는 `save_stage1_data`를 import하지만, 현재 `src/data/data_generator.py`에는 해당 함수 선언이 보이지 않고 저장 코드가 `return` 이후에 위치해 실행되지 않는 구조로 보인다.
- `requirements.txt`에는 `scipy`와 `tqdm`이 빠져 있지만 코드에서는 `main_data_generator.py`와 Stage 1 학습 스크립트가 각각 사용한다.
- 무거운 학습은 실행하지 않았고, Python 문법 검증은 `compileall`로 통과했다.

## 2026-06-23 프로젝트명 변경

- 새 프로젝트명은 `A_EYE`이고 풀네임은 `Anomaly Examination for Yielded Elements`이다.
- 현재 이름 표기는 주요 문서, UI 설계서, 패키지 주석, `pre_classifier.py`의 워크스페이스 경로 표기에 있었다.
- `recovered_conversation.md`는 과거 대화 복구 로그라 실제 프로젝트 표기 변경 대상에서 제외한다.
- 실제 폴더명 변경은 워크스페이스 루트의 부모 디렉터리 작업이므로 별도 권한이 필요하다.


- `recovered_conversation.md`를 제외한 현재 관리 대상 파일에서 이전 프로젝트명 문자열이 남지 않는 것을 확인했다.
- 변경된 Python 파일은 `python -m py_compile pre_classifier.py src\__init__.py src\data\__init__.py`로 문법 검증을 통과했다.
- Git 기준 untracked 상태였던 프로젝트 문서들은 작업 공간에서는 새 프로젝트명으로 갱신했지만, 파일 전체를 새로 추가하는 커밋이 되므로 이번 커밋 대상에서 제외한다.
- 루트 폴더 rename은 `Move-Item`과 `cmd ren` 모두 `The process cannot access the file because it is being used by another process.` 오류로 실패했다.
- 현재 Codex 세션이나 다른 프로세스가 기존 워크스페이스 폴더를 잡고 있는 상태로 판단된다.

## 2026-06-23 현재 구현 타당성 및 실시간 점검 파이프라인 리뷰

- 사용자의 요청은 전체 프로젝트와 현재 구현 결과의 타당성을 코드와 문서 기준으로 검토하고, 실제 디스플레이 기반 실시간 점검 파이프라인으로 연결하기 위한 보완 계획과 UI 계획을 세우는 것으로 해석했다.
- 이번 작업은 리뷰와 계획 수립이 중심이며, 명시적인 구현 요청은 아직 없으므로 코드 변경은 하지 않는다.
- Product Design 저장 컨텍스트는 존재하지 않았다. 따라서 UI 계획은 현재 저장소의 `docs/ui_design.md`와 실제 구현된 스크립트 및 결과 파일을 기준으로 한다.
- 현재 디자인 브리프는 멀티모달 센서 이상 탐지 결과를 실시간으로 관찰하고, 알람 검증, 센서 국소화, 피드백 수집, 모델 상태 점검을 한 화면에서 수행하는 운영자용 대시보드이다.

### 현재 리뷰 결과 요약

- `scripts/stage1_generate_data.py`는 `save_stage1_data`를 import하지만 `src/data/data_generator.py`에 해당 함수가 정의되어 있지 않다. 저장 로직은 `generate_stage1_simple_data`의 `return` 뒤에 있어 도달 불가능한 코드다.
- `main_data_generator.py`는 import 시점에 `matplotlib`를 로드하며, 현재 로컬 환경에서는 `NumPy 2.4.6`과 설치된 `matplotlib` 바이너리 ABI가 맞지 않아 import가 실패한다. 이 문제 때문에 `scripts/simulate_realtime_inspection.py`도 현재 환경에서 바로 실행될 수 없다.
- `generate_multimodal_data_advanced`는 정렬 데이터를 만들지 않지만 `MoEMultiSensorDataset`은 `s['aligned']`를 요구한다. 이 Dataset 경로는 현재 그대로 사용하면 `KeyError`가 발생한다.
- `AnomalyInspectionOrchestrator` 단독 스모크 테스트는 통과했다. `process_realtime_frame`은 JSONL 로그를 만들 수 있으므로 실시간 대시보드 연결의 최소 이벤트 소스는 존재한다.
- 저장된 결과 기준으로 시뮬레이션은 프레임 F1 0.7688, 이벤트 F1 0.8571, Top-5 국소화 100%를 보였지만 drift, spike, noiseburst 재현율은 낮다. NASA 평균 point F1은 0.3880, SWaT point F1은 0.24라 실제 데이터 일반화는 아직 약하다.
- UI 계획은 `docs/ui_design.md`의 센서 그리드, 추이 차트, 드리프트 게이지, 알람 진단, 피드백 콘솔을 유지하되, 로그 tailing이 아니라 오케스트레이터 상태 스트림 API를 중심으로 재정의해야 한다.

### 실시간 디스플레이 연결 계획

- 1단계는 실행 재현성 복구다. 의존성 pinning, `save_stage1_data` 복구, generator와 Dataset 스키마 정합성, ZOH 정렬 구현을 먼저 해결한다.
- 2단계는 오케스트레이터 출력 스키마 안정화다. `step`, `timestamp`, `sensor_status`, `score`, `threshold`, `drift`, `top_k_faults`, `feedback_state`, `model_state`를 명시한 이벤트 DTO를 만든다.
- 3단계는 API 계층 추가다. `POST /ingest-frame`, `GET /state`, `GET /events` 또는 WebSocket `/stream`, `POST /feedback`, `POST /control/threshold`를 제공한다.
- 4단계는 운영 UI 구현이다. 첫 화면은 센서 매트릭스, 전역 점수 추이, 알람 큐, 선택 센서 파형, 모델 상태, 피드백 콘솔로 구성한다.
- 5단계는 검증 루프다. 시뮬레이션 재생, NASA/SWaT 배치 재생, point 및 event 메트릭, 지연 시간, 오경보율, 국소화 Top-K를 같은 UI에서 확인한다.

## 2026-06-23 웹 대시보드 구현

- 사용자가 웹 대시보드 방식으로 진행하기로 결정했다.
- Product Design 저장 컨텍스트는 없으며, `docs/ui_design.md`와 기존 오케스트레이터 구조를 기준으로 구현한다.
- 로컬 환경에 `fastapi`가 설치되어 있지 않으므로 첫 구현은 네트워크 설치 없이 실행 가능한 표준 라이브러리 HTTP 서버와 Server-Sent Events 스트림으로 만든다.
- API 경계는 나중에 FastAPI로 이관하기 쉽도록 `/api/state`, `/api/events`, `/api/feedback`, `/api/threshold`, `/api/control` 형태로 유지한다.
- 대시보드 서버는 `main_data_generator.py`의 matplotlib import 문제를 우회하기 위해 자체 경량 합성 센서 버퍼를 생성하고 `AnomalyInspectionOrchestrator.process_realtime_frame`에 전달한다.

### 웹 대시보드 구현 결과

- `src/dashboard/runtime.py`는 기존 `AnomalyInspectionOrchestrator`를 직접 호출하는 실시간 런타임으로 추가했다. 입력 버퍼는 현재 환경의 `main_data_generator.py` import 문제를 피하기 위해 대시보드 런타임 내부의 경량 합성 센서 스트림으로 생성한다.
- `scripts/run_dashboard.py`는 표준 라이브러리 `ThreadingHTTPServer` 기반으로 구현했다. 제공 API는 `/api/state`, `/api/events`, `/api/sensor`, `/api/feedback`, `/api/threshold`, `/api/control`이다.
- `web_dashboard`는 운영자 첫 화면을 센서 매트릭스, 이상 점수 추이, 알람 콘솔, CLA 드리프트, 선택 센서 파형, 피드백 패널로 구성했다.
- 서브에이전트 검토 결과를 반영해 정적 파일 경로 탈출을 차단하고, 잘못된 입력은 400 응답으로 정리했다.
- 오케스트레이터 이벤트는 `revision`이 포함된 JSON 안전 상태로 정규화한다. `stop`, `start`, `threshold`, `feedback`처럼 step이 변하지 않는 상태 변경도 SSE로 즉시 전달된다.
- 프론트엔드는 SSE 오류 시 폴링으로 fallback하고, 재연결 시 폴링을 중단한다. 오래된 poll 응답이 최신 SSE 상태를 덮지 않도록 `revision` 기준으로 렌더링한다.
- 센서 버튼은 실시간 갱신마다 재생성하지 않고 기존 DOM을 업데이트한다. 선택 포커스와 접근성 라벨을 유지하기 위한 조치다.
- 검증은 `python -m py_compile`, `node --check`, 런타임 JSON 직렬화 스모크, `/api/state`, `/api/threshold`, `/api/events`, `/api/sensor`, 정적 자산 응답, 입력 오류 400 처리, 운영자 제어 API로 수행했다.
- 현재 로컬 서버는 `http://127.0.0.1:8765`에서 실행 중이며, 마지막 시작 PID는 `23084`이다.
- 브라우저 플러그인 기반 시각 검증은 Node REPL 샌드박스 ACL 오류로 수행하지 못했다. 대신 HTTP 응답과 정적 JS 문법 검증으로 대체했다.

## 2026-06-25 알고리즘 개선

- 개선 범위는 데이터 파이프라인 버그, 의존성 누락, 점수 산출 투명성, 하드코딩 규칙 격리, 임계값 보정, 평가 요약 표준화로 제한한다.
- 모델 아키텍처 전체 교체는 현재 실험 코드와 기존 평가 결과를 크게 흔들 수 있어 이번 변경에서는 새 대형 네트워크를 추가하지 않는다.
- `save_stage1_data`를 실제 함수로 복구해 `scripts/stage1_generate_data.py`의 import 계약을 만족시켰다.
- `AnomalyDetectorAgent`는 기존 `anomaly_score`를 유지하면서 `score_components`와 `threshold_info`를 반환하도록 바꿨다.
- 107번 pressure와 112번 valve bit 규칙은 `correlation_rules` 설정으로 이동했다.
- `evt_threshold`는 GPD 기반 계산을 시도하되 SciPy/NumPy 비호환이나 tail 부족 시 percentile fallback을 사용한다.
- SWaT와 NASA 평가 결과에는 기존 flat metric key를 보존하면서 표준 `report` 블록을 추가한다.
- 검증은 `py_compile`, `save_stage1_data` smoke test, `evt_threshold` smoke test, `AnomalyDetectorAgent` smoke test를 통과했다.

## 2026-06-26 벤치마크 교정 결과 확인

- 기존 `results/swat_performance.json`과 `results/nasa_performance.json`에는 새 `report.schema_version` 블록이 없어 개선 전 결과로 판단했다.
- 개선된 평가 스크립트 기준으로 SWaT와 NASA를 재실행해 결과를 확인한다.
- 벤치마크 재실행 중 현재 환경의 NumPy 2.4.6과 SciPy/sklearn ABI 비호환으로 `sklearn.metrics` import가 실패했다.
- `src.utils.evaluation.binary_detection_metrics`를 순수 NumPy 구현으로 바꿔 평가 지표 계산에서 sklearn/scipy 의존을 제거했다.
- 현재 환경의 pandas/pyarrow도 NumPy 2.4.6과 ABI 비호환으로 import에 실패했다.
- `swat_loader.py`와 `nasa_loader.py`에서 pandas 의존을 제거하고 표준 `csv`와 NumPy 기반 로딩으로 변경했다.

## 2026-06-26 벤치마크 교정 실행 완료

- SWaT 평가는 전체 CSV 숫자 파싱 대신 학습 stride 3000, 테스트 stride 500의 창 샘플링 로더로 교정했다. 새 결과는 Pt-F1 0.1667, PA-F1 0.3125이며 `results/swat_performance.json`에 `report.schema_version` 1과 `evaluated_windows` 2884를 기록했다.
- NASA 평가는 채널 `A-1`, `D-1`, `M-1`에 대해 빠른 교정 설정을 적용했다. 기본 설정은 train epoch 1, 최대 train batch 3, baseline fit batch 10, eval step 10이며 평균 Pt-F1 0.3405, 평균 PA-F1 0.5442를 기록했다.
- 현재 PC에서는 CUDA가 없고 `torch.cuda.is_available()` 경로가 멈출 수 있어 장치 선택을 `src/utils/device.py`로 분리했다. 기본 `auto`는 별도 Python 프로세스에서 제한 시간 안에 CUDA를 감지하고 실패하면 CPU로 떨어지며, GPU 장비에서는 `A_EYE_DEVICE=cuda`로 명시해 CUDA를 사용할 수 있다.
- NumPy 2.4.6 환경에서 SciPy/sklearn과 pandas/pyarrow ABI가 맞지 않아 평가 유틸과 SWaT/NASA 로더의 외부 의존성을 표준 라이브러리 `csv`와 NumPy 중심으로 줄였다.

## 2026-06-26 5번 시간 토큰 모델 구조 개선

- 기존 `MultiModalAnomalyModel`은 high, mid, low, bits를 각각 단일 벡터로 압축한 뒤 4개 modality 토큰만 fusion했다. 이 구조는 센서별 시간 패턴을 fusion 단계에서 직접 볼 수 없는 한계가 있었다.
- `TemporalTokenEncoder`를 추가해 high와 mid 입력을 각각 고정 개수 시간 토큰으로 보존했다. low와 bits는 센서 상태 토큰 1개씩으로 합쳐 총 `2 * time_tokens + 2`개의 토큰을 GNN과 Transformer에 전달한다.
- 기존 출력 키인 `fused`, `proj`, `recon`, `cpc_preds`, `bag_score`, `logits`, `inst_feats`는 유지했다. 새로 `tokens`, `sensor_feats`, `bag_repr`를 반환해 후속 분석과 국소화 개선에 사용할 수 있게 했다.
- 벤치마크와 운영 에이전트 기본값은 비용을 통제하기 위해 `time_tokens` 4로 두었다. 벤치마크 스크립트는 `A_EYE_TIME_TOKENS`로 값을 조정할 수 있다.
- 검증은 `py_compile`, 모델 forward shape 스모크 테스트, `AnomalyDetectorAgent.process` 스모크 테스트, NASA 한 채널 제한 평가 스모크 테스트로 수행했다.
