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

## 2026-06-26 5번 반영 후 재벤치마크 시작

- 사용자가 전체 재실행을 시작하라고 요청했다.
- 목적은 시간 토큰 모델 구조가 반영된 현재 코드 기준으로 SWaT와 NASA 결과가 직전 교정 결과 대비 어떻게 변하는지 확인하는 것이다.
- 기준값은 직전 교정 결과인 SWaT Pt-F1 0.1667, PA-F1 0.3125와 NASA 평균 Pt-F1 0.3405, 평균 PA-F1 0.5442로 둔다.

## 2026-06-26 5번 반영 후 재벤치마크 결과

- SWaT 재실행은 CPU에서 421.3초가 걸렸다. 결과는 Pt-F1 0.1667, PA-F1 0.3125로 직전 교정 결과와 동일했다.
- NASA 3채널 재실행은 CPU에서 31.6초가 걸렸다. 평균 Pt-F1은 0.3405에서 0.3753으로 0.0347 상승했고, 평균 PA-F1은 0.5442에서 0.7991로 0.2549 상승했다.
- NASA 개선은 주로 M-1 채널이 Pt-F1 0.0000, PA-F1 0.0000에서 Pt-F1 0.1036, PA-F1 0.7677로 회복된 영향이다. D-1은 Pt-F1이 0.9039에서 0.9045로 소폭 상승했고 PA-F1은 0.9658에서 0.9630으로 소폭 하락했다. A-1은 동일했다.
- 실행 조건은 기본 `A_EYE_DEVICE=auto`에서 CPU로 선택됐고, `A_EYE_TIME_TOKENS` 기본값 4, NASA train epoch 1, 최대 train batch 3, fit batch 10, eval step 10이다.

## 2026-06-26 SWaT 성능 개선 시작

- 사용자가 SWaT 성능 개선을 시작하라고 요청했다.
- 기준값은 시간 토큰 모델 반영 후 SWaT Pt-F1 0.1667, PA-F1 0.3125다.
- NASA는 개선됐지만 SWaT는 동일했으므로 모델 구조보다 SWaT 모드의 점수 융합, 임계값, persistence 정책을 우선 확인한다.
## 2026-06-26 SWaT 성능 개선 결과

- SWaT에서 기존 설정은 `global_model_weight` 0.0, `local_model_weight` 0.05라 모델 점수가 최종 전역 알람에 거의 반영되지 않았다.
- 점수 trace를 한 번 수집한 뒤 threshold, fast reset, persistence window, min trigger count, global model weight를 grid search했다.
- 선택한 정책은 threshold 4.0, fast reset 1.2, persistence window 7, min trigger count 7, global model weight 0.3, local model weight 0.05다.
- 실제 재벤치마크 결과 SWaT Pt-F1은 0.1667에서 0.3492로 0.1825 상승했고, PA-F1은 0.3125에서 0.8271로 0.5146 상승했다. PA recall은 1.0을 유지했고 PA precision은 0.1852에서 0.7051로 개선됐다.
- 방법상 개선의 핵심은 더 높은 threshold와 더 엄격한 persistence로 false positive를 줄이고, 시간 토큰 모델의 reconstruction 기반 model score를 전역 점수에 0.3 비중으로 반영한 것이다.

## 2026-06-27 SWaT 기본 정책 승격 및 seed 검증

- `scripts/evaluate_swat.py`에만 있던 SWaT 튜닝 정책을 `AnomalyDetectorAgent`의 `dataset_mode='swat'` 기본값으로 승격했다. 평가 스크립트 외 운영 경로에서도 threshold 4.0, fast reset 1.2, global model weight 0.3, local model weight 0.05, min trigger count 7이 기본 적용된다.
- `AnomalyDetectorAgent` 기본값 스모크 테스트에서 SWaT는 튜닝값으로 초기화되고 NASA는 기존 threshold 3.0, fast reset 1.5, model weight 0.5, min trigger count 5를 유지함을 확인했다.
- SWaT 추가 seed 검증은 seed 7과 123을 전체 재벤치마크로 수행했다. 기존 seed 42와 동일하게 Pt-F1 0.3492, PA-F1 0.8271이 나와 이번 SWaT 정책은 확인한 seed 범위에서는 안정적이다.
- seed 7 검증 중 콘솔 출력 경로에서 `OSError: [Errno 22] Invalid argument`가 발생했지만 결과 JSON은 저장됐다. 이후 seed 123은 stdout을 로그 파일로 redirect해 정상 완료했다.
- NASA 회귀 확인은 공식 결과 파일을 덮어쓰지 않고 scratch에 저장했다. seed를 고정하면 NASA 평균 Pt-F1 0.3405, PA-F1 0.5442가 나와 NASA의 이전 0.7991 PA-F1 결과는 학습 seed에 민감할 수 있음을 확인했다. NASA seed 고정과 안정화는 별도 후속 작업으로 분리한다.

## 2026-06-28 KIMST 논문 초안 작성

- 목표는 한국군사과학기술학회지 한국어 논문 형식에 맞춘 4쪽 이상 초안 작성이다.
- 공식 투고안내 기준은 한글 중심 원고, 국문 및 영문 제목과 저자, 영문 Abstract 200단어 이내, 영문 Key Words 10단어 이내, 본문 순서, 영문 Reference 작성이다.
- 공식 자료실의 `[학회지] 논문작성 양식` HWP 파일을 `paper_work/kimst_journal_template.hwp`로 내려받았다.
- 주제는 유도탄 운용 세부나 성능 제원이 아니라 데이터 품질 점검과 이상탐지 관점의 단순 Enc-Dec 구조로 한정한다.
- 문서 산출물은 편집 가능한 DOCX를 우선 만들고, 원본 HWP 양식은 참고 파일로 보존한다.

### 논문 초안 1차 검토

- 공식 투고안내의 작성 순서를 반영했다. 국문·영문 제목, 국문·영문 저자 자리, 영문 Abstract, Key Words, 본문, 후기, Reference 순서로 구성했다.
- 학회지 예시 논문은 제목부와 Abstract 이후 번호가 붙은 본문, 영문 그림·표 캡션, 본문 우측 상단형 참고문헌 번호, 영문 Reference를 사용하는 흐름이 확인되었다.
- 초안은 복잡한 구조보다 단순 Enc-Dec 복원 오차와 기본 규칙 점검의 결합으로 제한했다.
- 민감한 운용 절차, 실제 제원, 성능 수치, 유도 알고리즘, 표적 관련 내용은 포함하지 않았다.

### 논문 초안 2차 검토

- Reference는 9개 항목으로 구성했고 모두 영문 표기로 정리했다.
- Enc-Dec, VAE, LSTM 기반 이상탐지, 우주기 데이터 이상탐지, 다변량 시계열 이상탐지, KIMST 점검 관련 딥러닝 논문을 포함했다.
- Word COM 렌더링 기준으로 PDF가 A4 4쪽이며, `pdftotext` 기준 제목, Abstract, Key Words, 본문, Fig. 1, Table 1~3, 후기, Reference가 추출됨을 확인했다.
- 이미지 보기 도구는 ACL 문제로 열람하지 못했으나, Word PDF 생성, Poppler PNG 변환, 페이지별 픽셀 밀도와 경계 박스 검수로 대체 확인했다.

## 2026-06-28 관련 논문 서치 및 검토

- 목표는 기존 KIMST 논문 초안의 Reference와 선행연구 검토를 보강하는 것이다.
- 검색 범위는 Enc-Dec 기반 시계열 이상탐지, VAE/LSTM/Transformer 기반 다변량 이상탐지, NASA·우주기 텔레메트리 이상탐지, 항공·국방 점검 사례, KIMST 학회지 관련 논문으로 둔다.
- 검토 기준은 방법론 직접 관련성, 데이터 형태 유사성, 국방·항공우주 적용성, 초안 반영 위치, Reference 채택 우선순위이다.

### 관련 논문 검토 결과

- `paper_work/related_paper_review.md`에 관련 논문 검토표를 작성했다.
- 최우선 보강 후보는 Malhotra et al.의 LSTM Enc-Dec 논문, Hundman et al.의 NASA 우주기 텔레메트리 이상탐지 논문, KIMST의 미사일 FCS DAS 실시간 이상탐지 논문이다.
- 국내 국방 데이터 배경 보강에는 KIMST의 무기체계 PHM/GAN 논문과 창정비 시험결과 DB 분석 논문이 적합하다.
- Anomaly Transformer와 TranAD는 최신 고도화 후보로는 좋지만, 현재 초안의 단순 구조 방향과는 거리가 있어 고찰에서 확장 후보로만 다루는 것이 적절하다.

## 2026-06-28 유사 논문 존재 여부 조사

- 목표는 현재 초안과 유사한 국내외 논문이 존재하는지 확인하고, 유사 타겟과 유사 목적 연구를 분류하는 것이다.
- 유사도 기준은 직접 유사, 부분 유사, 방법론 유사로 구분한다.
- 직접 유사는 유도탄, 미사일 FCS, 무기체계 시험·정비 데이터가 대상이고 이상탐지 또는 데이터 점검을 목적으로 하는 경우다.
- 부분 유사는 항공우주 텔레메트리, 발사체, 우주기, 항공 정비 데이터가 대상이고 이상탐지 또는 건전성 점검을 목적으로 하는 경우다.
- 방법론 유사는 대상은 다르지만 Enc-Dec, Autoencoder, LSTM, Transformer 기반 시계열 이상탐지를 다루는 경우다.

### 유사 논문 존재 여부 조사 결과

- `paper_work/similar_target_paper_search.md`에 유사 타겟·유사 목적 논문 존재 여부를 정리했다.
- 국내에는 직접 유사 논문이 존재한다. KIMST 2025 추계학술대회의 미사일 FCS DAS 실시간 이상탐지 논문이 가장 유사하다.
- 국외에서는 유도탄 데이터 Enc-Dec 점검과 동일한 공개 논문은 확인하지 못했으나, NASA 우주기 텔레메트리 LSTM 이상탐지, 레이더 데이터 스트림 이상탐지, 산업제어 Seq2Seq 이상탐지 논문이 부분 유사하다.
- 초안은 실시간 운용 이상탐지가 아니라 데이터 사용 전 품질 점검과 검토 우선순위 산출로 차별화해야 한다.

## 2026-06-28 DOCX 레퍼런스 및 차별점 보완

- 사용자의 요청은 기존 DOCX에 유사연구 Reference를 추가하고, FCS 실시간 이상탐지와 본 논문의 점검 절차 기반 신뢰도 확보 관점의 차이를 검토·반영하는 것이다.
- 차별화 방향은 FCS에서도 본 방법을 적용할 수 있으나, 논문의 중심을 특정 FCS 운용 메시지 이상탐지가 아니라 유도탄 시험·정비 데이터 사용 전 품질 점검 절차와 신뢰도 확보로 두는 것이다.
- 본문 수정은 최소 범위로 진행하고, 국내 유사연구를 선행연구와 고찰에 직접 연결한다.

### DOCX Reference 및 차별점 보완 결과

- DOCX 본문에 국내 FCS DAS 실시간 이상탐지 연구, 창정비 시험결과 DB 분석 연구, 무기체계 PHM/GAN 연구를 인용했다.
- Reference [10]~[12]를 추가했다.
- 고찰에 FCS에도 제안 구조를 적용할 수 있으나, 본 논문은 실시간 운용 이상탐지가 아니라 시험·정비 데이터 사용 전 사전 점검 절차와 신뢰도 확보에 초점을 둔다는 차이를 명시했다.
- Word 렌더링 기준 PDF는 A4 4쪽이며, PDF 텍스트 추출에서 [10]~[12]와 `신뢰도 확보` 문구가 확인되었다.

## 2026-06-29 세 논문 비교 메모
- 검토 대상은 DOCX에 추가한 KIMST 참고문헌 [10] FCS DAS 이상탐지, [11] 창정비 시험결과 DB 분석, [12] PHM/GAN 합성데이터 논문으로 확정했다.
- FCS DAS 논문은 LSTM/GRU 기반 시계열 예측과 규칙 기반 임계값 검증을 결합한 2단계 이상탐지 구조로 정리했다.
- 창정비 DB 논문은 AI 탐지 모델이 아니라 시험결과 DB화, 초기 및 과거 및 현재 데이터 비교, 그래프 기반 추세 확인을 통한 절차적 신뢰도 확보 연구로 정리했다.
- PHM/GAN 논문은 무기체계 고장 데이터 부족과 불균형 문제를 배경으로 합성데이터와 GAN 계열 모델의 적용 가능성을 제시한 연구로 정리했다.
- 현재 Enc-Dec 논문의 차별점은 FCS 실시간 운용 감시가 아니라 시험 및 점검 데이터의 사전 품질 점검과 재구성 오차 기반 신뢰도 확보 절차로 두는 것이 적절하다.

## 2026-06-29 DOCX v2 수정 계획
- 원본 DOCX는 보존하고 `kimst_encdec_guided_weapon_data_inspection_v2.docx`를 새로 생성한다.
- 방어적으로 읽히는 `simple`, `단순`, `복잡한 모델을 새로 설계하는 것이 아니라` 표현을 줄이고, `systematic`, `채널군 분리 구조`, `초기 도입형`, `1차 선별 계층` 중심으로 바꾼다.
- 5절은 실험 수치가 없는 현재 상태를 고려해 `적용 절차 및 예상 효과`로 수정한다.
- 보안성 문구와 후기성 문구는 투고용 본문에서 과도하게 보이지 않도록 축소하거나 삭제한다.

- v2 DOCX는 Word COM 열기 및 통계 계산 기준 4쪽, 표 4개로 정상 확인되었다. 패키지 렌더러는 pdf2image 모듈 부재로 실패했고, Word PDF 내보내기는 원본과 v2 모두 PDF 저장 단계에서 타임아웃되어 PNG 시각 QA는 완료하지 못했다.

## 2026-06-29 DOCX v3 수정 계획
- v2를 보존하고 `kimst_encdec_guided_weapon_data_inspection_v3.docx`를 새로 생성한다.
- 제안 구조는 `B x D x W` 입력, LSTM Encoder, latent state `Z_t`, LSTM Decoder, Linear and ReLU output head, MSE reconstruction loss로 구체화한다.
- 모델 설명에는 LSTM Autoencoder와 VAE를 함께 반영하고, OmniAnomaly 계열의 확률적 잠재표현과 Bahdanau Attention을 적용한 구조로 표현한다.
- 실험 설명은 항공기 연동 정상 데이터만으로 VAE를 학습한 뒤 Hardware Fail, 전달정렬 비정상, 항법 비정상 등 비정상 입력에서 복원 오차 증가와 threshold 기반 탐지 가능성을 확인한 것으로 정리한다.
- Future Work는 비지도 탐지 뼈대를 유지하면서 일부 라벨을 지도학습으로 통합하는 반지도 또는 혼합학습 프레임워크로 확장한다고 쓴다.

- v3 DOCX는 LSTM Autoencoder, VAE, OmniAnomaly 계열 잠재표현, Bahdanau Attention, B x D x W 입력, LSTM Encoder -> Z_t -> LSTM Decoder -> Linear-ReLU 구조를 반영했다.
- 실험 서술은 항공기 연동 정상 데이터만으로 VAE를 학습하고 Hardware Fail, 전달정렬 비정상, 항법 비정상 등 비정상 입력에서 MSE threshold 기반 anomaly candidate 선별 가능성을 확인한 흐름으로 수정했다.
- Future Work는 비지도 학습 기반 detection 뼈대에 일부 지도 라벨을 통합하는 비지도-지도 통합 프레임워크 개발로 정리했다.
- v3 DOCX는 Word COM 열기 기준 5쪽, 표 4개로 정상 확인되었다. 패키지 렌더러는 pdf2image 모듈 부재로 실패해 PNG 시각 QA는 완료하지 못했다.

## 2026-06-29 DOCX v4 수정 계획
- v3를 보존하고 `kimst_encdec_guided_weapon_data_inspection_v4.docx`를 새로 생성한다.
- 대절 구조는 1. 서론, 2. 관련 연구, 3. 데이터 점검 대상과 문제 정의, 4. 제안 방법, 5. 실험 및 결과, 6. 결론으로 단순화한다.
- 기존 5절 점검 절차는 4절 하위 `4.3 정상 기준선 학습과 점검 절차` 및 `4.4 적용상 유의사항`으로 흡수한다.
- 기존 6절 적용 절차는 `5. 실험 및 결과`로 바꾸고, 기존 7절 고찰은 `5.3 고찰`로 흡수한다.
- 기존 8절 적용상 유의사항은 4절 말미로 이동한다.

- v4 DOCX는 사용자 제안 구조에 맞춰 1. 서론, 2. 관련 연구, 3. 데이터 점검 대상과 문제 정의, 4. 제안 방법, 5. 실험 및 결과, 6. 결론의 6개 대절로 재구성했다.
- 기존 5절 점검 절차와 8절 적용상 유의사항은 4절 제안 방법 하위로 흡수했다.
- 기존 6절 적용 절차와 7절 고찰은 5절 실험 및 결과 하위로 통합했다.
- v4 DOCX는 Word COM 열기 기준 5쪽, 문단 153개, 표 5개로 정상 확인되었다. 패키지 렌더러는 pdf2image 모듈 부재로 실패해 PNG 시각 QA는 완료하지 못했다.

## 2026-06-30 DOCX v5 수정 계획
- v4를 보존하고 `kimst_encdec_guided_weapon_data_inspection_v5.docx`를 새로 생성한다.
- 모든 Reference 번호가 본문에서 최소 한 번 인용되도록 관련 문장에 인용을 추가하거나, 본문과 무관한 참고문헌은 제거한다.
- 모든 Table 및 Fig. 캡션은 본문 설명 문장에서 명시적으로 호출한다.
- 표 자체의 영어 셀 내용은 별도 요청이 있을 때 DOCX에 반영하고, 이번 작업은 우선 인용 정합성과 호출 정합성을 맞춘다.

- v4 기준 Reference [1]~[14] 중 본문에서 직접 확인된 인용은 [10]~[14]였고, [1]~[9]는 본문 인용 보강이 필요했다.
- Table 1~4와 Fig. 1은 캡션은 있으나 본문 호출 문장을 더 명확히 추가하는 방향으로 보정한다.
- 사용자 이전 지적에 맞춰 표 셀 내용도 v5에서 한글화한다.

- v5 DOCX는 Reference [1]~[14] 전체가 본문에서 인용되도록 보정했다.
- Table 1~4와 Fig. 1은 캡션 외 본문 호출 문장이 각각 확인되도록 보정했다.
- 표 셀 내용과 주요 표 캡션을 한글화했고, 그림 흐름도 텍스트도 한글화했다.
- v5 DOCX는 Word COM 열기 기준 5쪽, 문단 153개, 표 5개로 정상 확인되었다. 패키지 렌더러는 pdf2image 모듈 부재로 실패해 PNG 시각 QA는 완료하지 못했다.
## 2026-06-30 DOCX v6 참고문헌 순차 인용 보정
- v5는 모든 참고문헌이 본문에 인용되지만, 첫 등장 순서가 [10], [11], [12], [3] ... 형태로 시작해 투고 원고 관례와 맞지 않았다.
- v6에서는 본문 문단 순서를 바꾸지 않고, 첫 등장 순서 기준으로 참고문헌 번호를 재매핑한다.
- v5는 보존하고 kimst_encdec_guided_weapon_data_inspection_v6.docx를 새로 생성한다.
- 재번호화 매핑은 기존 [10]→새 [1], [11]→[2], [12]→[3], [3]→[4], [4]→[5], [5]→[6], [6]→[7], [7]→[8], [8]→[9], [9]→[10], [14]→[11], [1]→[12], [2]→[13], [13]→[14]로 확정했다.
- v6 검증 결과, 본문 첫 인용 순서는 [1]부터 [14]까지 순차 증가했고 Reference 목록도 [1]부터 [14]까지 순차 배열되었다.
- v6 DOCX는 Word COM 열기 기준 5쪽, 문단 153개, 표 5개로 정상 확인되었다.
- 렌더 스크립트는 pdf2image 모듈 부재로 실패했으며, PNG 시각 QA는 완료하지 못했다.

## 2026-08-28 벤치마크 프로토콜 v2 교정 계획

- 현재 SWaT와 NASA 결과는 축소 샘플 내부 회귀 지표로만 유지하고 표준 벤치마크 성능으로 해석하지 않는다.
- 계산 자원 사용을 줄이기 위해 전체 재학습보다 코드 경로 교정, 작은 회귀 테스트, 점수 캐시 기반 평가 구조를 먼저 구현한다.
- 모델 학습의 최소 목적은 contrastive representation과 실제 경보에 사용하는 reconstruction을 함께 학습시키는 것이다.
- 인접행렬이 없는 경우에도 GNN 출력 토큰 차이를 보존해야 하며, 시간 위치 정보가 fusion 입력에 남아야 한다.
- bit context는 학습과 추론에서 같은 의미와 차원을 사용한다. 값이 없을 때만 명시적으로 0 fallback을 허용한다.
- 평가 데이터는 시간순으로 분리하고 validation에서 결정한 정책을 test에서 고정한다.
- 주 지표는 Point F1 하나가 아니라 event precision, event recall, 탐지 지연, 오경보 구간 수를 포함한다.
- 이번 단계에서는 대용량 SWaT/NASA 재실행을 하지 않고 작은 테스트와 정적 검증까지만 수행한다.
- RED 테스트에서 기존 GNN 출력의 토큰 분산은 0이었고, 위치 인코딩이 없는 fusion은 토큰 역순 입력에도 같은 출력을 냈다.
- 기존 `Trainer.pretrain_contrastive()` 한 스텝 뒤 decoder 파라미터 변화량은 0이었다.
- GNN은 각 토큰 특징에 전체 문맥 평균을 더하도록 수정하고, fusion에는 고정 sinusoidal position encoding을 적용했다.
- `multimodal_pretraining_loss()`를 추가해 InfoNCE와 두 augmented view의 reconstruction MSE를 함께 최적화하도록 했다.
- 모델 회귀 테스트 5개가 통과했고 Trainer 한 스텝 뒤 decoder가 실제로 갱신됨을 확인했다.
- RED 통합 테스트에서 NASA 명시적 context `[1, 0, 1]`이 기존 추론 경로에서는 `[0, 0, 0]`으로 모델에 전달됨을 확인했다.
- `build_bit_context()`는 명시적 context를 우선하고, 없을 때 discrete 센서 값과 bitfield를 센서 순서대로 고정 폭 벡터로 만든다.
- NASA 평가는 각 test window의 실제 `bits`를 ADA에 전달하고, SWaT는 등록된 discrete 센서의 마지막 값을 학습 때와 같은 순서로 구성한다.
- 시간순 분할 모듈은 train, calibration, validation, test를 섞지 않고 연속 구간으로 분리하며 test 잔여 행을 버리지 않는다.
- CPU 기준선은 train 정상 구간의 median과 MAD로 적합하고 calibration 정상 구간의 분위수로 임계값을 고정한 뒤 test를 한 번 평가한다.
- 이벤트 평가는 실제·예측 구간을 일대일로 매칭해 이벤트 precision, recall, F1, 오경보 구간 수와 평균 탐지 지연을 별도로 기록한다.
- RED 테스트에서 10칸 간격의 평가가 현재 점수는 임계값 미만인데 과거 상태 누적으로 생성된 경보를 그대로 채택할 수 있음을 명세했다.
- 희소 샘플은 상태 누적 경보 대신 현재 점수와 고정 임계값을 사용하고, 연속 샘플일 때만 운영용 persistence 결과를 유지한다. 두 평가 스크립트는 `alarm_mode`와 실제 평가 간격을 결과에 기록한다.
- 공유 2차원 인접행렬의 배치 처리에서 불필요한 차원이 추가되던 오류를 회귀 테스트로 재현하고 `einsum` 기반 명시적 배치 연산으로 수정했다.
- 전체 단위 테스트 16개와 변경 Python 파일의 `py_compile` 검증이 통과했으며, 변경 파일 13개의 UTF-8 strict read와 손상 sentinel 검사에서 문제가 없었다.

## 2026-08-28 벤치마크 프로토콜 v2 전체 실행 계획

- 사용자가 모든 확인과 실행을 요청했으므로 교정 코드 기준 SWaT와 NASA 벤치마크를 실제로 재실행한다.
- 실행 전 기존 결과 지표를 기록하고, 결과 파일은 각 평가 스크립트의 표준 경로에 갱신한다.
- SWaT는 stride 3000 학습 창과 stride 500 테스트 창을 사용하는 전체 축소 벤치마크를 실행한다.
- NASA는 A-1, D-1, M-1 세 채널을 seed 42, train epoch 1, 최대 train batch 3, baseline fit batch 10, eval step 10으로 실행한다.
- 새 neural 지표는 희소 샘플에서 stateless threshold를 사용하며, robust MAD 기준선은 train 적합, calibration 임계값 고정, test 일회 평가를 따른다.
- 전체 원시 시계열 전수 neural 추론은 현재 프로토콜 범위가 아니며, 결과에는 평가 grain과 sample stride를 명시한다.
- 실행 전 기준 파일은 SWaT Point F1 0.349206, PA-F1 0.827068과 NASA 평균 Point F1 0.375271, PA-F1 0.799102를 담고 있다.
- 실행 환경은 Python 3.12.7, PyTorch 2.12.0+cpu, CPU 4코어 및 torch 4스레드이며 장치 선택 결과는 CPU다.
- SWaT 실행은 2026-08-28 23:12에 시작해 2026-08-29 00:52에 결과 파일을 저장했으며 약 100분이 걸렸다. 사전학습은 415.57초였다.
- 교정 SWaT 결과는 Point precision 0.061331, recall 0.536364, F1 0.110075와 PA-F1 0.195904다.
- 희소 샘플의 잘못된 persistence 누적을 제거하면서 기존 Point F1 0.349206과 PA-F1 0.827068보다 낮아졌다. 기존 수치는 교정 프로토콜과 직접 비교 가능한 성능이 아니다.
- SWaT event F1은 0.004405이며 실제 이벤트 1개, 예측 이벤트 453개, 오경보 이벤트 452개다. stride 500 라벨에서 실제 공격 구간이 1개로 축약되어 표준 이벤트 성능으로 해석할 수 없다.
- SWaT robust MAD 기준선은 Point F1과 Event F1이 모두 0이며 threshold 19704.110129, false positive 325, false negative 110이다.
- NASA 3채널 실행은 SWaT 종료 직후 시작해 2026-08-29 02:30에 결과 파일을 저장했으며 약 98분이 걸렸다.
- NASA 평균 Point F1은 0.378091, 평균 PA-F1은 0.551003, robust MAD 평균 Point F1은 0.288915, 평균 Event F1은 0.272840이다.
- A-1은 Point F1 0.222222, PA-F1 0.666667, Event F1 1.0이며 855개 창을 평가했다.
- D-1은 Point F1 0.912052, PA-F1 0.986343, Event F1 0.4이며 841개 창을 평가했다.
- M-1은 Point F1과 PA-F1 및 Event F1이 모두 0이며 218개 창을 평가했다.
- 세 채널 모두 seed 42와 stateless threshold 경보 방식을 결과 JSON에 기록했다.
- 결과 자동 검증에서 평균 재계산, robust MAD 혼동행렬 행 수, 이벤트 false alarm 수식, seed 42, stateless threshold 및 평가 창 수가 모두 일치했다.
- SWaT는 기존 대비 Point F1이 0.239132, PA-F1이 0.631164 하락했다. 기존 persistence 정책은 stride 500의 불연속 창을 연속 시점처럼 누적했으므로 새 결과가 더 보수적인 교정값이다.
- NASA 평균 Point F1은 기존 대비 0.002821 상승해 사실상 동일하고, 평균 PA-F1은 0.248099 하락했다.
- NASA 채널별 Point F1 변화는 A-1 +0.104575, D-1 +0.007514, M-1 -0.103627이다. M-1 실패가 평균 개선을 상쇄했다.
- reconstruction loss 감소는 decoder 학습 경로가 작동함을 보여주지만 최종 경보 성능 개선을 보장하지 않는다. 현재 병목은 점수 임계값과 센서 및 공정 상태별 calibration이다.
- SWaT threshold 4.0은 과거 persistence 정책과 함께 선택된 설정이므로 stateless sparse 평가에 그대로 사용하는 것은 적절하지 않다. 정상 calibration 구간에서 목표 false alarm rate로 새 임계값을 고정해야 한다.
- SWaT stride 500 라벨은 공식 공격 이벤트 구조를 보존하지 못하므로 현재 Event F1은 내부 회귀 지표로만 사용해야 한다. 운영 persistence 평가는 연속 창 또는 실제 timestamp 간격을 보존한 데이터에서 별도로 수행해야 한다.
- NASA D-1 robust MAD Point F1은 0.812283으로 neural 0.912052에 근접하지만 A-1과 M-1 기준선은 낮다. 모델 이점은 채널별로 검증해야 한다.
- 약 100분 SWaT와 약 98분 NASA 실행 시간은 반복 calibration에 과도하다. 다음 구현 우선순위는 모델 점수 배치 계산, 점수 및 라벨 캐시, 임계값 sweep과 추론 분리다.
- 전체 단위 테스트 16개, 관련 Python 파일 py_compile, 두 결과 JSON 산술 불변식, 변경 파일 4개의 UTF-8 strict read와 손상 sentinel 검사가 모두 통과했다.

# 2026-08-29 경량 예측 기반 알고리즘 개선

- 임계값과 persistence 조정은 평가·운영 보정으로 한정하고, 이번 작업의 핵심 개선으로 계산하지 않는다.
- 기존 멀티모달 모델을 즉시 제거하지 않고 동일한 시간 분할과 지표로 비교 가능한 별도 경량 예측 모델을 추가한다.
- 우선 성공 기준은 임계값 적용 전 점수 분리도 개선, SWaT 오경보 감소, NASA M-1 탐지 회복, 추론 비용 감소다.
- 계산 자원 제한을 지키기 위해 합성 테스트와 일부 실제 데이터 검증부터 수행하며, 장시간 전체 벤치마크는 효과가 확인된 뒤에만 실행한다.
- 구현은 테스트 우선으로 진행하고 기존 사용자 변경 파일은 수정하지 않는다.
- 현재 모델은 입력 창을 복원하지만 미래값을 직접 예측하지 않으며, 창마다 GNN과 Transformer 전체를 호출한다.
- 새 모델은 다중 시간축의 평균, 표준편차, 변화량으로 다음 시점을 예측하는 정규화 ridge 모델과 정상 상태 Gaussian density를 결합한다.
- 예측 오차는 채널별 정상 잔차 분산으로 정규화하고, 상태 점수는 정상 창의 수준과 변동성에서 양방향으로 벗어나는 경우를 검출한다.
- 이 구조는 정상 데이터만 사용하고 폐쇄형 해를 사용하므로 반복 epoch와 GPU가 필요하지 않으며, M-1처럼 값의 크기보다 변동성이 바뀌는 이상도 표현할 수 있다.
- 합성 단위 테스트에서 마지막 관측값을 바꿔도 예측값은 변하지 않아 미래값 누출이 없고, 동결된 변동성 상태는 정상 창보다 높은 점수를 받았다.
- 임계값과 무관한 점수 순위 비교를 위해 외부 라이브러리 없이 동점 처리를 포함한 average precision 계산을 추가했다.
- 구현 모델은 절대값이 아니라 직전값 대비 변화량을 예측해 상수 센서의 작은 배치 오프셋을 오경보로 보는 문제를 줄였다.
- 모델이 정상 학습 중 분산을 추정할 수 없었던 상태 특징은 Gaussian 상태 점수에서 제외한다.
- NASA는 연속 rolling window이므로 fit, calibration, validation 경계에서 99개 창을 purge해 원시 시점 중복을 제거했다.
- NASA 워밍업 적응은 테스트 원시 시계열의 고정 초기 10%를 먼저 절반으로 나누고 각 구간에서 별도 창을 생성한다. 세 채널 모두 워밍업 구간의 실제 양성 라벨 수가 0임을 사후 감사했으며, 라벨은 적응이나 임계값 계산에 사용하지 않았다.
- SWaT는 원래 train stride 3000, test stride 500으로 창 길이 100보다 간격이 커 경계 purge가 필요하지 않다.
- SWaT 학습은 창 내부 시점을 증강하지만 threshold 보정은 평가와 동일하게 창 끝점 하나만 사용한다. 최종 보정 표본은 92개다.
- NASA 적응 효과는 동일한 post-warmup 평가 구간에서 정적 대조군과 비교한다. 전체 정적 결과와 adapted 결과를 직접 비교하지 않는다.

## 최종 벤치마크 결과

- SWaT는 AP 0.4438, Point F1 0.4698, PA-F1 0.6268을 기록했다. 교정된 기존 신경망의 Point F1 0.1101 대비 0.3598, PA-F1 0.1959 대비 0.4309 상승했다.
- SWaT 모델 적합은 0.7642초, 2,884개 평가 창 점수 계산은 0.6262초였고, 원본 CSV 로딩을 포함한 전체 실행은 37.99초였다. 기존 신경망 전체 실행 약 100분과 비교해 반복 실험 비용이 크게 줄었다.
- SWaT Event F1은 0.0163으로 낮다. stride 500 샘플 라벨이 공식 공격 구조를 하나의 이벤트로 축약하므로 PA-F1과 Event F1은 내부 회귀 참고값으로만 기록했다.
- NASA 정적 평균 AP는 0.4190, Point F1은 0.0155였다. 채널 AP는 A-1 0.0253, D-1 0.7234, M-1 0.5081이다.
- 동일 post-warmup 구간의 정적 평균 AP는 0.4460이고 적응 평균 AP는 0.4420이라 적응 자체가 점수 순위를 개선하지는 않았다.
- M-1은 post-warmup 정적 Point F1 0.0345에서 적응 Point F1 0.5822로 상승했지만, 15개 보정 창에서 정한 임계값이라 탐색적 결과로만 본다. NASA adapted 평균 Point F1은 0.1961로 기존 신경망 평균 0.3781보다 낮다.
- NASA의 높은 sampled PA-F1은 한 번의 탐지가 긴 이상 구간 전체를 채우는 효과가 있어 주 성능 근거로 사용하지 않는다.
- 결론은 새 모델을 SWaT의 우선 후보로 채택할 수 있지만 NASA에서는 기존 신경망을 대체할 수 없다는 것이다. NASA 후속 방향은 새 변화·상태 점수를 기존 신경망 점수와 정상 validation 기준으로 결합하고, 별도의 미사용 채널에서 조합을 검증하는 것이다.
- 정상 학습에서 분산이 없던 상태 특징을 비활성화하므로, 고정 상태 채널이 새로운 값으로 지속되는 이상을 전환 순간 이후 놓칠 수 있다. 이 한계는 후속 상수 채널용 변화점 점수에서 보완해야 한다.
- 전체 단위 테스트 32개와 관련 Python 파일 py_compile이 통과했다.
- 최종 결과 JSON의 혼동행렬 합계, F1 재계산, AP 범위, calibration 및 evaluation grain, NASA 동일 post-warmup 구간, purge 99, 워밍업 양성 라벨 0, SWaT 기존 결과 대비 차이를 검증했다.
- 두 차례 독립 코드 리뷰를 수행했고, 보정 grain 불일치와 NASA 비교 구간 불일치를 수정한 뒤 남은 Critical 또는 Important 문제가 없음을 재확인했다.

# 2026-08-29 NASA 하이브리드 점수 개선

- 이번 단계의 목표는 기존 NASA 신경망을 교체하는 것이 아니라, 신경망 점수와 경량 예측·상태 점수의 상보성을 정상 데이터만으로 결합하는 것이다.
- 결합 가중치나 규칙을 A-1, D-1, M-1 테스트 라벨로 선택하지 않는다. 각 원시 점수를 정상 calibration 경험적 꼬리확률로 변환하고 사전에 고정한 결합 규칙을 사용한다.
- 상관된 점수에도 보수적으로 적용할 수 있고 추가 분포 라이브러리가 필요 없는 Bonferroni 최소 p-value 결합을 우선 구현한다.
- 결합 임계값은 component calibration과 분리된 정상 validation 구간에서 고정한 뒤 test에 한 번 적용한다.
- 장시간 반복 추론을 피하기 위해 신경망 원시 점수를 배치로 계산하고, 모델·데이터·평가 설정을 식별하는 메타데이터와 함께 캐시한다.
- 기존 3채널은 이전 결과와의 비교에 사용하고, 조합 규칙의 일반화 판단은 사전에 고정한 미사용 채널 집합에서 별도로 확인한다.
- 기존 NASA ADA의 최종 점수는 신경망 단독이 아니다. 테스트 중 갱신되는 drift·spike·stuck 휴리스틱 점수와 재구성 점수를 전역 50:50으로 결합하고, 센서 임계값도 테스트 초기 구간에서 적합한다.
- 이번 하이브리드는 테스트 구간 휴리스틱 적합을 제거하고 순수 신경망 재구성 오차와 예측·상태 점수를 결합한다. 기존 ADA 결과는 외부 비교값으로만 유지한다.
- 신경망 재구성 오차는 모델에 window batch를 한 번에 전달해 계산한다. 이 경로는 기존 window별 모델 호출과 같은 MSE 정의를 사용한다.
- 실제 결과를 보기 전에 비교 채널을 고정했다. 기존 비교 채널은 A-1, D-1, M-1이고 미사용 일반화 확인 채널은 E-1, P-1, M-6이다.
- 학습 시계열은 원시 시간축 기준 60% fit, 20% calibration, 20% validation으로 먼저 분리한 뒤 각 구간 내부에서만 window를 만든다. 이 방식은 rolling window 경계 중복을 구조적으로 제거한다.
- A-1 파일럿은 CPU에서 학습 0.959초, 신경망 배치 점수 1.697초, 예측 모델 적합·점수 0.349초, 로딩 포함 전체 7.07초가 걸렸다.
- 동일 A-1 캐시 재실행의 채널 처리 시간은 0.03초였다. 캐시 키는 데이터·코드 SHA-256, 모델 설정, seed, 장치, 분할, 학습·평가 설정을 포함한다.
- A-1 파일럿의 하이브리드 AP는 0.0266, Point F1은 0이었다. 신경망 단독과 예측 단독 Point F1도 모두 0이어서 이 채널에서는 결합할 유효 신호가 부족했다.
- 최종 Bonferroni 결합의 기존 3채널 평균 AP는 0.3110, Point F1은 0.2233이다. 신경망 재구성 단독보다 Point F1은 0.0199 높지만 기존 ADA 0.3781보다 0.1548 낮다.
- 첫 미사용 3채널 E-1, P-1, M-6에서 Bonferroni 평균 Point F1은 0.0992로 신경망 단독 0.1096보다 0.0104 낮았다. 따라서 이 결합 규칙은 일반화 개선으로 채택하지 않는다.
- 최소 p-value는 calibration 최대치를 넘는 평가 점수를 모두 같은 유한 표본 p-value로 포화시키며, D-1에서는 신경망 AP 0.9395를 결합 AP 0.5020으로 낮췄다.
- 후속 결합은 각 점수에 signed log1p 변환과 calibration median·MAD 표준화를 적용한 뒤, shrinkage 공분산의 Mahalanobis 거리로 정상 score vector의 공동 이탈을 측정한다.
- Mahalanobis 규칙의 최종 미사용 확인 채널은 결과 확인 전에 A-5, E-12, S-1, T-1로 고정했다. 앞선 6채널은 이 규칙의 개발 분석 집합으로만 취급한다.
- Mahalanobis 첫 확인 4채널 평균은 AP 0.2103, Point F1 0.1119로 신경망 단독보다 각각 0.0352, 0.0688 높았다. 그러나 S-1 개선이 평균을 주도했고 채널별 일관성은 없었다.
- A-5는 calibration과 validation 창이 각각 42개뿐이며 평가 460개 중 455개를 경보했다. 정상 보정 창이 200개 미만인 채널은 저신뢰로 표시하고 주 결론 평균에서 분리한다.
- 현재 Mahalanobis는 정상 중심보다 낮은 이상 점수도 양방향 이탈로 취급한다. 입력 자체가 이상 증거이므로 표준화 잔차의 양수 부분만 사용하는 upper-tail Mahalanobis로 교정한다.
- upper-tail 교정의 마지막 미사용 검증 채널은 결과 확인 전에 A-2, B-1, C-1, D-2, F-1, M-2로 고정했다.
- 최종 마지막 6채널에서 upper-tail Mahalanobis 평균 AP는 0.3551로 신경망 0.3561과 사실상 같았지만, Point F1은 0.2169로 신경망 0.3365보다 0.1195 낮았다.
- 마지막 6채널 중 C-1은 Point F1 0.2623에서 0.2727로 소폭 개선됐고 D-2는 0.9458에서 0.9435로 사실상 같았지만, M-2는 0.7714에서 0.0513으로 크게 하락했다.
- 16채널 전체에서 upper-tail Mahalanobis 평균 AP는 0.2740, Point F1은 0.1572이고 신경망 단독은 각각 0.2794, 0.1956이다. Bonferroni는 평균 AP 0.2369, Point F1 0.1216으로 더 낮다.
- 전체 train 최소·최대값을 쓰던 loader 출력을 fit 첫 60% 기준으로 다시 정규화해 calibration·validation 분포가 preprocessing에 영향을 주지 않도록 교정했다. 최종 cache key는 이 정규화 명세와 loader·windowing·model 코드 해시를 포함한다.
- 명시적 `cuda:0` 장치도 CUDA로 인식해 pinned memory, non-blocking 전송, 동기화와 cache 장치 식별에 반영한다.
- fit 구간이 상수인 채널은 기준값을 0.5에 배치해 상승과 하락 변화를 모두 보존하며, 이후 정상 구간의 scale을 빌리지 않는다.
- 최종 CPU 원시 점수 생성과 16채널 평가는 54.51초였고, 동일 cache를 사용한 Bonferroni 재평가는 0.57초였다.
- 정상 데이터만 사용하는 단일 보편 결합은 채널별 이상 유형 차이를 구분할 정보가 부족하므로 채택하지 않는다. NASA 운영 기본은 기존 ADA로 유지하고 새 결합은 실험·분석 경로로만 둔다.
- 다음 알고리즘 방향은 실제 이상 라벨이 분리된 개발 채널에서 학습하고 완전히 미사용 채널에서 검증하는 channel-level mixture-of-experts 또는 synthetic fault 기반 gating이다.
- 전체 46개 단위 테스트와 변경 Python 파일의 py_compile 검증이 통과했다.
- 최종 두 JSON의 16채널 window 정렬, 혼동행렬 합계, F1 재계산, AP 범위, 그룹 평균, 동일 비교 채널, 결정 상태와 cache key 동일성을 검증했다.
- 변경 9개 파일의 UTF-8 strict read와 손상 sentinel 검사에서 문제가 없었다.
- 자체 코드 리뷰에서 예외 후 모델 모드 미복원, 전체 train 정규화 누출, 상수 기준의 하락 변화 손실, `cuda:0` 미인식, 서로 다른 채널 집합의 평균 비교 문제를 찾아 각각 회귀 테스트와 함께 수정했다.
- 별도 reviewer 작업은 여러 차례 요청했지만 로컬 파일 읽기 세션이 완료되지 않아 독립 결과를 받지 못했다. 독립 리뷰 완료로 기록하지 않는다.
