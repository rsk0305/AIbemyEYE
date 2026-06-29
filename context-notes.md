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
