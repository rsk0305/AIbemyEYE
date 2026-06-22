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
