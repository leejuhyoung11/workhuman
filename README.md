# Workhuman Promotion Prediction System

LLM 기반 직원 승진 예측 시스템 - LangGraph 워크플로우 비교 실험

## 프로젝트 개요

이 프로젝트는 기업의 employee 데이터(award 메시지 + 직무 이력)를 기반으로 어떤 직원이 VP(또는 상위 직책)로 승진할 가능성이 있는지 LLM 기반 워크플로우로 분석·예측하는 시스템입니다.

### 주요 목표

1. **비정형 텍스트 분석**: award title/message에서 직원의 성과/특징/리더십 등 signal 추출
2. **구조적 패턴 분석**: history 데이터(직무 변경, 매니저 변경 등)에서 승진과 관련 있는 behavioral pattern 추출
3. **워크플로우 비교**: 여러 LangGraph workflow 구조를 비교하여 어떤 workflow가 승진 패턴을 가장 잘 잡아내는지 실험
4. **최종 보고서**: 실험 결과와 best workflow를 논문형 형태로 제시

## 데이터 구조

### A. CONTROL 데이터셋
승진하지 않은 직원 또는 baseline 그룹
- 주요 컬럼: `nom_id`, `title`, `message`, `award_date`

### B. TREATMENT 데이터셋
승진 또는 특정 outcome을 가진 직원 그룹
- 구조는 control과 동일

### C. HISTORY 데이터셋
직원 career progression 정보
- 주요 컬럼: `pk_user`, `fk_direct_manager`, `job_title`, `effective_start_date`, `effective_end_date`

## 실험 구성

### Experiment 1: Single-step LLM vs Prompt Chaining
- **Single-step**: 모든 정보를 한 번에 LLM에 입력하여 판단
- **Prompt Chaining**: 단계별로 분석 후 최종 판단 (award 요약 → history 패턴 추출 → 결합)

### Experiment 2: Prompt Chaining vs Multi-agent Workflow
- **Prompt Chaining**: Linear pipeline
- **Multi-agent**: 역할 분리된 에이전트들 (Award Extractor, History Pattern Agent, Evidence Aggregator, Judge)

### Experiment 3: Sequential vs Parallel Workflow
- **Sequential**: award 분석 후 → history 분석 → 결합
- **Parallel**: award와 history를 동시에 분석 → aggregator에서 통합

### Experiment 4: Router-enabled vs No-router Workflow
- **No-router**: 모든 입력을 하나의 workflow가 처리
- **Router-enabled**: task에 따라 다른 agent로 라우팅

## 설치 및 설정

### 1. 의존성 설치

```bash
uv sync
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:

```bash
OPENAI_API_KEY=your_api_key_here
```

`.env.example` 파일을 참고하세요.

## 사용 방법

### 전체 실험 실행

```bash
uv run python main.py
```

### 개별 실험 실행

Jupyter notebook에서 개별 실험을 실행할 수 있습니다:

```python
from src.experiment_runner import ExperimentRunner

runner = ExperimentRunner(data_dir="data", sample_size=10)
test_data = runner.prepare_test_data()

# Experiment 1 실행
results = runner.run_experiment1(test_data)
```

## 프로젝트 구조

```
workhuman/
├── data/                    # 데이터 파일
│   ├── control copy.csv
│   ├── treatment copy.csv
│   └── wh_history_full.csv
├── src/
│   ├── data_loader.py       # 데이터 로딩 유틸리티
│   ├── evaluator.py         # 평가 프레임워크
│   ├── experiment_runner.py # 실험 실행기
│   └── workflows/           # LangGraph 워크플로우
│       ├── experiment1.py   # Single-step vs Prompt Chaining
│       ├── experiment2.py   # Prompt Chaining vs Multi-agent
│       ├── experiment3.py   # Sequential vs Parallel
│       └── experiment4.py   # Router vs No-router
├── notebook/                # Jupyter notebooks
├── main.py                  # 메인 실행 스크립트
└── pyproject.toml          # 프로젝트 설정
```

## 평가 메트릭

각 워크플로우는 다음 메트릭으로 평가됩니다:
- **Accuracy**: 전체 정확도
- **Precision**: 정밀도
- **Recall**: 재현율
- **F1 Score**: F1 점수
- **Confusion Matrix**: 혼동 행렬

## 결과

실험 결과는 `experiment_results.txt` 파일에 저장됩니다.

## 개발 환경

- Python >= 3.11
- LangGraph >= 1.0.3
- LangChain >= 0.3.0
- OpenAI API

## 라이선스

이 프로젝트는 연구 목적으로 개발되었습니다.
