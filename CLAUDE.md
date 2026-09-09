# FinTwin — Cohort Trajectory Digital Twin

## 프로젝트 개요
해커톤 출품작. 고객의 미래를 예측하는 대신, 유사한 재무 궤적을 걸었던
합성 고객 수천 명의 "실제 결과"를 집계해 보여주는 디지털 트윈.
핵심 서사: "예측하지 않는다. 같은 길을 먼저 걸은 사람들의 결과를 보여준다."

## 기술 스택
- Python 3.11+, pandas, numpy, scikit-learn(유사도 계산만, 학습 없음)
- UI: Streamlit
- LLM 브리핑(선택): Anthropic API

## 디렉토리 구조
- data_gen/        합성 고객 풀 생성 (페르소나 규칙 기반)
- engine/          궤적 매칭 + 결과 집계 + 분기점 분석
- app/             Streamlit 데모 UI
- data/            생성된 CSV (git ignore, 재생성 가능해야 함)
- tests/           핵심 로직 단위 테스트

## 실행 커맨드
- 데이터 생성: `python -m data_gen.generate --n 5000 --months 36 --seed 42`
- 데모 실행: `streamlit run app/main.py`
- 테스트: `pytest tests/ -x -q`
- 린트: `ruff check . && ruff format .`

## 절대 규칙 (해커톤 제약)
1. ML 모델 학습 금지. 매칭은 거리/유사도 계산만. "예측 모델 아님"이 핵심 방어 논리.
2. 모든 난수는 seed 고정. 데모 중 결과가 바뀌면 안 됨.
3. 실제 고객 데이터 절대 사용 금지. 모든 데이터는 data_gen에서 생성.
4. 데모 고객은 customer_id=1001 "김OO" 고정. 이 고객의 스토리가 3분 피치의 축.
5. 하루 안에 완성. 완벽한 추상화보다 동작하는 코드. 파일 하나 300줄 이하 유지.

## 도메인 용어
- 궤적(trajectory): 고객의 월별 재무 스냅샷 시퀀스 (저축률, 지출증가율, DSR)
- 코호트(cohort): 대상 고객과 궤적이 유사한 상위 N명
- 분기점(divergence point): 코호트 내 건전/스트레스 그룹이 갈라진 시점과 변수
- 결과 라벨: HEALTHY / STRESS / DELINQUENT (36개월 시점 기준)
- 상품 필요 라벨(product_need): SAVINGS_PRODUCT(예적금) / CREDIT_LOAN(신용대출) /
  OVERDRAFT(마이너스통장, 일시적 충격형) / CARD_LOAN_RISK(카드론·리볼빙 위험군) /
  NO_PRODUCT_NEEDED. 36개월 궤적 전체(최종 DSR·저축률, DSR 정점, 지출증가율 추세)로
  규칙 기반 판정 (data_gen/personas.py의 PRODUCT_THRESHOLDS). 예측이 아니라
  "이 궤적을 걸은 사람들이 실제로 어떤 상품군을 필요로 했는가"를 코호트 집계로 보여줌.

## 페르소나 5종 (data_gen 규칙)
STABLE(안정형), SLOW_DECLINE(서서히 악화), SHOCK(이벤트 충격: 이직/출산),
RECOVERY(회복형), OVERSPEND(과소비형)
