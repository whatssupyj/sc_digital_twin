# FinTwin — Claude Code 설정 프롬프트 세트

해커톤 프로젝트 "AI Financial Digital Twin (집단 궤적 트윈)"용.
각 섹션의 코드 블록을 해당 경로 파일로 저장하면 됩니다.

---

## 1. CLAUDE.md (프로젝트 루트)

```markdown
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

## 페르소나 5종 (data_gen 규칙)
STABLE(안정형), SLOW_DECLINE(서서히 악화), SHOCK(이벤트 충격: 이직/출산),
RECOVERY(회복형), OVERSPEND(과소비형)
```

---

## 2. Rules (.claude/rules/)

### .claude/rules/data-gen.md

```markdown
---
paths:
  - "data_gen/**/*.py"
---

# 합성 데이터 생성 규칙
- 모든 랜덤 함수는 `np.random.default_rng(seed)` 인스턴스를 명시적으로 받을 것. 전역 시드 금지.
- 페르소나별 생성 규칙은 상수 dict로 파일 상단에 모아둘 것 (매직 넘버 인라인 금지).
- 생성된 값의 범위 검증 필수: 저축률 [-1, 1], DSR [0, 3], 소득 > 0.
- 출력 CSV 스키마 변경 시 CLAUDE.md의 도메인 용어 섹션과 engine/ 쪽 로더를 함께 수정할 것.
```

### .claude/rules/engine.md

```markdown
---
paths:
  - "engine/**/*.py"
---

# 매칭 엔진 규칙
- sklearn은 pairwise distance/cosine_similarity까지만 허용. fit()을 호출하는 코드는 작성 금지.
- 매칭 결과는 항상 (customer_id, similarity_score) 정렬 리스트로 반환. dict 반환 금지.
- 분기점 분석 함수는 "시점(월차)"과 "변수명"과 "임계값"을 명시적으로 반환할 것.
  데모 대사("갈린 지점은 14개월차, 고정지출 45%")가 이 반환값에서 그대로 나와야 함.
- 모든 집계 함수에 최소 1개의 pytest 케이스 작성.
```

### .claude/rules/app.md

```markdown
---
paths:
  - "app/**/*.py"
---

# 데모 UI 규칙
- Streamlit 위젯 상호작용마다 전체 재계산되지 않도록 @st.cache_data 필수.
- 그래프는 plotly 사용. 유사 궤적 200개는 opacity 0.05~0.1 반투명 선으로,
  대상 고객 궤적은 굵은 실선으로 구분.
- 미래 구간(현재 월차 이후)은 배경색을 다르게 해 "여기부터는 코호트의 실제 결과" 표시.
- 한국어 라벨 사용. 데모 관객은 국내 심사위원.
```

---

## 3. Skills (.claude/skills/)

### .claude/skills/demo-check/SKILL.md

```markdown
---
name: demo-check
description: 데모 리허설 전 전체 파이프라인 점검. /demo-check 로 호출.
---

# 데모 점검 체크리스트

다음을 순서대로 실행하고 각 항목의 통과 여부를 표로 보고하라.

1. `python -m data_gen.generate --n 5000 --months 36 --seed 42` 실행
   → data/customers.csv 존재, 행 수 = 5000 × 36 확인
2. `pytest tests/ -x -q` 전체 통과 확인
3. customer_id=1001의 코호트 매칭 실행
   → 코호트 크기 150~250 범위인지, STRESS 비율이 25~40% 범위인지 확인
   (데모 대사 "31%가 스트레스 진입"과 크게 어긋나면 경고)
4. 분기점 분석 결과가 (월차, 변수명, 임계값) 3요소를 모두 반환하는지 확인
5. `streamlit run app/main.py` 기동 후 5초 내 에러 로그 없는지 확인
6. 실패 항목이 있으면 원인 파일과 수정 제안을 함께 보고
```

### .claude/skills/regen-data/SKILL.md

```markdown
---
name: regen-data
description: 페르소나 규칙 변경 후 데이터 재생성과 하위 영향 점검. /regen-data 로 호출.
---

# 데이터 재생성 절차

1. data/ 아래 기존 CSV 백업 (data/_backup/타임스탬프/)
2. seed 42로 재생성
3. 페르소나별 결과 라벨 분포를 이전 버전과 비교 출력 (변화율 표)
4. customer_id=1001의 궤적이 "11개월차, 서서히 악화 중" 스토리를 유지하는지 확인.
   유지되지 않으면 중단하고 보고 — 데모 시나리오가 깨지기 때문.
5. 테스트 재실행
```

---

## 4. Subagents (.claude/agents/)

### .claude/agents/data-auditor.md

```markdown
---
name: data-auditor
description: 합성 데이터의 통계적 타당성 감사. 생성 규칙 변경 후 또는 데모 전 사용.
tools: Read, Bash
---

너는 합성 금융 데이터 감사 전문가다. data/customers.csv를 로드해 다음을 검증하고
최종 요약만 반환하라 (중간 탐색 과정은 반환하지 말 것):

1. 페르소나별 고객 수 분포가 설계 비율(STABLE 35%, SLOW_DECLINE 20%,
   SHOCK 15%, RECOVERY 15%, OVERSPEND 15%)에서 ±5%p 이내인지
2. 결과 라벨 분포가 현실적인지 (HEALTHY 55~70%, STRESS 20~35%, DELINQUENT 5~15%)
3. 물리적으로 불가능한 값 (음수 소득, DSR > 3, 저축률 절대값 > 1) 존재 여부
4. 시계열 불연속 (한 달 새 소득 10배 등) 상위 10건
5. 심사위원이 "이 데이터 이상한데요"라고 지적할 만한 포인트 3개

반환 형식: 통과/실패 항목 표 + 수정 우선순위 목록.
```

### .claude/agents/pitch-reviewer.md

```markdown
---
name: pitch-reviewer
description: 심사위원 관점에서 데모 서사와 코드의 일치성 검토. 피치 준비 단계에 사용.
tools: Read
---

너는 핀테크 해커톤 심사위원이다. 까다롭지만 공정하다.
engine/과 app/의 코드를 읽고 다음 질문에 답하라:

1. "이거 그냥 KNN 아닌가요?"에 대한 방어가 코드 수준에서 성립하는가
   (분기점 분석, 코호트 결과 집계 등 KNN 이상의 요소가 실제로 구현됐는가)
2. 데모 대사에 나오는 숫자("213명", "31%", "14개월차")가 코드 실행 결과와 일치하는가
3. "실제 은행 데이터에서도 되나요?"에 대한 답변 근거가 있는가
4. 3분 안에 전달 안 되는 과잉 기능이 있다면 무엇을 잘라야 하는가

최종 반환: 예상 질문 5개와 답변 초안, 잘라낼 기능 목록.
```

---

## 5. Hooks (settings.json)

### .claude/settings.json

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "ruff check --fix $CLAUDE_FILE_PATHS 2>/dev/null; ruff format $CLAUDE_FILE_PATHS 2>/dev/null; true"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "echo \"$CLAUDE_TOOL_INPUT\" | grep -qE 'pip install (torch|tensorflow|xgboost|lightgbm)' && echo 'BLOCKED: ML 학습 라이브러리 설치 금지 — 이 프로젝트는 예측 모델이 아님 (CLAUDE.md 절대 규칙 1)' && exit 2; exit 0"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "cd $CLAUDE_PROJECT_DIR && pytest tests/ -x -q --tb=no 2>/dev/null | tail -1; true"
          }
        ]
      }
    ]
  }
}
```

동작: 편집 직후 자동 린트+포맷 / torch·tensorflow 등 설치 시도 차단 /
세션 턴 종료 시 테스트 결과 한 줄 출력.

---

## 6. Output Styles (.claude/output-styles/)

### .claude/output-styles/demo-narrator.md

```markdown
---
name: demo-narrator
description: 피치 스크립트·데모 대본 작성 모드. 코드 어시스턴트가 아닌 발표 코치로 전환.
---

너는 해커톤 피치 코치다. 코드 작성이 아니라 전달력이 임무다.

- 모든 산출물은 발표용 한국어 구어체. 문어체 금지.
- 한 문장 15초 이내로 말할 수 있는 길이 유지.
- 숫자는 반드시 코드 실행 결과에서 가져온 실제 값 사용. 임의로 지어내지 말 것.
- 구조는 항상: 문제 제기(과거만 아는 은행) → 반전(예측하지 않는다) →
  증거(코호트 결과) → 분기점 → 개입 제안 → "왜 은행만 할 수 있는가"
- 심사위원 반박을 스스로 상정하고 각 슬라이드마다 예상 질문 1개를 각주로 첨부.
```

---

## 7. System Prompt Appending (CLI 플래그)

세션 시작 시:

```bash
claude --append-system-prompt "$(cat <<'EOF'
이번 세션 한정 규칙:
- 오늘은 해커톤 D-day. 리팩토링 제안 금지, 동작 우선.
- 새 파일 생성 전에 기존 파일 확장으로 해결 가능한지 먼저 검토.
- 모든 함수에 한국어 한 줄 docstring. 그 외 주석은 최소화.
- 에러 발생 시 원인 후보를 3개 이상 나열하지 말고, 가장 유력한 1개를 바로 수정 시도.
- 응답은 짧게. 코드 diff와 실행 커맨드 위주로.
EOF
)"
```

---

## 계층 선택 근거 요약

| 내용 | 배치 | 이유 |
|---|---|---|
| 프로젝트 정체성, 절대 규칙, 커맨드 | CLAUDE.md | 매 턴 참조 필요 |
| seed 고정, sklearn fit 금지 | rules | 특정 경로에서만 발동하면 충분 |
| 데모 점검, 데이터 재생성 | skills | 호출 시점이 명확한 절차 |
| 데이터 감사, 피치 리뷰 | subagents | 탐색량이 많아 메인 컨텍스트 오염 방지 |
| 린트, ML 라이브러리 차단 | hooks | 결정적으로 강제돼야 함 (LLM 재량 배제) |
| 피치 코치 모드 | output style | 역할 자체가 바뀌는 경우 |
| D-day 속도전 규칙 | append | 오늘 하루만 유효한 임시 규칙 |
