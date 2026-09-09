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
