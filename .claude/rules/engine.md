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
