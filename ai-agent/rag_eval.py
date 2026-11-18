"""
RAG 검색 평가 스크립트 (Guide Agent의 search_game_guide 검색 품질 측정)

실행 위치: ai-agent 폴더
실행 방법: python rag_eval.py
- Chroma + Ollama(bge-m3) 임베딩만 사용합니다. Gemini API는 호출하지 않습니다.
- 사전 조건: Ollama 실행 중, python utils/DBupdate.py로 공략 문서가 적재되어 있어야 합니다.

측정 방법
- 검색어마다 정답 청크(문서 파일 + 섹션)를 미리 정해 두고, 전체 청크를 유사도 순으로 정렬해 정답이 몇 등인지 확인합니다.
- Hit@k: 정답이 상위 k개 안에 들어온 검색어 비율
- 동적 k 적중: 실제 서비스처럼 decide_k(검색어)로 정한 k개 안에 정답이 들어왔는지
- MRR: 1 / (정답 순위)의 평균 (1등이면 1.0, 2등이면 0.5)
- 검색어는 Guide Agent의 LLM이 Tool을 호출할 때 만들 법한 형태로 직접 작성했습니다.
"""
from tools.guide import vector_store, decide_k, find_known_name, KNOWN_NAMES

SERKA = "serka.md"
BELGARDIN = "belgardin.md"
OVERVIEW = "개요 및 입장 정보"
SERKA_G1 = "1관문: 고통의 마녀 세르카 주요 패턴 및 기믹"
SERKA_G2 = "2관문: 태초의 그림자 코르부스 툴 라크 주요 패턴 및 기믹"
BEL_G1 = "1관문: 성자 벨가르딘 크로체 주요 패턴 및 기믹"
BEL_G2 = "2관문: 페투스 안 크라그마 주요 패턴 및 기믹"
# 2026-11-19 추가 문서
MORDUM, ARMOCHE, KAZEROS, ARSENOS = "mordum.md", "armoche.md", "kazeros.md", "arsenos.md"
MOR_G1 = "1관문: 카멘 & 인페르나스 주요 패턴 및 기믹"
MOR_G2 = "2관문: 나이트레아 주요 패턴 및 기믹"
MOR_G3 = "3관문: 모르둠 주요 패턴 및 기믹"
ARM_G1 = "1관문: 욕망의 주인 에키드나 주요 패턴 및 기믹"
ARM_G2 = "2관문: 심연의 파수꾼 아르모체 주요 패턴 및 기믹"
KAZ_G1 = "1관문: 심연의 군주 카제로스 주요 패턴 및 기믹"
KAZ_G2 = "2관문: 대악마 카제로스 주요 패턴 및 기믹"
ARS_G1 = "1관문: 대주교 아르세노스 주요 패턴 및 기믹"
ARS_G2 = "2관문: 광신의 인도자 아르세노스 주요 패턴 및 기믹"

# (검색어, 정답 청크 목록[(파일, 섹션)], 유형)
# - 정답이 두 개면 그중 하나만 들어와도 정답 (두 섹션에 모두 근거가 있는 경우)
TEST_CASES = [
    # 세르카 개요
    ("세르카 입장 정보", [(SERKA, OVERVIEW)], "일반"),
    ("세르카 대난투 시스템", [(SERKA, OVERVIEW)], "일반"),
    ("세르카 난이도 종류", [(SERKA, OVERVIEW)], "일반"),
    ("세르카 간파 시스템", [(SERKA, OVERVIEW)], "일반"),
    # 세르카 1관문
    ("세르카 1관문 패턴", [(SERKA, SERKA_G1)], "일반"),
    ("세르카 폭탄 빙고 공략", [(SERKA, SERKA_G1)], "일반"),
    ("세르카 모루 저스트 가드", [(SERKA, SERKA_G1)], "일반"),
    ("거대 망치 장벽 회피", [(SERKA, SERKA_G1)], "일반"),
    ("화염 메이든 빙고", [(SERKA, SERKA_G1)], "일반"),
    # 세르카 2관문
    ("코르부스 툴 라크 진짜 찾기", [(SERKA, SERKA_G2)], "일반"),
    ("세르카 2관문 분신 타임어택", [(SERKA, SERKA_G2)], "일반"),
    ("저저카저 연속 저스트 가드", [(SERKA, SERKA_G2)], "일반"),
    # 벨가르딘 개요
    ("벨가르딘 입장 정보", [(BELGARDIN, OVERVIEW)], "일반"),
    ("벨가르딘 사령의 저주", [(BELGARDIN, OVERVIEW)], "일반"),
    ("벨가르딘 공략 전체 요약", [(BELGARDIN, OVERVIEW)], "일반"),
    # 벨가르딘 1관문
    ("벨가르딘 1관문 패턴", [(BELGARDIN, BEL_G1)], "일반"),
    ("3방향 동시 무력화", [(BELGARDIN, BEL_G1)], "일반"),
    ("벨가르딘 관 레이저 패턴", [(BELGARDIN, BEL_G1)], "일반"),
    # 벨가르딘 2관문
    ("페투스 안 크라그마 패턴", [(BELGARDIN, BEL_G2)], "일반"),
    ("벨가르딘 광신도 처리", [(BELGARDIN, BEL_G2)], "일반"),
    ("흑색 거울 무력화", [(BELGARDIN, BEL_G2)], "일반"),
    # 헷갈리기 쉬운 검색어: 같은 표현이 여러 섹션에 있음
    ("세르카 1관문 240줄", [(SERKA, SERKA_G1)], "혼동"),       # 1관문 240줄 = 모루 저스트 가드
    ("세르카 2관문 240줄", [(SERKA, SERKA_G2)], "혼동"),       # 2관문 240줄 = 진짜 찾기
    ("세르카 240줄 패턴", [(SERKA, SERKA_G1), (SERKA, SERKA_G2)], "혼동"),  # 관문을 안 밝힘
    ("홀딩 저스트 가드", [(BELGARDIN, OVERVIEW), (BELGARDIN, BEL_G1)], "혼동"),  # 시스템 설명 + 1관문 패턴
    ("카운터 타이밍", [(SERKA, SERKA_G1), (SERKA, SERKA_G2), (BELGARDIN, BEL_G2)], "혼동"),  # 레이드를 안 밝힘
    # ---- 2026-11-19 추가 문서 (칠흑 폭풍의 밤 / 파멸의 성채 / 최후의 날 / 지평의 성당) ----
    ("칠흑 폭풍의 밤 입장 레벨", [(MORDUM, OVERVIEW)], "추가 문서"),
    ("카멘 인페르나스 270줄 역능지", [(MORDUM, MOR_G1)], "추가 문서"),
    ("인페르나스 어둠의 안개", [(MORDUM, MOR_G1)], "추가 문서"),
    ("나이트레아 기생충 포식", [(MORDUM, MOR_G2)], "추가 문서"),
    ("나이트레아 엘버하스틱의 힘", [(MORDUM, MOR_G2)], "추가 문서"),
    ("모르둠 슈퍼노바 카운터", [(MORDUM, MOR_G3)], "추가 문서"),
    ("모르둠 중력장 패턴", [(MORDUM, MOR_G3)], "추가 문서"),
    ("파멸의 성채 입장 레벨", [(ARMOCHE, OVERVIEW)], "추가 문서"),
    ("에키드나 분신 장판 회피", [(ARMOCHE, ARM_G1)], "추가 문서"),
    ("아브렐슈드 난입 무력화", [(ARMOCHE, ARM_G1)], "추가 문서"),
    ("아르모체 협동 퍼펙트 가드", [(ARMOCHE, ARM_G2)], "추가 문서"),
    ("아르모체 방패 던지기 회오리", [(ARMOCHE, ARM_G2)], "추가 문서"),
    ("최후의 날 입장 조건", [(KAZEROS, OVERVIEW)], "추가 문서"),
    ("카제로스 심연의 잔상", [(KAZEROS, KAZ_G1)], "추가 문서"),
    ("카제로스 4방향 격돌", [(KAZEROS, KAZ_G1)], "추가 문서"),
    ("대악마 카제로스 콜로세움", [(KAZEROS, KAZ_G2)], "추가 문서"),
    ("카제로스 블랙홀 패턴", [(KAZEROS, KAZ_G2)], "추가 문서"),
    ("지평의 성당 입장 레벨", [(ARSENOS, OVERVIEW)], "추가 문서"),
    ("구원의 종탑 공략", [(ARSENOS, OVERVIEW), (ARSENOS, ARS_G1), (ARSENOS, ARS_G2)], "추가 문서"),
    ("아르세노스 구원의 시간", [(ARSENOS, ARS_G1)], "추가 문서"),
    ("아르세노스 소리굽쇠 부수기", [(ARSENOS, ARS_G2)], "추가 문서"),
    # 문서 간 혼동: 같은 패턴 이름·단어가 다른 레이드에도 있음 (레이드·보스 이름은 밝힘)
    ("에키드나 진짜 찾기", [(ARMOCHE, ARM_G1)], "문서 간 혼동"),          # 세르카 2관문에도 '진짜 찾기'
    ("모르둠 모루 저스트 가드", [(MORDUM, MOR_G3)], "문서 간 혼동"),       # 세르카 1관문에도 '모루 저스트 가드'
    ("아르세노스 심판 큰 종", [(ARSENOS, ARS_G1)], "문서 간 혼동"),        # '심판'은 여러 문서에 나옴
    ("카제로스 언령 삭제", [(KAZEROS, KAZ_G1)], "문서 간 혼동"),           # 2관문 '언령: 정지'와 구분
    ("나이트레아 벨가누스 페이즈", [(MORDUM, MOR_G2)], "문서 간 혼동"),     # '벨가'로 시작하는 벨가르딘 문서와 구분
    ("아르모체 240줄", [(ARMOCHE, ARM_G2)], "문서 간 혼동"),               # 240줄은 세르카·에키드나에도 있음
    # LLM이 실제로 만든 잘못된 검색어 (B-7 실제 답변 확인에서 관찰): 사용자는 벨가르딘을 물었음
    ("벨가누스 거울 무력화 패턴", [(BELGARDIN, BEL_G2)], "변형 검색어"),
    ("카멘 2관문 280줄 카운터 타이밍", [(BELGARDIN, BEL_G2)], "변형 검색어"),
]


# 공략 문서가 없는 레이드·보스 검색어: 검색하지 않고 '문서 없음'으로 거절해야 정답
UNKNOWN_CASES = [
    ("발탄 2관문 공략", "문서 없는 실제 레이드"),
    ("비아키스 1관문 패턴", "문서 없는 실제 레이드"),
    ("쿠크세이튼 3관문 기믹", "문서 없는 실제 레이드"),
    ("아브렐슈드 군단장 레이드 4관문", "문서 없는 실제 레이드"),   # 이름은 문서 본문에 한 줄 있지만 이 레이드 문서는 없음
    ("일리아칸 3관문 패턴", "문서 없는 실제 레이드"),
    ("에기르 1관문 기믹", "문서 없는 실제 레이드"),
    ("드래곤킹 바하무트 2관문 패턴", "존재하지 않는 보스"),
    ("크로노스 레이드 공략", "존재하지 않는 보스"),
    ("하늘성 수호자 오르딘 패턴", "존재하지 않는 보스"),
    ("붉은 용왕 이그니스 공략", "존재하지 않는 보스"),
]


def evaluate_name_check():
    """문서에 없는 이름을 거절하는지, 정상 검색어를 잘못 거절하지 않는지 확인한다 (임베딩 호출 없음)"""
    print(f"\n=== 레이드·보스 이름 확인 평가 (문서에서 자동 추출한 이름 {len(KNOWN_NAMES)}개) ===")
    rejected = [(q, t) for q, t in UNKNOWN_CASES if find_known_name(q) is None]
    print(f"문서 없는 검색어 거절: {len(rejected)}/{len(UNKNOWN_CASES)}")
    for q, t in UNKNOWN_CASES:
        matched = find_known_name(q)
        if matched is not None:
            print(f"  - 거절 못 함 ({t}) {q}: '{matched}'와 일치로 판단")

    wrongly = [(q, t) for q, _, t in TEST_CASES if find_known_name(q) is None]
    print(f"정상 검색어 통과: {len(TEST_CASES) - len(wrongly)}/{len(TEST_CASES)}")
    for q, t in wrongly:
        print(f"  - 잘못 거절 ({t}) {q}")


def chunk_key(doc):
    return (doc.metadata.get("source_file"), doc.metadata.get("Header 2"))


def main():
    if vector_store is None:
        print("벡터 스토어를 열 수 없습니다. Ollama 실행 여부와 .env 설정을 확인하세요.")
        return

    total_chunks = len(vector_store.get()["ids"])
    print(f"적재된 청크 수: {total_chunks}개\n")

    hits = {1: 0, 2: 0, 3: 0}
    dynamic_hit = 0
    reciprocal_ranks = []
    failed = []
    by_type = {}  # 유형별 [동적 k 적중 수, 전체 수]

    for query, answers, case_type in TEST_CASES:
        # 전체 청크를 유사도 순으로 정렬해 정답 순위를 구한다
        ranked = [chunk_key(doc) for doc in vector_store.similarity_search(query=query, k=total_chunks)]
        rank = min(ranked.index(a) + 1 for a in answers if a in ranked)
        k = decide_k(query)

        for n in hits:
            hits[n] += rank <= n
        dynamic_hit += rank <= k
        reciprocal_ranks.append(1 / rank)
        type_stat = by_type.setdefault(case_type, [0, 0])
        type_stat[0] += rank <= k
        type_stat[1] += 1
        if rank > k:
            failed.append((query, case_type, rank, k, ranked[0]))

        mark = "O" if rank <= k else "X"
        top1 = f"{ranked[0][0]} / {ranked[0][1]}"
        print(f"[{mark}] ({case_type}) {query} -> 정답 순위 {rank}위, 동적 k={k}, 1위: {top1}")

    n = len(TEST_CASES)
    print("\n=== RAG 검색 평가 결과 ===")
    print(f"검색어 수: {n}개, 청크 수: {total_chunks}개")
    for k in hits:
        print(f"Hit@{k}: {hits[k]}/{n} ({hits[k] / n:.0%})")
    print(f"동적 k 적중: {dynamic_hit}/{n} ({dynamic_hit / n:.0%})")
    print(f"MRR: {sum(reciprocal_ranks) / n:.3f}")
    print("유형별 동적 k 적중: " + ", ".join(f"{t} {hit}/{total}" for t, (hit, total) in by_type.items()))

    if failed:
        print("\n=== 동적 k 안에 정답이 없었던 검색어 ===")
        for query, case_type, rank, k, top1 in failed:
            print(f"- ({case_type}) {query}: 정답 {rank}위, k={k}, 1위는 {top1[0]} / {top1[1]}")


if __name__ == "__main__":
    main()
    evaluate_name_check()
