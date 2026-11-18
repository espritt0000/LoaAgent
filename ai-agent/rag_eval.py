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
from tools.guide import vector_store, decide_k

SERKA = "serka.md"
BELGARDIN = "belgardin.md"
OVERVIEW = "개요 및 입장 정보"
SERKA_G1 = "1관문: 고통의 마녀 세르카 주요 패턴 및 기믹"
SERKA_G2 = "2관문: 태초의 그림자 코르부스 툴 라크 주요 패턴 및 기믹"
BEL_G1 = "1관문: 성자 벨가르딘 크로체 주요 패턴 및 기믹"
BEL_G2 = "2관문: 페투스 안 크라그마 주요 패턴 및 기믹"

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
]


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

    for query, answers, case_type in TEST_CASES:
        # 전체 청크를 유사도 순으로 정렬해 정답 순위를 구한다
        ranked = [chunk_key(doc) for doc in vector_store.similarity_search(query=query, k=total_chunks)]
        rank = min(ranked.index(a) + 1 for a in answers if a in ranked)
        k = decide_k(query)

        for n in hits:
            hits[n] += rank <= n
        dynamic_hit += rank <= k
        reciprocal_ranks.append(1 / rank)
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

    if failed:
        print("\n=== 동적 k 안에 정답이 없었던 검색어 ===")
        for query, case_type, rank, k, top1 in failed:
            print(f"- ({case_type}) {query}: 정답 {rank}위, k={k}, 1위는 {top1[0]} / {top1[1]}")


if __name__ == "__main__":
    main()
