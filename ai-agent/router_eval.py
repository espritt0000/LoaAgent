"""
라우터 분기 정확도 측정 스크립트 (문서 7장 '동작 검증'용)

실행 위치: ai-agent 폴더 (agent/, tools/ 와 같은 위치)
실행 방법: python router_eval.py
- 라우터 LLM만 호출합니다. Stove API·Chroma는 호출하지 않습니다.
- 질문 40개 = Gemini API 호출 40회입니다.
"""
import time
from collections import defaultdict
from langchain_core.messages import HumanMessage
from agent.agent import router_llm, router_system

# (질문, 기대 경로) — 캐릭터명은 가상의 닉네임
TEST_CASES = [
    # character_agent
    ("테스트닉네임 아이템 레벨 알려줘", "character_agent"),
    ("모코코짱123 전투력 몇이야?", "character_agent"),
    ("가상캐릭터A 직업이 뭐야", "character_agent"),
    ("샘플유저 길드 어디야?", "character_agent"),
    ("포트폴리오용캐릭 서버 알려줘", "character_agent"),
    ("테스트닉네임 스펙 좀 봐줘", "character_agent"),
    ("가상캐릭터B 전투 레벨 몇이야", "character_agent"),
    ("모코코짱123 캐릭터 정보 조회해줘", "character_agent"),
    ("샘플유저 템렙이랑 전투력 같이 알려줘", "character_agent"),
    ("가상캐릭터C 프로필 보여줘", "character_agent"),
    # market_agent
    ("운명의 파괴석 결정 가격", "market_agent"),
    ("운명의 돌파석 시세 알려줘", "market_agent"),
    ("경매장에서 각인서 얼마야?", "market_agent"),
    ("재련 재료 요즘 시세 어때?", "market_agent"),
    ("운명의 수호석 결정 어제보다 올랐어?", "market_agent"),
    ("아비도스 융화 재료 가격 알려줘", "market_agent"),
    ("거래소에서 제일 싼 파괴석 얼마야", "market_agent"),
    ("명예의 파편 주머니 시세", "market_agent"),
    ("재료 사는 데 골드 얼마나 들어?", "market_agent"),
    ("보석 경매장 가격 확인해줘", "market_agent"),
    # guide_agent
    ("세르카 1관문 기믹 알려줘", "guide_agent"),
    ("세르카 2관문 240줄 패턴 뭐야?", "guide_agent"),
    ("벨가르딘 공략 전체 요약해줘", "guide_agent"),
    ("세르카 대난투 시스템이 뭐야", "guide_agent"),
    ("코르부스 툴 라크 진짜 찾기 패턴 어떻게 해?", "guide_agent"),
    ("세르카 폭탄 빙고 공략", "guide_agent"),
    ("벨가르딘 관문 구성 알려줘", "guide_agent"),
    ("어비스 던전 세르카 난이도 종류", "guide_agent"),
    ("저스트 가드 패턴 타이밍 공략 알려줘", "guide_agent"),
    ("레이드 공략 좀 알려줘", "guide_agent"),
    # general_agent
    ("안녕!", "general_agent"),
    ("오늘 기분 어때?", "general_agent"),
    ("넌 누구야?", "general_agent"),
    ("하루 동안 초능력을 가질 수 있다면 뭘 고를래?", "general_agent"),
    ("고마워 잘 쓸게", "general_agent"),
    ("심심한데 얘기 좀 하자", "general_agent"),
    ("점심 메뉴 추천해줘", "general_agent"),
    ("로스트아크 재밌어?", "general_agent"),
    ("너는 어떤 걸 도와줄 수 있어?", "general_agent"),
    ("오늘 숙제 하기 싫다", "general_agent"),
]


def main():
    total = defaultdict(int)
    correct = defaultdict(int)
    wrong = []

    for question, expected in TEST_CASES:
        try:
            result = router_llm.invoke([router_system, HumanMessage(content=question)])
            actual = result.destination
        except Exception as e:  # 구조화 출력 파싱 실패 등
            actual = f"ERROR({type(e).__name__})"
        total[expected] += 1
        if actual == expected:
            correct[expected] += 1
        else:
            wrong.append((question, expected, actual))
        print(f"[{'O' if actual == expected else 'X'}] {question} -> {actual}")
        time.sleep(0.5)  # 요청 속도 제한 대비

    print("\n=== 결과 (문서 7장 표에 그대로 옮기면 됩니다) ===")
    print(f"측정일: {time.strftime('%Y-%m-%d')}")
    for route in ["character_agent", "market_agent", "guide_agent", "general_agent"]:
        print(f"{route}: {correct[route]}/{total[route]}")
    print(f"전체: {sum(correct.values())}/{len(TEST_CASES)}")

    if wrong:
        print("\n=== 잘못 분기된 질문 ===")
        for q, exp, act in wrong:
            print(f"- {q} (기대: {exp}, 실제: {act})")


if __name__ == "__main__":
    main()