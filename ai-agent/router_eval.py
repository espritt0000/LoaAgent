"""
라우터 분기 정확도 측정 스크립트 (문서 7장 '동작 검증'용)

실행 위치: ai-agent 폴더 (agent/, tools/ 와 같은 위치)
실행 방법:
- 단일 턴: python router_eval.py              (질문 40개 = Gemini API 호출 40회)
- 멀티턴: python router_eval.py multiturn    (28건 x 2회(맥락 있음/없음) = 호출 56회)
- 어려운 멀티턴: python router_eval.py hard  (24건 x 2회 = 호출 48회)
- 맥락 있는 쪽만 측정: 뒤에 --ctx-only (예: python router_eval.py hard --ctx-only → 24회). 비교용 '후속 질문만' 호출을 건너뜀
- 섞인 질문: python router_eval.py compound  (12건 = 호출 12회)
- 처음 보는 질문: python router_eval.py holdout  (10건 = 호출 10회, 맥락 있는 쪽만)
- 라우터 LLM만 호출합니다. Stove API·Chroma는 호출하지 않습니다.
- 모델 바꿔서 측정: .env의 LLM_MODEL을 바꾸거나 실행할 때 지정 (Git Bash 예: LLM_MODEL=gemini-3.1-flash-lite python router_eval.py)
"""
import sys
import time
from collections import defaultdict
from langchain_core.messages import HumanMessage, AIMessage
from agent.agent import router_llm, router_system, plan_from

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


# 멀티턴: (이전 질문, 이전 AI 답변, 후속 질문, 기대 경로, 유형)
# - 이전 AI 답변은 실제 에이전트 답변이 아니라 테스트용으로 짧게 작성한 예시
# - 유형 "유지": 후속 질문만으로는 주제를 알 수 없어 앞 대화를 봐야 하는 경우
# - 유형 "전환": 앞 대화와 다른 주제로 넘어가, 앞 대화에 끌려가면 안 되는 경우
MULTITURN_CASES = [
    # 유지 - guide_agent
    ("세르카 1관문 기믹 알려줘", "세르카 1관문은 시작 직후 장판 패턴과 카운터 패턴이 핵심이에요.", "그럼 2관문은?", "guide_agent", "유지"),
    ("벨가르딘 공략 전체 요약해줘", "벨가르딘은 3개 관문으로 구성되고, 관문마다 무력화 패턴이 있어요.", "거기서 제일 어려운 패턴은 뭐야?", "guide_agent", "유지"),
    ("세르카 2관문 240줄 패턴 뭐야?", "240줄에서는 파티원이 정해진 위치로 흩어져 장판을 처리해야 해요.", "그거 실패하면 어떻게 돼?", "guide_agent", "유지"),
    ("세르카 폭탄 빙고 공략", "폭탄 빙고는 바닥 칸에 폭탄을 놓아 한 줄을 완성하는 기믹이에요.", "좀 더 자세히 설명해줘", "guide_agent", "유지"),
    ("벨가르딘 관문 구성 알려줘", "1관문, 2관문, 3관문으로 나뉘고 3관문에서 최종 보스와 싸워요.", "3관문은?", "guide_agent", "유지"),
    # 유지 - market_agent
    ("운명의 파괴석 결정 가격", "운명의 파괴석 결정은 전일 평균가 85 골드, 최근 거래가 83 골드예요.", "어제보다 올랐어?", "market_agent", "유지"),
    ("운명의 파괴석 결정 가격", "운명의 파괴석 결정은 전일 평균가 85 골드, 최근 거래가 83 골드예요.", "돌파석은?", "market_agent", "유지"),
    ("보석 경매장 가격 확인해줘", "현재 경매장에 등록된 보석 가격을 정리해 드릴게요.", "그럼 제일 싼 건 얼마야?", "market_agent", "유지"),
    ("운명의 수호석 결정 시세", "운명의 수호석 결정은 최근 거래가 12 골드예요.", "100개 사면 얼마야?", "market_agent", "유지"),
    ("재련 재료 요즘 시세 어때?", "주요 재련 재료는 전일 대비 소폭 하락했어요.", "지금 사는 게 나아?", "market_agent", "유지"),
    # 유지 - character_agent
    ("모코코짱123 전투력 알려줘", "모코코짱123의 전투력은 1,850,000이에요.", "길드는 어디야?", "character_agent", "유지"),
    ("테스트닉네임 아이템 레벨 알려줘", "테스트닉네임의 아이템 레벨은 1,720이에요.", "직업은?", "character_agent", "유지"),
    ("샘플유저 프로필 보여줘", "샘플유저는 전투 레벨 70, 아이템 레벨 1,700이에요.", "서버는 어디야?", "character_agent", "유지"),
    ("가상캐릭터A 스펙 봐줘", "가상캐릭터A는 아이템 레벨 1,680, 전투력 1,500,000이에요.", "전투 레벨도 알려줘", "character_agent", "유지"),
    ("가상캐릭터B 직업이 뭐야", "가상캐릭터B의 직업은 브레이커예요.", "그 캐릭터 템렙은?", "character_agent", "유지"),
    # 유지 - general_agent
    ("안녕!", "안녕하세요! 로스트아크 AI 비서예요.", "요즘 뭐 하고 지내?", "general_agent", "유지"),
    ("점심 메뉴 추천해줘", "오늘은 따뜻한 김치찌개 어떠세요?", "다른 건?", "general_agent", "유지"),
    ("심심한데 얘기 좀 하자", "좋아요! 어떤 이야기를 해 볼까요?", "너는 뭘 좋아해?", "general_agent", "유지"),
    ("넌 누구야?", "저는 로스트아크 AI 비서예요.", "그럼 뭘 할 수 있어?", "general_agent", "유지"),
    ("하루 동안 초능력을 가질 수 있다면 뭘 고를래?", "저는 순간이동을 고를 것 같아요.", "왜 그걸 골랐어?", "general_agent", "유지"),
    # 전환
    ("세르카 1관문 기믹 알려줘", "세르카 1관문은 시작 직후 장판 패턴과 카운터 패턴이 핵심이에요.", "고마워!", "general_agent", "전환"),
    ("운명의 파괴석 결정 가격", "운명의 파괴석 결정은 전일 평균가 85 골드, 최근 거래가 83 골드예요.", "모코코짱123 전투력은?", "character_agent", "전환"),
    ("모코코짱123 전투력 알려줘", "모코코짱123의 전투력은 1,850,000이에요.", "세르카 1관문 기믹은?", "guide_agent", "전환"),
    ("점심 메뉴 추천해줘", "오늘은 따뜻한 김치찌개 어떠세요?", "운명의 파괴석 결정 가격 알려줘", "market_agent", "전환"),
    ("벨가르딘 공략 전체 요약해줘", "벨가르딘은 3개 관문으로 구성되고, 관문마다 무력화 패턴이 있어요.", "운명의 돌파석 시세 알려줘", "market_agent", "전환"),
    ("보석 경매장 가격 확인해줘", "현재 경매장에 등록된 보석 가격을 정리해 드릴게요.", "오늘 숙제 하기 싫다", "general_agent", "전환"),
    ("테스트닉네임 아이템 레벨 알려줘", "테스트닉네임의 아이템 레벨은 1,720이에요.", "운명의 파괴석 결정 가격", "market_agent", "전환"),
    ("세르카 2관문 240줄 패턴 뭐야?", "240줄에서는 파티원이 정해진 위치로 흩어져 장판을 처리해야 해요.", "샘플유저 길드 어디야?", "character_agent", "전환"),
]


def classify(messages):
    """라우터 LLM의 작업 계획을 에이전트 이름으로 돌려준다 (2단계면 "a+b"). 실패하면 ERROR(예외 이름)
    단일 영역 질문은 계획이 1단계여야 정답이므로, 불필요하게 2단계로 나누면 틀린 것으로 센다"""
    try:
        plan = plan_from(router_llm.invoke([router_system] + messages))
        return "+".join(step["agent"] for step in plan) or "ERROR(빈 계획)"
    except Exception as e:
        return f"ERROR({type(e).__name__})"


# 어려운 멀티턴: ([(이전 질문, 이전 답변), ...], 후속 질문, 기대 경로, 유형)
# - 유형 "같은 질문": 단서가 없는 같은 후속 질문을 서로 다른 앞 대화와 짝지어, 앞 대화에 따라 정답 경로가 달라지는지 확인
# - 유형 "두 턴 전": 바로 앞이 아니라 두 턴 전 주제로 돌아가는 질문
# - 두 영역이 섞인 질문은 현재 구조(한 턴에 에이전트 하나)로 답할 수 없어 여기서 측정하지 않음
GUIDE_TURN = ("세르카 1관문 기믹 알려줘", "세르카 1관문은 시작 직후 장판 패턴과 카운터 패턴이 핵심이에요.")
MARKET_TURN = ("운명의 파괴석 결정 가격", "운명의 파괴석 결정은 전일 평균가 85 골드, 최근 거래가 83 골드예요.")
CHARACTER_TURN = ("모코코짱123 전투력 알려줘", "모코코짱123의 전투력은 1,850,000이에요.")
GENERAL_TURN = ("점심 메뉴 추천해줘", "오늘은 따뜻한 김치찌개 어떠세요?")

HARD_CASES = [
    # 같은 질문 - "그거 자세히 알려줘"
    ([GUIDE_TURN], "그거 자세히 알려줘", "guide_agent", "같은 질문"),
    ([MARKET_TURN], "그거 자세히 알려줘", "market_agent", "같은 질문"),
    ([CHARACTER_TURN], "그거 자세히 알려줘", "character_agent", "같은 질문"),
    # 같은 질문 - "그거 괜찮은 거야?"
    ([MARKET_TURN], "그거 괜찮은 거야?", "market_agent", "같은 질문"),
    # 전투력이 괜찮은지 판단할 비교 기준 데이터가 어느 에이전트에도 없어, 정답 경로를 정할 수 없음 → 점수 제외
    ([CHARACTER_TURN], "그거 괜찮은 거야?", None, "정답 없음"),
    ([GENERAL_TURN], "그거 괜찮은 거야?", "general_agent", "같은 질문"),
    # 같은 질문 - "다른 건?"
    ([GUIDE_TURN], "다른 건?", "guide_agent", "같은 질문"),
    ([MARKET_TURN], "다른 건?", "market_agent", "같은 질문"),
    ([GENERAL_TURN], "다른 건?", "general_agent", "같은 질문"),
    # 같은 질문 - "그거 다시 말해줘"
    ([GUIDE_TURN], "그거 다시 말해줘", "guide_agent", "같은 질문"),
    ([MARKET_TURN], "그거 다시 말해줘", "market_agent", "같은 질문"),
    ([CHARACTER_TURN], "그거 다시 말해줘", "character_agent", "같은 질문"),
    # 두 턴 전 주제로 돌아가기
    ([CHARACTER_TURN, GUIDE_TURN], "아까 그 캐릭터 길드는?", "character_agent", "두 턴 전"),
    ([MARKET_TURN, CHARACTER_TURN], "아까 그 재료 어제보다 올랐어?", "market_agent", "두 턴 전"),
    ([CHARACTER_TURN, MARKET_TURN], "처음에 물어본 그 사람 서버는?", "character_agent", "두 턴 전"),
    # "아까 그거"가 공략인지 시세인지 사람이 봐도 모호함. 둘 중 어느 쪽이든(둘 다 포함) 전문 에이전트로 가면 정답,
    # 일반 대화(general)로 빠지면 오답으로 본다
    ([GUIDE_TURN, MARKET_TURN], "아까 그거 다시 설명해줘", ("guide_agent", "market_agent"), "두 턴 전"),
    ([MARKET_TURN, GENERAL_TURN], "아까 그 가격 다시 알려줘", "market_agent", "두 턴 전"),
    # 단서 없이 두 턴 전으로 돌아가는 질문 (앞 전문 주제가 하나뿐이라 정답이 분명함)
    ([GUIDE_TURN, GENERAL_TURN], "아까 그거 다시 설명해줘", "guide_agent", "두 턴 전(단서 없음)"),
    ([CHARACTER_TURN, GENERAL_TURN], "아까 그거 다시 알려줘", "character_agent", "두 턴 전(단서 없음)"),
    ([MARKET_TURN, GENERAL_TURN], "아까 말한 거 다시 알려줘", "market_agent", "두 턴 전(단서 없음)"),
    # 앞 대화 없이 지시어만 있는 질문: 가리킬 대상이 없으므로 general_agent
    # (지시어를 앞 대화로 해석하라는 규칙이 앞 대화가 없을 때까지 전문 에이전트로 보내지 않는지 확인)
    ([], "그거 자세히 알려줘", "general_agent", "앞 대화 없음"),
    ([], "그거 다시 말해줘", "general_agent", "앞 대화 없음"),
    ([], "아까 그거 다시 설명해줘", "general_agent", "앞 대화 없음"),
    ([], "아까 말한 거 다시 알려줘", "general_agent", "앞 대화 없음"),
]


# 처음 보는 질문(과적합 확인): v3 [이전 대화 활용] 규칙을 다듬을 때 보지 않은 질문
# - HARD_CASES로 규칙을 세 번 고쳤으므로, 그 질문에만 맞춰졌는지 확인하려고 따로 둔다
# - 앞 대화 주제(벨가르딘·돌파석·샘플유저)와 말투를 HARD_CASES와 다르게 하고, 규칙마다 최소 한 문항
# - 이 질문을 보고 규칙을 고치면 더 이상 '처음 보는 질문'이 아니므로, 규칙을 고친 뒤에는 새 질문을 만들 것
HOLDOUT_GUIDE = ("벨가르딘 2관문 공략 알려줘", "벨가르딘 2관문은 무력화 패턴 뒤 전멸기를 피하는 것이 핵심이에요.")
HOLDOUT_MARKET = ("운명의 돌파석 시세 어때?", "운명의 돌파석은 최근 거래가 120 골드예요.")
HOLDOUT_CHARACTER = ("샘플유저 아이템 레벨 몇이야?", "샘플유저의 아이템 레벨은 1,740이에요.")
HOLDOUT_SMALLTALK = ("요즘 너무 피곤하다", "푹 쉬는 것도 중요해요. 오늘은 일찍 주무세요!")

HOLDOUT_CASES = [
    ([HOLDOUT_MARKET], "그거 지난주랑 비교하면 어때?", "market_agent", "생략·지시어"),
    ([HOLDOUT_GUIDE], "거기서 제일 어려운 패턴은 뭐야?", "guide_agent", "생략·지시어"),
    ([HOLDOUT_CHARACTER], "그 사람 직업은 뭐야?", "character_agent", "생략·지시어"),
    ([HOLDOUT_GUIDE, HOLDOUT_SMALLTALK], "아까 하던 얘기 이어서 해줘", "guide_agent", "이전 시점(잡담 사이)"),
    ([HOLDOUT_CHARACTER, HOLDOUT_SMALLTALK], "아까 걔 전투력은?", "character_agent", "이전 시점(잡담 사이)"),
    ([HOLDOUT_MARKET, HOLDOUT_GUIDE], "처음에 물어본 거 지금도 그 가격이야?", "market_agent", "이전 시점(주제 사이)"),
    ([HOLDOUT_MARKET], "그건 됐고 세르카 1관문 기믹 알려줘", "guide_agent", "새 주제 우선"),
    # 부작용 확인: 전문 에이전트로 보내면 안 되는 질문
    ([HOLDOUT_CHARACTER], "고마워, 덕분에 알았어", "general_agent", "부작용(인사)"),
    ([HOLDOUT_SMALLTALK], "그거 말고 다른 방법은?", "general_agent", "부작용(잡담 지시어)"),
    ([], "그 사람 길드 어디야?", "general_agent", "부작용(앞 대화 없음)"),
]

# 두 영역이 섞인 질문(에이전트 간 협업 14번): (질문, 필요한 에이전트 목록, 유형)
# - 유형 "의존": 앞 에이전트의 결과가 있어야 답할 수 있음 (공략 문서의 입장 레벨 → 캐릭터 아이템 레벨과 비교)
# - 유형 "독립": 두 질문을 한 문장에 묶어, 각각 따로 답하면 됨
# - 유형 "함정": 대상이 둘이라 섞인 것처럼 보이지만 에이전트 하나로 답할 수 있음
# - 공략 문서에는 재료·골드 정보가 없고 입장 레벨만 있어, "레이드 가려면 재료값 얼마?" 같은 질문은 넣지 않음
COMPOUND_CASES = [
    # 의존 - 공략(입장 레벨) + 캐릭터(아이템 레벨)
    ("모코코짱123으로 칠흑 폭풍의 밤 하드 갈 수 있어?", ["guide_agent", "character_agent"], "의존"),
    ("테스트닉네임 템렙이면 종막 하드 입장 가능해?", ["guide_agent", "character_agent"], "의존"),
    ("샘플유저가 지평의 성당 3단계 갈 수 있는 레벨이야?", ["guide_agent", "character_agent"], "의존"),
    ("가상캐릭터A로 파멸의 성채 노말 입장되나?", ["guide_agent", "character_agent"], "의존"),
    # 독립 - 두 질문을 한 문장에
    ("종막 2관문 패턴이랑 운명의 파괴석 결정 시세 알려줘", ["guide_agent", "market_agent"], "독립"),
    ("세르카 1관문 기믹 알려주고 운명의 돌파석 가격도 알려줘", ["guide_agent", "market_agent"], "독립"),
    ("모코코짱123 전투력이랑 운명의 수호석 결정 시세 알려줘", ["character_agent", "market_agent"], "독립"),
    ("테스트닉네임 길드 어디야? 그리고 보석 경매장 가격도 알려줘", ["character_agent", "market_agent"], "독립"),
    ("벨가르딘 관문 구성이랑 샘플유저 아이템 레벨 알려줘", ["guide_agent", "character_agent"], "독립"),
    # 함정 - 에이전트 하나로 충분
    ("운명의 파괴석 결정이랑 운명의 돌파석 시세 비교해줘", ["market_agent"], "함정"),
    ("세르카랑 벨가르딘 중에 관문이 더 많은 건 어디야?", ["guide_agent"], "함정"),
    ("모코코짱123 템렙이랑 전투력 알려줘", ["character_agent"], "함정"),
]


def run_compound_eval():
    """라우터의 작업 계획이 필요한 에이전트와 같은지 센다 (순서는 보지 않음: 마지막 단계가 앞 결과를 합쳐 답하므로)
    Planner 이전 구조(에이전트 하나만 선택)의 기준선: 섞인 질문 9건 모두 둘 중 하나만 고름 (2026-11-19, 문서 7.5)"""
    stats = defaultdict(lambda: {"total": 0, "hit": 0})
    for question, expected, case_type in COMPOUND_CASES:
        actual = classify([HumanMessage(content=question)])
        time.sleep(0.5)  # 요청 속도 제한 대비
        hit = set(actual.split("+")) == set(expected)
        stats[case_type]["total"] += 1
        stats[case_type]["hit"] += hit
        print(f"[{'O' if hit else 'X'}] ({case_type}) {question} -> {actual} (필요: {' + '.join(expected)})")

    print("\n=== 섞인 질문 결과 ===")
    print(f"측정일: {time.strftime('%Y-%m-%d')}")
    print("유형: 계획이 필요한 에이전트와 같은 수 / 전체 (함정은 1단계여야 정답)")
    for case_type in ["의존", "독립", "함정"]:
        s = stats[case_type]
        print(f"{case_type}: {s['hit']} / {s['total']}")


def to_messages(history, follow_up):
    """[(질문, 답변), ...] + 후속 질문을 라우터에 넣을 메시지 목록으로 바꾼다"""
    messages = []
    for question, answer in history:
        messages += [HumanMessage(content=question), AIMessage(content=answer)]
    return messages + [HumanMessage(content=follow_up)]


def run_context_eval(cases, title, ctx_only=False):
    """후속 질문을 '앞 대화와 함께' 넣은 결과와 '후속 질문만' 넣은 결과를 비교한다
    ctx_only=True면 '후속 질문만' 호출을 건너뛰어 API 호출을 절반으로 줄인다 (결과에는 '-'로 표시)"""
    stats = defaultdict(lambda: {"total": 0, "with_ctx": 0, "no_ctx": 0})
    wrong = []

    for history, follow_up, expected, case_type in cases:
        prev_desc = " → ".join(question for question, _ in history)
        # 실제 서비스처럼 이전 질문·답변 뒤에 후속 질문을 붙여 라우터에 전달
        with_ctx = classify(to_messages(history, follow_up))
        time.sleep(0.5)  # 요청 속도 제한 대비
        # 비교용: 앞 대화 없이 후속 질문만 전달
        if ctx_only:
            no_ctx = "-"
        else:
            no_ctx = classify([HumanMessage(content=follow_up)])
            time.sleep(0.5)

        # 정답 없음(None): 어느 에이전트도 근거 있게 답할 수 없는 질문이라 점수에서 빼고 결과만 기록
        if expected is None:
            print(f"[-] ({case_type}) {prev_desc} → {follow_up} "
                  f"-> 맥락 있음: {with_ctx} / 맥락 없음: {no_ctx} (정답 없음, 점수 제외)")
            continue

        # 기대 경로가 튜플이면 그중 어느 쪽이든 정답. 계획의 모든 에이전트가 허용 목록 안에 있어야 정답
        # (단일 기대 경로면 1단계로 그 에이전트만 골라야 정답, 모호한 지시어는 공략·시세 하나 또는 둘 다 허용)
        accepted = expected if isinstance(expected, tuple) else (expected,)
        def ok(result):
            return set(result.split("+")) <= set(accepted)
        key = f"{case_type}/{'|'.join(accepted)}"
        stats[key]["total"] += 1
        stats[key]["with_ctx"] += ok(with_ctx)
        stats[key]["no_ctx"] += ok(no_ctx)
        if not ok(with_ctx):
            wrong.append((prev_desc, follow_up, "|".join(accepted), with_ctx))
        print(f"[{'O' if ok(with_ctx) else 'X'}] ({case_type}) {prev_desc} → {follow_up} "
              f"-> 맥락 있음: {with_ctx} / 맥락 없음: {no_ctx}")

    print(f"\n=== {title} 결과 ===")
    print(f"측정일: {time.strftime('%Y-%m-%d')}")
    print("유형/기대 경로: 맥락 있음 정답 수 / 맥락 없음 정답 수 / 전체")
    for key in sorted(stats):
        s = stats[key]
        print(f"{key}: {s['with_ctx']} / {s['no_ctx']} / {s['total']}")
    total = sum(s["total"] for s in stats.values())
    print(f"전체: {sum(s['with_ctx'] for s in stats.values())} / "
          f"{sum(s['no_ctx'] for s in stats.values())} / {total}")

    if wrong:
        print("\n=== 맥락이 있는데도 잘못 분기된 질문 ===")
        for prev_desc, follow_up, exp, act in wrong:
            print(f"- {prev_desc} → {follow_up} (기대: {exp}, 실제: {act})")


def main():
    total = defaultdict(int)
    correct = defaultdict(int)
    wrong = []

    for question, expected in TEST_CASES:
        actual = classify([HumanMessage(content=question)])  # 실패하면 ERROR(...)
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
    mode = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ""
    ctx_only = "--ctx-only" in sys.argv
    if mode == "multiturn":
        # 기존 멀티턴 케이스는 (이전 질문, 이전 답변)이 한 턴이므로 같은 형식으로 바꿔서 실행
        cases = [([(q, a)], follow_up, exp, t) for q, a, follow_up, exp, t in MULTITURN_CASES]
        run_context_eval(cases, "멀티턴", ctx_only)
    elif mode == "hard":
        run_context_eval(HARD_CASES, "어려운 멀티턴", ctx_only)
    elif mode == "holdout":
        run_context_eval(HOLDOUT_CASES, "처음 보는 질문", ctx_only=True)
    elif mode == "compound":
        run_compound_eval()
    else:
        main()