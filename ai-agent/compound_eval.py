"""
두 영역이 섞인 질문의 실제 답변 확인 스크립트 (에이전트 간 협업 14번의 변경 전·후 비교용)

실행 위치: ai-agent 폴더
실행 방법: python compound_eval.py   (Git Bash에서 모델 지정: LLM_MODEL=gemini-3.1-flash-lite python compound_eval.py)
- 서비스 그래프 전체(라우터 → 에이전트 → 툴 → 답변)를 실행합니다. Gemini, Stove API, Chroma(Ollama)를 실제로 호출합니다.
- 대화 기록은 서비스 SQLite 파일이 아니라 메모리(MemorySaver)에 저장하고, 질문마다 새 대화로 시작합니다.
- 답변에 각 영역의 정보가 들어갔는지를 정해 둔 기준으로 확인합니다. 답변 전문은 compound_eval_answers.json에 저장합니다(git 제외).
"""
import ast
import json
import re
import time
import uuid
from pathlib import Path
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from agent.agent import workflow

CHARACTER = "니나브았아"  # 실제 캐릭터 (Stove API로 조회됨)

# (질문, 유형, 확인할 영역 목록)
# - guide_level:<레벨>: 공략 검색을 했고, 답변에 문서의 입장 레벨이 들어감
# - guide_words:<단어|단어>: 공략 검색을 했고, 답변에 해당 관문 패턴 단어가 하나 이상 들어감
# - character:<항목>: 캐릭터 조회를 했고, 답변에 조회 결과의 해당 항목 값이 들어감
# - market: 시세 조회(search_market_items)를 했고, 답변에 '골드'가 들어감
CASES = [
    (f"{CHARACTER}로 칠흑 폭풍의 밤 하드 갈 수 있어?", "의존", ["guide_level:1700", "character:아이템레벨"]),
    (f"{CHARACTER} 템렙이면 지평의 성당 3단계 입장 가능해?", "의존", ["guide_level:1750", "character:아이템레벨"]),
    ("종막 2관문 패턴이랑 운명의 파괴석 결정 시세 알려줘", "독립",
     ["guide_words:언령|950줄|900줄|콜로세움|3연속 저스트 가드|블랙홀", "market"]),
    ("세르카 1관문 기믹 알려주고 운명의 돌파석 가격도 알려줘", "독립",
     ["guide_words:270줄|240줄|폭탄 빙고|화염 메이든|모루", "market"]),
    (f"{CHARACTER} 전투력이랑 운명의 수호석 결정 시세 알려줘", "독립", ["character:전투력", "market"]),
    (f"{CHARACTER} 템렙이랑 전투력 알려줘", "함정", ["character:아이템레벨", "character:전투력"]),
]

app = workflow.compile(checkpointer=MemorySaver())


def to_text(content) -> str:
    if isinstance(content, list):
        return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    return content or ""


def digits(text) -> str:
    """쉼표·공백을 빼고 숫자 비교용 문자열로 바꾼다 (1,712.5 → 1712.5)"""
    return re.sub(r"[,\s]", "", str(text))


def check(spec, answer, tool_calls, tool_results):
    """확인 기준 하나를 검사해 (통과 여부, 근거)를 돌려준다"""
    kind, _, arg = spec.partition(":")
    plain = digits(answer)
    if kind == "guide_level":
        if "search_game_guide" not in tool_calls:
            return False, "공략 검색 안 함"
        return arg in plain, f"입장 레벨 {arg} {'있음' if arg in plain else '없음'}"
    if kind == "guide_words":
        if "search_game_guide" not in tool_calls:
            return False, "공략 검색 안 함"
        found = [w for w in arg.split("|") if w in answer]
        return bool(found), f"패턴 단어 {found or '없음'}"
    if kind == "character":
        profile = tool_results.get("get_character_profile")
        if profile is None:
            return False, "캐릭터 조회 안 함"
        value = profile.get(arg)
        if value is None:
            return False, f"조회 결과에 {arg} 없음"
        # 아이템 레벨은 소수점 아래를 생략해 답할 수 있어 정수 부분만 비교
        target = digits(value).split(".")[0]
        return target in plain, f"{arg} {value} {'있음' if target in plain else '없음'}"
    if kind == "market":
        if "search_market_items" not in tool_calls:
            return False, "시세 조회 안 함"
        return "골드" in answer, f"'골드' {'있음' if '골드' in answer else '없음'}"
    raise ValueError(spec)


def run_case(question):
    config = {"configurable": {"thread_id": f"eval:{uuid.uuid4()}"}, "recursion_limit": 15}
    result = app.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    messages = result["messages"]
    tool_calls, tool_results = [], {}
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            tool_calls.append(call["name"])
        if isinstance(msg, ToolMessage) and msg.name == "get_character_profile":
            try:
                tool_results["get_character_profile"] = ast.literal_eval(to_text(msg.content))
            except (ValueError, SyntaxError):
                tool_results["get_character_profile"] = json.loads(to_text(msg.content))
    return result.get("next_node"), tool_calls, tool_results, to_text(messages[-1].content)


def main():
    records, passed_cases = [], 0
    for question, case_type, specs in CASES:
        try:
            route, tool_calls, tool_results, answer = run_case(question)
            checks = [(spec, *check(spec, answer, tool_calls, tool_results)) for spec in specs]
        except Exception as e:
            route, tool_calls, answer = "ERROR", [], f"{type(e).__name__}: {e}"
            checks = [(spec, False, "실행 오류") for spec in specs]
        ok = all(c[1] for c in checks)
        passed_cases += ok
        records.append({"question": question, "type": case_type, "route": route, "tools": tool_calls,
                        "checks": [{"spec": s, "pass": p, "detail": d} for s, p, d in checks], "answer": answer})
        time.sleep(1)  # 요청 속도 제한 대비

    out = Path(__file__).resolve().parent / "compound_eval_answers.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 섞인 질문 실제 답변 확인 결과 ===")
    print(f"측정일: {time.strftime('%Y-%m-%d')}")
    for r in records:
        mark = "O" if all(c["pass"] for c in r["checks"]) else "X"
        print(f"[{mark}] ({r['type']}) {r['question']}")
        print(f"     라우터: {r['route']} / 툴: {', '.join(r['tools']) or '없음'}")
        for c in r["checks"]:
            print(f"     - {'통과' if c['pass'] else '실패'} {c['spec'].split(':')[0]}: {c['detail']}")
    print(f"모든 영역 확인 통과: {passed_cases} / {len(CASES)}")
    print(f"답변 전문: {out.name}")


if __name__ == "__main__":
    main()
