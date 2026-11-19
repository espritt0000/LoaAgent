from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage, trim_messages
from langchain_core.exceptions import ModelError
from pydantic import BaseModel, Field
import os
import sqlite3
from pathlib import Path
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv


# 도구 임포트
from tools.character import get_character_profile
from tools.market import get_market_options, search_market_items
from tools.guide import search_game_guide

load_dotenv()

class AgentState(MessagesState):
    """
    LangGraph 전역 상태 객체
    - messages: 대화 히스토리 (사용자 질문, AI 답변, 툴 실행 결과 등 누적)
    - next_node: 라우터가 다음에 실행해야 한다고 판단한 전문 에이전트 노드 이름
    """
    next_node: str = Field(default="", description="다음에 실행할 node 이름")
    # [Planner] 라우터가 세운 이번 턴의 작업 계획과 현재 단계 번호 (두 영역이 섞인 질문을 순서대로 처리)
    plan: list = Field(default_factory=list, description="[{'agent': 에이전트 이름, 'task': 이 단계에서 할 일}, ...]")
    step: int = Field(default=0, description="지금 실행 중인 계획 단계 (0부터)")
# [LLM 교체] 모델을 코드가 아니라 .env 설정값으로 정한다 (코드에 기본 모델을 두지 않음, .env.example 참고)
# - LLM_PROVIDER: LangChain 공급자 이름 (google_genai, ollama, anthropic 등)
#   google_genai 외의 공급자는 해당 패키지(langchain-ollama, langchain-anthropic 등)와 API 키가 필요하다
# - LLM_MODEL: 공급자의 모델 이름
LLM_PROVIDER = os.getenv("LLM_PROVIDER")
LLM_MODEL = os.getenv("LLM_MODEL")
if not LLM_PROVIDER or not LLM_MODEL:
    # 설정이 빠졌는데 다른 모델로 조용히 실행되지 않도록 서버 시작 단계에서 바로 멈춘다
    raise RuntimeError("🚨 [LLM CONFIG] .env에 LLM_PROVIDER와 LLM_MODEL을 설정해야 합니다 (.env.example 참고)")
print(f"🤖 [LLM CONFIG] provider={LLM_PROVIDER}, model={LLM_MODEL}")
llm = init_chat_model(LLM_MODEL, model_provider=LLM_PROVIDER)

# 토큰 트림 설정
trimmer = trim_messages(
    max_tokens=12000,
    strategy="last",
    token_counter=llm,
    include_system=True,
    start_on="human",
)

class PlanStep(BaseModel):
    """작업 계획의 한 단계: 어느 에이전트가 무엇을 할지"""
    agent: Literal["character_agent", "market_agent", "guide_agent", "general_agent"] = Field(
        description=
            "이 단계를 맡을 에이전트: "
            "1. character_agent: 특정 캐릭터의 기본 프로필(서버, 직업, 전투 레벨, 아이템 레벨, 전투력, 길드) 조회 관련 "
            "2. market_agent: 경매장, 시세, 가격, 아이템 구매/판매, 재료 시세 조회 관련 "
            "3. guide_agent: 레이드 공략, 군단장/어비스 던전 패턴, 보스 공략법, 입장 레벨, 게임 가이드, 게임 정보 및 육성법 지식 검색 관련 "
            "4. general_agent: 단순 인사, 일반적인 잡담 및 위 카테고리에 속하지 않는 질문"
    )
    task: str = Field(description="이 에이전트가 맡을 일. 캐릭터명·아이템명·레이드명은 사용자가 쓴 그대로 넣을 것")


class RouterOutput(BaseModel):
    """라우터가 세우는 작업 계획 (대부분 1단계, 두 영역이 모두 필요할 때만 2단계)"""
    steps: list[PlanStep] = Field(description="실행할 단계 목록 (1~2개, 순서대로 실행되고 2단계면 마지막에 결과를 합쳐 최종 답변을 만듦)")

router_llm = llm.with_structured_output(RouterOutput)
# 라우터 분류 시도 횟수 (첫 시도 + 재시도 1회)
ROUTER_MAX_ATTEMPTS = 2
# 분류에 끝내 실패했을 때 보낼 기본 경로
ROUTER_FALLBACK = "general_agent"
# 한 턴에 실행할 최대 단계 수 (LLM 호출 수와 recursion_limit을 고려해 2로 제한)
MAX_PLAN_STEPS = 2
router_system = SystemMessage(content=
    """너는 로스트아크 AI 비서의 라우터야. 사용자의 질문을 분석해 적절한 에이전트를 지정해.
    [분류 기준 및 예시]
    1. character_agent (캐릭터 스펙/정보 조회):
        - 특정 닉네임이 언급되면서 서버, 직업, 전투 레벨, 아이템 레벨, 전투력, 길드 등 기본 프로필을 물어볼 때
        - 단어에 "캐릭터"라는 말이 없어도, 사람 이름/닉네임과 게임 스펙 키워드가 함께 나오면 무조건 이쪽으로 보낼 것.
    2. market_agent (경매장/시세):
        - 특정 아이템의 가격, 시세, 경매장 검색, 재료 비용 문의
    3. guide_agent (공략/게임 지식):
        - 레이드 공략, 관문·패턴·기믹, 게임 가이드
    4. general_agent (일반):
        - 인사, 잡담, 위 카테고리에 속하지 않는 질문
    [이전 대화 활용]
    ※ '이전 대화'는 이 지시문이 아니라, 마지막 질문보다 앞에 사용자와 AI가 실제로 주고받은 메시지만을 말한다.
    1. 먼저 이전 대화가 있는지 확인할 것. 사용자 메시지가 마지막 질문 하나뿐이면 이전 대화가 없는 것이다.
       이전 대화가 없는데 지시어나 생략 때문에 가리킬 대상이 필요한 질문이면, 가리킬 대상이 없으므로 general_agent로 보낼 것.
    2. 이전 대화가 있고 마지막 질문에 대상이 생략되거나 지시어가 있으면, 이전 대화에서 그 대상을 찾아 해당 주제의 에이전트로 보낼 것.
    3. 마지막 질문이 이전 시점을 가리키면 바로 앞 대화가 아니라 그 시점의 주제를 찾을 것. 그 사이에 인사·잡담만 있었다면 잡담이 아니라 그 전의 주제를 가리키는 것으로 본다.
    4. 가리키는 대상이 이전 대화의 여러 주제 중 어느 것인지 모호하면 general_agent로 보내지 말고, 이전 대화에서 가장 최근에 다룬 캐릭터·시세·공략 주제를 따를 것.
    5. 단, 마지막 질문이 새로운 주제를 분명히 담고 있으면 이전 대화보다 마지막 질문을 우선할 것.
    [작업 계획 규칙]
    - 대부분의 질문은 에이전트 하나로 답한다. steps에는 1개만 넣어라.
    - 한 질문 안에 서로 다른 에이전트의 정보가 모두 필요할 때만 2개를 넣어라. 최대 2개다.
    - 같은 에이전트가 답할 수 있는 대상이 여러 개(아이템 여러 개, 레이드 여러 개 등)여도 1개로 처리한다.
    - 한 단계의 결과가 있어야 다른 단계가 답할 수 있으면, 정보를 먼저 찾는 단계를 앞에 둔다.
    - 레이드 입장 레벨·입장 조건은 guide_agent가 공략 문서에서 찾는다.
    """)


def plan_from(result) -> list:
    """라우터 출력을 실행할 계획으로 정리한다: 같은 에이전트는 한 번만, 최대 MAX_PLAN_STEPS개"""
    plan, seen = [], set()
    for step in result.steps:
        if step.agent not in seen:
            seen.add(step.agent)
            plan.append({"agent": step.agent, "task": step.task})
    kept, dropped = plan[:MAX_PLAN_STEPS], plan[MAX_PLAN_STEPS:]
    if dropped:
        # 잘린 부분을 조용히 빠뜨리거나 지어내지 않도록, 마지막 단계에게 따로 물어봐 달라고 안내하게 한다
        dropped_text = "; ".join(s["task"] for s in dropped)
        print(f"✂️ [Plan Truncated] 최대 {MAX_PLAN_STEPS}단계를 넘어 제외: {dropped_text}")
        kept[-1] = {**kept[-1], "task": kept[-1]["task"] +
                    f" (이번에 처리하지 못하는 부분: {dropped_text} → 이 부분은 조회하거나 지어내지 말고, 따로 다시 물어봐 달라고 안내할 것)"}
    return kept

def router(state: AgentState):
    """
    [라우터 노드]
    사용자의 최근 질문을 분석하여 어떤 전문 에이전트로 보낼지 결정합니다.
    """
    messages = trimmer.invoke(state["messages"])
    
    print("\n🔀 [NODE ENTER] Router 진입")
    print(f"📦 [State] 현재 보유한 총 메시지 수: {len(messages)}개")
    if messages:
        print(f"💬 [Latest Message] 마지막 입력: {messages[-1].content}")
    
    plan = None
    for attempt in range(1, ROUTER_MAX_ATTEMPTS + 1):
        try:
            result = router_llm.invoke([router_system] + messages)
            # 응답이 비었거나 스키마에 맞는 값을 못 받은 경우도 실패로 본다
            if result is None:
                raise ValueError("구조화 출력이 비어 있음 (result=None)")
            plan = plan_from(result)
            if not plan:
                raise ValueError("작업 계획이 비어 있음 (steps=[])")
            break
        except ModelError as e:
            # LLM 호출 자체의 실패(429, 인증 오류, 서버 장애 등)는 SDK가 이미 재시도했고,
            # general_agent도 같은 LLM을 쓰므로 재시도·폴백 없이 바로 올려서 main.py에서 안내한다
            print(f"🚨 [Router LLM Error] LLM 호출 실패, 재시도·폴백 없이 중단 - {type(e).__name__}")
            raise
        except Exception as e:
            print(f"⚠️ [Router Error] 분류 실패 ({attempt}/{ROUTER_MAX_ATTEMPTS}회차) - {type(e).__name__}: {e}")

    # 재시도까지 실패하면 서비스를 멈추지 않고 일반 대화 에이전트로 보낸다
    if not plan:  # None(예외) 또는 빈 계획
        plan = [{"agent": ROUTER_FALLBACK, "task": ""}]
        print(f"🛟 [Router Fallback] 분류에 실패해 기본 경로로 보냅니다 ➡️ {ROUTER_FALLBACK}")

    destination = plan[0]["agent"]
    print(f"🎯 [Router Decision] 작업 계획 ➡️ {' → '.join(s['agent'] for s in plan)}")
    print("-" * 50)

    return {"next_node": destination, "plan": plan, "step": 0}

def route_decision(state: AgentState):
    """라우터가 선택한 다음 노드 이름을 리턴하는 조건부 엣지 함수"""
    return state.get("next_node", "general_agent")


def system_with_task(system: SystemMessage, state: AgentState) -> SystemMessage:
    """계획이 2단계 이상이면 시스템 프롬프트 뒤에 이번 단계에서 할 일을 붙인다 (대화 기록에는 저장하지 않음)
    Gemini는 시스템 메시지를 여러 개 받지 못할 수 있어 하나로 합친다"""
    plan, step = state.get("plan", []), state.get("step", 0)
    if len(plan) < 2 or step >= len(plan):
        return system
    steps_text = "\n".join(f"        {i + 1}단계 {s['agent']}: {s['task']}" for i, s in enumerate(plan))
    # [synthesizer] 합치기는 마지막에 따로 하므로 모든 단계가 같은 역할(자기 담당만 확인)을 받는다
    return SystemMessage(content=system.content + f"""
    [이번 질문의 작업 계획]
    이 질문은 여러 에이전트가 순서대로 나눠 처리한다.
{steps_text}
    너는 {step + 1}단계 담당이다: {plan[step]['task']}
    너는 네 담당 부분만 툴로 확인해 결과를 정리해라. 다른 단계의 담당 내용은 다루지 마라. 최종 답변은 모든 단계가 끝난 뒤 따로 합쳐서 만든다.
    """)


def agent_input(system: SystemMessage, state: AgentState, messages: list) -> list:
    """에이전트 LLM에 넣을 메시지 목록을 만든다.
    2단계 이상 계획에서 앞 단계 답변(AI 메시지)으로 대화가 끝나 있으면, LLM이 그 답변을 이어 쓰기만 하고
    자기 단계의 툴을 부르지 않는 문제가 있어(측정 중 확인), 이번 단계 지시를 사용자 메시지로 덧붙인다 (대화 기록에는 저장하지 않음)"""
    full = [system_with_task(system, state)] + messages
    plan, step = state.get("plan", []), state.get("step", 0)
    last = messages[-1] if messages else None
    if len(plan) >= 2 and 0 < step < len(plan) and last is not None and last.type == "ai" and not last.tool_calls:
        full.append(HumanMessage(content=f"[작업 계획 {step + 1}단계] {plan[step]['task']}"))
    return full


def finish_step(state: AgentState, response) -> dict:
    """에이전트 응답을 상태에 반영한다. 툴 호출 없이 답을 냈으면 이 단계가 끝난 것이므로 다음 단계로 넘긴다"""
    if response.tool_calls:
        return {"messages": [response]}
    return {"messages": [response], "step": state.get("step", 0) + 1}


def after_agent(tools_node: str):
    """에이전트 다음 경로: 툴 호출이 있으면 툴 노드, 계획이 남았으면 다음 에이전트,
    끝났으면 2단계 이상은 synthesizer(합치기), 1단계는 종료"""
    def decide(state: AgentState):
        if tools_node and tools_condition(state) == "tools":
            return tools_node
        plan, step = state.get("plan", []), state.get("step", 0)
        if step < len(plan):
            print(f"➡️ [Plan Next] {step + 1}단계로 이동 ➡️ {plan[step]['agent']}")
            return plan[step]["agent"]
        return "synthesizer" if len(plan) >= 2 else END
    return decide


# --- [1] 마켓 전문 에이전트 ---
market_tools = [get_market_options, search_market_items]
market_llm = llm.bind_tools(market_tools)
market_system = SystemMessage(content=
    """
    너는 로스트아크 경매장 및 시세 조회 전문 에이전트야.
    [매우 중요한 규칙]
        1. 시세나 가격 정보를 답변하거나 비교할 때는 반드시 **'골드'** 단위를 사용해라.
        2. 이전 대화 기록이나 툴의 결과 데이터에 있는 수치 및 단위를 임의로 변경하여 출력하는 행위를 엄격히 금지한다.
        모든 가격 및 수치 데이터(골드, 수량 등)를 표기할 때는 **천 단위 구분 쉼표(,)를 포함하여 표기할 것** (예: `2,191 골드`, `100개`)
        3. 사용자가 입력한 아이템 이름을 절대 축약하거나, 변형하지 마라.
    """
    )

def market_agent(state: AgentState):
    """마켓(경매장/시세) 관련 질문을 전담하여 처리하는 노드"""
    messages = trimmer.invoke(state["messages"])
    
    print("\n🛒 [NODE ENTER] Market Agent 진입")
    print(f"📦 [State Messages 로그]")
    for idx, msg in enumerate(messages):
        print(f"   [{idx}] {msg.type}: {msg.content}")

    full_messages = agent_input(market_system, state, messages)
    response = market_llm.invoke(full_messages)
    
    print(f"🤖 [Market Agent Response] 생성된 답변/툴호출: {response.content} (Tool Calls: {response.tool_calls})")
    print("-" * 50)
    
    return finish_step(state, response)


# --- [2] 캐릭터 전문 에이전트 ---
character_tools = [get_character_profile]
character_llm = llm.bind_tools(character_tools)
character_system = SystemMessage(content=
    """너는 로스트아크 캐릭터 정보 조회 전문 에이전트야.
    [매우 중요한 규칙]
        1. 사용자가 입력한 캐릭터 닉네임을 절대 임의로 줄이거나, 축약하거나, 변형하지 마라.
        2. 툴(`get_character_profile`)을 호출할 때 `character_name` 인자에는 사용자가 입력한 닉네임 문자열 전체를 오탈자나 변형 없이 그대로 전달해야 한다.
        3. 닉네임에 특정 서버 이름이나 다른 단어가 포함되어 있더라도, 사용자가 지정한 닉네임 자체를 훼손하지 말고 그대로 사용할 것.
    """                             
    )

def character_agent(state: AgentState):
    """캐릭터 정보 조회 관련 질문을 전담하여 처리하는 노드"""
    messages = trimmer.invoke(state["messages"])
    
    print("\n🛡️ [NODE ENTER] Character Agent 진입")
    print(f"📦 [State Messages 로그]")
    for idx, msg in enumerate(messages):
        print(f"   [{idx}] {msg.type}: {msg.content}")

    full_messages = agent_input(character_system, state, messages)
    response = character_llm.invoke(full_messages)
    
    print(f"🤖 [Character Agent Response] 생성된 답변/툴호출: {response.content} (Tool Calls: {response.tool_calls})")
    print("response.usage_metadata : ", response.usage_metadata)
    print("-" * 50)
    
    return finish_step(state, response)


# --- [3] 가이드 에이전트 ---
guide_tools = [search_game_guide]
guide_llm = llm.bind_tools(guide_tools)
guide_system = SystemMessage(content=
    """너는 로스트아크 레이드 공략 및 게임 가이드 전문 에이전트야.
    [매우 중요한 규칙]
    1. 사용자의 질문에 답변할 때 필요한 정보가 있다면 반드시 tools를 사용하여 공략 문서를 검색해라.
    2. 만약 첫 번째 검색 결과로 얻은 정보가 부족하거나 추가로 확인해야 할 패턴이 있다면, 툴을 한 번 더 호출하여 정확한 정보를 찾아라.
    3. 툴을 통해 얻은 참고 자료를 최우선으로 활용하여, 레이드 패턴 및 공략 팁을 보기 쉽게 정리해 안내할 것.
    4. 참고 자료에 없는 내용이거나 불확실한 경우, 지어내지 말고 모른다고 솔직하게 답변해라.
    5. 툴의 검색어(query)를 만들 때 사용자가 말한 레이드·보스 이름을 그대로 넣어라.
       다른 레이드·보스 이름으로 바꾸거나, 사용자가 말하지 않은 레이드 이름을 덧붙이지 마라. 관문 번호와 줄 수도 그대로 유지할 것.
    """
)

def guide_agent(state: AgentState):
    """
    [가이드/RAG 노드]
    사용자의 최근 질문을 기반으로 벡터 DB에서 공략 문서를 검색(Retrieval)하고,
    문맥을 프롬프트에 주입하여 답변을 생성(Generation)합니다.
    """
    messages = trimmer.invoke(state["messages"])
    print("\n📚 [NODE ENTER] Guide Agent 진입")
    for idx, msg in enumerate(messages):
        print(f"   [{idx}] {msg.type}: {msg.content}")

    full_messages = agent_input(guide_system, state, messages)
    response = guide_llm.invoke(full_messages)
    
    print(f"🤖 [Guide Agent Response] 생성된 답변/툴호출: {response.content} (Tool Calls: {response.tool_calls})")
    print("-" * 50)

    return finish_step(state, response)

# --- [4] 일반 에이전트 ---
general_system = SystemMessage(content="너는 로스트아크 친절한 AI 비서야. 일반적인 질문에 답변해 줘.")

def general_agent(state: AgentState):
    """일반 대화 등 툴이 필요 없는 질문을 처리하는 노드"""
    messages = trimmer.invoke(state["messages"])
    
    print("\n💬 [NODE ENTER] General Agent 진입")
    print(f"📦 [State Messages 로그]")
    for idx, msg in enumerate(messages):
        print(f"   [{idx}] {msg.type}: {msg.content}")

    full_messages = agent_input(general_system, state, messages)
    # 💡 general_llm 대신 기본 llm을 사용하도록 수정
    response = llm.invoke(full_messages)
    
    print(f"🤖 [General Agent Response] 생성된 답변: {response.content}")
    print("-" * 50)
    
    return finish_step(state, response)


# --- [5] 합치기 노드 (synthesizer) ---
# 2단계 이상 계획에서 마지막 에이전트가 조회와 합치기를 같이 하면, 대화에 남은 앞 단계 답을
# '이미 사용자에게 한 말'로 보고 빠뜨리거나("앞에서 설명한") 언급하는 문제가 있어, 합치기만 하는 노드로 나눈다
synthesizer_system = SystemMessage(content=
    """너는 로스트아크 AI 비서의 최종 답변 작성자야.
    사용자 질문을 여러 에이전트가 나눠 처리했고, 너는 그 단계별 결과(자료)를 받아 사용자에게 보낼 답변 하나로 정리한다.
    사용자는 단계별 결과를 본 적이 없고, 네 답변만 보게 된다.
    [매우 중요한 규칙]
    1. 모든 단계 결과를 빠짐없이 답변에 포함해라. 어느 단계의 내용도 생략하거나 '앞에서 설명했다'고 넘기지 마라.
    2. 자료에 없는 내용은 덧붙이지 마라. 수치·이름·단위는 자료에 있는 그대로 쓸 것.
    3. 단계 결과에 '찾을 수 없음', '문서 없음', '확인 필요', '다시 물어봐 달라' 같은 안내가 있으면 빼지 말고 그대로 전달해라.
       작업 계획에 '이번에 처리하지 못하는 부분'이 있으면 그 부분은 따로 다시 물어봐 달라고 안내해라.
    4. 질문이 여러 결과를 함께 따져야 답할 수 있으면, 자료에 있는 값만으로 판단해 결론을 써라.
    5. '단계', '에이전트', '작업 계획', '자료' 같은 내부 처리 표현은 답변에 쓰지 마라.
    """
)


def content_text(content) -> str:
    """Gemini는 답변을 [{'type': 'text', 'text': ...}] 같은 리스트로 줄 때가 있어 문자열로 합친다"""
    if isinstance(content, list):
        return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    return content or ""


def synthesis_input(state: AgentState) -> list:
    """이번 턴의 질문과 단계별 결과를 대화가 아니라 '자료' 하나(사용자 메시지)로 정리한다 (대화 기록에는 저장하지 않음)"""
    messages, plan = state["messages"], state.get("plan", [])
    # 이번 턴 범위: 마지막 사용자 질문 이후 메시지
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    question = content_text(messages[last_human].content)
    # 단계별 결과: 툴 호출 없이 낸 AI 답변 (finish_step이 이 답변마다 단계를 하나씩 넘기므로 계획 순서와 짝이 맞는다)
    results = [content_text(m.content) for m in messages[last_human + 1:] if m.type == "ai" and not m.tool_calls]

    parts = [f"[사용자 질문]\n{question}",
             "[작업 계획]\n" + "\n".join(f"{i + 1}단계 {s['agent']}: {s['task']}" for i, s in enumerate(plan))]
    for i, s in enumerate(plan):
        result = results[i] if i < len(results) else "(결과 없음)"
        parts.append(f"[{i + 1}단계 결과 - {s['agent']}]\n{result}")
    return [synthesizer_system, HumanMessage(content="\n\n".join(parts))]


def synthesizer(state: AgentState):
    """모든 단계 결과를 받아 툴 없이 최종 답변 하나로 합치는 노드 (2단계 이상 계획에서만 실행)"""
    print("\n🧩 [NODE ENTER] Synthesizer 진입")
    full_messages = synthesis_input(state)
    print(f"📦 [Synthesizer Input] {full_messages[-1].content}")

    response = llm.invoke(full_messages)

    print(f"🤖 [Synthesizer Response] 생성된 최종 답변: {response.content}")
    print("-" * 50)

    return {"messages": [response]}


# LangGraph 워크플로우 조립
workflow = StateGraph(AgentState)

workflow.add_node("router", router)
workflow.add_node("market_agent", market_agent)
workflow.add_node("market_tools", ToolNode(market_tools))
workflow.add_node("character_agent", character_agent)
workflow.add_node("character_tools", ToolNode(character_tools))
workflow.add_node("guide_agent", guide_agent)
workflow.add_node("guide_tools", ToolNode(guide_tools))
workflow.add_node("general_agent", general_agent)
workflow.add_node("synthesizer", synthesizer)

workflow.add_edge(START, "router")
workflow.add_edge("synthesizer", END)

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "market_agent": "market_agent",
        "character_agent": "character_agent",
        "general_agent": "general_agent",
        "guide_agent" : "guide_agent",
    }
)

# [Planner] 에이전트가 답을 내면 바로 끝내지 않고, 계획에 다음 단계가 남았으면 그 에이전트로 보낸다
AGENT_NODES = ["market_agent", "character_agent", "guide_agent", "general_agent"]
for agent_name, tools_node in [("market_agent", "market_tools"), ("character_agent", "character_tools"),
                               ("guide_agent", "guide_tools"), ("general_agent", None)]:
    paths = {name: name for name in AGENT_NODES if name != agent_name}
    paths[END] = END
    paths["synthesizer"] = "synthesizer"  # 2단계 이상 계획이 끝나면 합치기 노드로
    if tools_node:
        paths[tools_node] = tools_node
        workflow.add_edge(tools_node, agent_name)
    workflow.add_conditional_edges(agent_name, after_agent(tools_node), paths)

# [대화 기록 영구 저장] MemorySaver(메모리) 대신 SQLite 파일에 저장해 서버를 다시 시작해도 대화가 남게 한다
CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH")
if not CHECKPOINT_DB_PATH:
    raise RuntimeError("🚨 [CHECKPOINT CONFIG] .env에 CHECKPOINT_DB_PATH를 설정해야 합니다 (.env.example 참고)")
checkpoint_path = Path(__file__).resolve().parent.parent / CHECKPOINT_DB_PATH  # ai-agent 폴더 기준 상대 경로
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
print(f"💾 [CHECKPOINT] 대화 기록 저장 위치: {checkpoint_path}")
# FastAPI가 요청을 여러 스레드에서 처리하므로 check_same_thread=False (SqliteSaver가 내부 lock으로 읽기·쓰기를 보호)
checkpoint_conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
checkpointer = SqliteSaver(checkpoint_conn)
app = workflow.compile(checkpointer=checkpointer)
# app = workflow.compile()