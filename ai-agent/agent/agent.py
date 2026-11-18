from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, trim_messages
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
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
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    # temperature=0
)

# 토큰 트림 설정
trimmer = trim_messages(
    max_tokens=12000,
    strategy="last",
    token_counter=llm,
    include_system=True,
    start_on="human",
)

class RouterOutput(BaseModel):
    """라우터가 분류해야 할 목적지 정의"""
    destination: Literal["character_agent", "market_agent", "guide_agent", "general_agent"] = Field(
        description=
            "질문 유형에 따른 전송 경로: "
            "1. character_agent: 특정 캐릭터의 기본 프로필(서버, 직업, 전투 레벨, 아이템 레벨, 전투력, 길드) 조회 관련 "
            "2. market_agent: 경매장, 시세, 가격, 아이템 구매/판매, 재료 시세 조회 관련 "
            "3. guide_agent: 레이드 공략, 군단장/어비스 던전 패턴, 보스 공략법, 게임 가이드, 게임 정보 및 육성법 지식 검색 관련 "
            "4. general_agent: 단순 인사, 일반적인 잡담 및 위 카테고리에 속하지 않는 질문"
    )

router_llm = llm.with_structured_output(RouterOutput)
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
    """)

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
    
    result = router_llm.invoke([router_system] + messages)
    print(f"🎯 [Router Decision] 선택된 다음 경로 ➡️ {result.destination}")
    print("-" * 50)
    
    return {"next_node": result.destination}

def route_decision(state: AgentState):
    """라우터가 선택한 다음 노드 이름을 리턴하는 조건부 엣지 함수"""
    return state.get("next_node", "general_agent")


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

    full_messages = [market_system] + messages
    response = market_llm.invoke(full_messages)
    
    print(f"🤖 [Market Agent Response] 생성된 답변/툴호출: {response.content} (Tool Calls: {response.tool_calls})")
    print("-" * 50)
    
    return {"messages": [response]}


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

    full_messages = [character_system] + messages
    response = character_llm.invoke(full_messages)
    
    print(f"🤖 [Character Agent Response] 생성된 답변/툴호출: {response.content} (Tool Calls: {response.tool_calls})")
    print("response.usage_metadata : ", response.usage_metadata)
    print("-" * 50)
    
    return {"messages": [response]}


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

    full_messages = [guide_system] + messages
    response = guide_llm.invoke(full_messages)
    
    print(f"🤖 [Guide Agent Response] 생성된 답변/툴호출: {response.content} (Tool Calls: {response.tool_calls})")
    print("-" * 50)

    return {"messages": [response]}

# --- [4] 일반 에이전트 ---
general_system = SystemMessage(content="너는 로스트아크 친절한 AI 비서야. 일반적인 질문에 답변해 줘.")

def general_agent(state: AgentState):
    """일반 대화 등 툴이 필요 없는 질문을 처리하는 노드"""
    messages = trimmer.invoke(state["messages"])
    
    print("\n💬 [NODE ENTER] General Agent 진입")
    print(f"📦 [State Messages 로그]")
    for idx, msg in enumerate(messages):
        print(f"   [{idx}] {msg.type}: {msg.content}")

    full_messages = [general_system] + messages
    # 💡 general_llm 대신 기본 llm을 사용하도록 수정
    response = llm.invoke(full_messages)
    
    print(f"🤖 [General Agent Response] 생성된 답변: {response.content}")
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

workflow.add_edge(START, "router")

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

workflow.add_conditional_edges(
    "market_agent",
    tools_condition,
    {
        "tools": "market_tools",
        END: END
    }
)
workflow.add_edge("market_tools", "market_agent")

workflow.add_conditional_edges(
    "character_agent",
    tools_condition,
    {
        "tools": "character_tools",
        END: END
    }
)
workflow.add_edge("character_tools", "character_agent")

workflow.add_conditional_edges(
    "guide_agent",
    tools_condition,
    {
        "tools": "guide_tools",
        END: END
    }
)
workflow.add_edge("guide_tools", "guide_agent")

workflow.add_edge("general_agent", END)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
# app = workflow.compile()