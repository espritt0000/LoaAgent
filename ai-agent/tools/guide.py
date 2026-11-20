import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CHROMA_PERSIST_DIR = str(BASE_DIR / os.getenv("CHROMA_PERSIST_DIR"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# 1. 툴 내부에서 사용할 임베딩 및 벡터 스토어(Chroma) 초기화
try:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
except Exception as e:
    print(f"⚠️ [Warning] 벡터 스토어 초기화 실패: {str(e)}")
    vector_store = None


# 검색어 단어와 문서 속 이름의 유사도가 이 값 이상이면 같은 이름으로 본다 (오타 허용, 예: 세르까 ↔ 세르카 0.67)
NAME_SIMILARITY = 0.6


def split_names(text: str):
    """제목·보스 이름을 단어로 나눈다 ("칠흑, 폭풍의 밤" → 칠흑 / 폭풍의, 1자 단어는 제외)"""
    return [word for word in re.split(r"[\s,&():]+", text) if len(word) >= 2]


def build_raid_index():
    """적재된 청크의 메타데이터로 레이드 표를 만든다 (사람이 따로 관리하는 목록 없음, 문서를 다시 적재할 필요 없음)
    RAIDS:       {source_file: {"title": 제목, "overview": 개요 섹션명, "gates": {관문 번호: 섹션명}}}
    TITLE_NAMES: {레이드 제목 단어: source_file}                 예: "세르카" → serka.md
    BOSS_NAMES:  {보스 이름 단어: [(source_file, 관문 번호), ...]}  예: "카멘" → [(mordum.md, 1)]
    """
    raids, title_names, boss_names = {}, {}, {}
    if vector_store is None:
        return raids, title_names, boss_names
    for meta in vector_store.get(include=["metadatas"])["metadatas"]:
        source, title, header = meta.get("source_file", ""), meta.get("title", ""), meta.get("Header 2", "")
        raid = raids.setdefault(source, {"title": title, "overview": None, "gates": {}})
        for word in split_names(title):
            title_names[word] = source

        match = re.match(r"(\d+)관문:\s*(.*)", header)
        if match:  # "2관문: 태초의 그림자 코르부스 툴 라크 주요 패턴 및 기믹"
            gate = int(match.group(1))
            raid["gates"][gate] = header
            for word in split_names(match.group(2).replace("주요 패턴 및 기믹", "")):
                boss_names.setdefault(word, []).append((source, gate))
        else:      # "개요 및 입장 정보"
            raid["overview"] = header

    # "심연의"처럼 여러 레이드의 보스 이름에 나오는 단어는 어느 레이드인지 정할 수 없어 뺀다
    ambiguous = [word for word, places in boss_names.items() if len({src for src, _ in places}) > 1]
    for word in ambiguous:
        del boss_names[word]
    if ambiguous:
        print(f"ℹ️ [Raid Index] 여러 레이드에 걸친 보스 단어는 이름 확인에서 제외: {ambiguous}")
    return raids, title_names, boss_names


# 서버 시작 시 한 번 만든다 (DBupdate.py로 문서를 바꾸면 서버를 다시 시작해야 반영됨)
RAIDS, TITLE_NAMES, BOSS_NAMES = build_raid_index()

# "2관문", "2 관문", "이관문", "두 번째 관문", "두번째 관문"
GATE_NUMBERS = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "첫": 1, "두": 2, "세": 3, "네": 4, "다섯": 5}


def boss_of(header: str) -> str:
    """섹션명에서 보스 이름만 남긴다 (예: 2관문: 나이트레아 주요 패턴 및 기믹 → 나이트레아)"""
    return re.sub(r"^\d+관문:\s*", "", header).replace(" 주요 패턴 및 기믹", "")


def find_gate(query: str):
    """검색어의 관문 번호를 돌려준다. 없으면 None"""
    match = re.search(r"(\d+)\s*관문", query)
    if match:
        return int(match.group(1))
    match = re.search(r"(첫|두|세|네|다섯)\s*번?\s*째\s*관문", query)
    if match:
        return GATE_NUMBERS[match.group(1)]
    # "이관문"은 앞이 공백이거나 문장 처음일 때만 (예: "세르카이관문"의 조사 '이'를 2관문으로 읽지 않기 위해)
    match = re.search(r"(?:^|\s)(일|이|삼|사|오)관문", query)
    if match:
        return GATE_NUMBERS[match.group(1)]
    return None


def name_score(query: str, name: str) -> float:
    """검색어 안에 이름이 그대로 있으면 1.0, 아니면 검색어 단어와의 가장 높은 유사도 (오타 허용)"""
    if name in query:
        return 1.0
    words = [word for word in re.split(r"[\s,]+", query) if len(word) >= 2]
    return max((SequenceMatcher(None, word, name).ratio() for word in words), default=0.0)


def find_raid(query: str):
    """검색어가 가리키는 레이드를 돌려준다: (source_file, 보스 이름으로 찾았으면 [(source, 관문), ...] 아니면 None, 찾은 이름)
    문서에 있는 이름이 없으면 None. 가장 점수가 높은 이름을 고르고, 같은 점수면 레이드 제목 > 긴 이름 순
    ("세르카"는 레이드 제목이자 1관문 보스 이름이라 레이드로 본다)"""
    candidates = [(name_score(query, name), 1, len(name), name, source, None) for name, source in TITLE_NAMES.items()]
    candidates += [(name_score(query, name), 0, len(name), name, places[0][0], places) for name, places in BOSS_NAMES.items()]
    if not candidates:
        return None
    score, _, _, name, source, boss_places = max(candidates, key=lambda c: c[:4])
    if score < NAME_SIMILARITY:
        return None
    return source, boss_places, name


def latest_question(state) -> str:
    """대화 기록에서 사용자가 이번 턴에 입력한 질문을 꺼낸다
    (2단계 계획의 단계 지시는 state에 저장되지 않는 임시 메시지라 여기 나오지 않음)"""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else ""
    return ""


def pick_raid(query: str, question: str):
    """LLM이 만든 검색어의 레이드와 사용자 원래 질문의 레이드 중 무엇을 쓸지 정한다
    - 질문에 레이드 이름이 없음(예: "그럼 2관문은?") → 검색어 그대로
    - 검색어의 이름이 질문에도 있음(예: 두 레이드를 비교하는 질문) → 검색어 그대로
    - 같은 레이드를 가리킴(예: 질문 "모르둠 1관문", 검색어 "칠흑 1관문") → 검색어 그대로
    - 그 밖에는 LLM이 이름을 바꾸거나 빠뜨린 것으로 보고 질문의 레이드를 쓴다
    (그래서 사용자가 말하지 않은 다른 레이드는 LLM이 스스로 검색할 수 없다)"""
    found = find_raid(query)
    from_user = find_raid(question) if question else None
    if from_user is None:
        return found
    if found and (found[0] == from_user[0] or name_score(question, found[2]) >= NAME_SIMILARITY):
        return found
    if found:
        print(f"🛡️ [Raid Fixed] 검색어의 '{found[2]}' → 사용자 질문의 '{from_user[2]}'")
    return from_user


def pick_gate(query: str, question: str, source: str):
    """검색어에 관문 번호가 없으면 사용자 질문에서 채운다
    단, 질문이 다른 레이드를 가리키면 채우지 않는다 (예: "세르카 2관문이랑 칠흑 개요"에서 칠흑 검색에 2관문을 붙이지 않기 위해)"""
    gate = find_gate(query)
    if gate is not None or not question:
        return gate
    from_user = find_raid(question)
    if from_user is None or from_user[0] == source:
        gate = find_gate(question)
        if gate is not None:
            print(f"🛡️ [Gate Filled] 검색어에 관문 번호가 없어 사용자 질문의 {gate}관문 사용")
    return gate


def retrieve(query: str, question: str = ""):
    """검색어로 공략 청크를 찾는다 (search_game_guide와 rag_eval.py가 같이 사용)
    성공하면 Document 목록, 검색하지 않고 안내해야 하면 안내 문자열을 돌려준다
    - question: 사용자가 이번 턴에 입력한 질문. 있으면 레이드·관문 번호를 검색어와 대조한다
      (LLM이 검색어를 만들며 레이드 이름을 바꾸거나 관문 번호를 빠뜨리는 경우를 막기 위해, rag_eval.py는 넘기지 않음)
    - 레이드·보스 이름이 없음 → 문서 없음 안내
    - 관문 번호 없음 → 그 레이드 문서 안에서만 유사도 검색
    - 관문 번호 있음 → 그 관문 청크를 유사도와 상관없이 넣고, 같은 레이드 개요를 보조로 붙임
      (임베딩은 "1관문"과 "2관문"을 거의 구분하지 못해, 순위에 맡기면 다른 관문이 1위가 됨)
    - 없는 관문, 보스와 관문 불일치 → 비슷한 문서를 주지 않고 안내
    """
    found = pick_raid(query, question)
    if found is None:
        print("🚫 [No Matching Raid] 검색어에서 문서에 있는 레이드·보스 이름을 찾지 못함")
        titles = " / ".join(sorted(raid["title"] for raid in RAIDS.values()))
        return ("검색어에서 공략 문서가 있는 레이드·보스 이름을 찾지 못했습니다. "
                f"공략 문서가 있는 레이드: {titles}. "
                "이 목록에 없는 레이드라면 공략 문서가 없다고 답하고, "
                "레이드 이름이 빠진 질문이라면 어느 레이드인지 물어보세요.")

    source, boss_places, name = found
    raid = RAIDS[source]
    gate = pick_gate(query, question, source)

    if gate is None:
        k = decide_k(query)
        print(f"🎯 [Raid Filter] '{name}' → {raid['title']} 문서 안에서 검색 (k={k})")
        return vector_store.similarity_search(query=query, k=k, filter={"source_file": source})

    if gate not in raid["gates"]:
        print(f"⚠️ [Gate Not Found] {raid['title']}에 {gate}관문 없음")
        return (f"공략 문서에서 '{raid['title']}'은(는) {max(raid['gates'])}관문까지 있습니다. "
                f"{gate}관문 공략 문서는 없다고 답할 것.")

    if boss_places and (source, gate) not in boss_places:
        boss_gates = ", ".join(f"{g}관문" for _, g in boss_places)
        print(f"⚠️ [Gate Mismatch] '{name}'은(는) {raid['title']} {boss_gates} 보스인데 {gate}관문을 물음")
        return (f"공략 문서에서 '{name}'은(는) '{raid['title']}' {boss_gates} 보스이고, "
                f"{gate}관문 보스는 '{boss_of(raid['gates'][gate])}'입니다. "
                "어느 쪽을 물은 것인지 사용자에게 확인할 것.")

    print(f"🎯 [Gate Filter] '{name}' → {raid['title']} {gate}관문 + 개요")
    docs = vector_store.similarity_search(
        query=query, k=1, filter={"$and": [{"source_file": source}, {"Header 2": raid["gates"][gate]}]})
    if raid["overview"]:
        docs += vector_store.similarity_search(
            query=query, k=1, filter={"$and": [{"source_file": source}, {"Header 2": raid["overview"]}]})
    return docs


def decide_k(query: str) -> int:
    """[동적 검색 전략] 검색어의 성격에 따라 가져올 문서 개수(k)를 정한다 (rag_eval.py에서도 같은 규칙 사용)"""
    if any(keyword in query for keyword in ["전체", "요약", "개요", "알려줘", "소개"]):
        return 3  # 전체적인 내용이 필요할 때는 넓게 탐색
    # 예전에는 "줄·카운터·저스트 가드" 같은 패턴 키워드에 k=1을 썼지만, 이 단어들은 여러 섹션에 공통으로 나와
    # 정답 청크가 2~3위로 밀리는 경우가 많아(rag_eval.py 측정) k=1 규칙을 없앴다
    return 2  # 기본값


@tool
def search_game_guide(query: str, state: Annotated[dict, InjectedState]) -> list:
    """
    로스트아크 레이드 관문 정보, 군단장 패턴, 보스 공략, 게임 가이드 문서에서 필요한 정보를 검색합니다.
    정보가 부족하거나 추가 확인이 필요한 패턴이 있다면 키워드를 바꾸어 다시 호출할 수 있습니다.
    
    Args:
        query (str): 
            검색할 공략 키워드 또는 질문 내용. 
            [주의] 사용자가 질문한 핵심 패턴이나 보스 이름, 공략 명칭을 포함하여 구체적으로 작성할 것.
        state: 그래프 상태. LangGraph(ToolNode)가 넣어 주며 LLM에게는 보이지 않는다 (사용자 원래 질문을 꺼내는 데 사용)
    """
    print("-" * 50)
    print(f"🔍 [Tool Call] search_game_guide 검색어: '{query}'")
    
    if not vector_store:
        return [{"error": "벡터 DB(Chroma)가 초기화되지 않았습니다."}]
    
    try:
        # 이름 확인, 레이드·관문 필터, 동적 k는 retrieve()에서 처리
        retrieved_docs = retrieve(query, latest_question(state))
        if isinstance(retrieved_docs, str):  # 문서 없음·없는 관문·보스와 관문 불일치 안내
            print("-" * 50)
            return [{"error": retrieved_docs}]
        
        if not retrieved_docs:
            return [{"error": f"'{query}'에 해당하는 공략 문서를 찾지 못했습니다."}]
        
        # LLM이 쉽게 이해하고 참고할 수 있도록 데이터 정제
        result = []
        for doc in retrieved_docs:
            result.append({
                "source_file": doc.metadata.get("source_file", "unknown"),
                "page_content": doc.page_content
            })
            
        print("result : ", result)
        print(f"📄 [Tool Result] 검색된 문서 수: {len(result)}개")
        print("-" * 50)
        
        return result

    except Exception as e:
        print(f"❌ [Error] 가이드 검색 중 오류 발생: {str(e)}")
        print("-" * 50)
        return [{"error": f"가이드 검색 중 오류가 발생했습니다: {str(e)}"}]

