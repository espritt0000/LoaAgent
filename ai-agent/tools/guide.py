import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
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


def decide_k(query: str) -> int:
    """[동적 검색 전략] 검색어의 성격에 따라 가져올 문서 개수(k)를 정한다 (rag_eval.py에서도 같은 규칙 사용)"""
    if any(keyword in query for keyword in ["전체", "요약", "개요", "알려줘", "소개"]):
        return 3  # 전체적인 내용이 필요할 때는 넓게 탐색
    # 예전에는 "줄·카운터·저스트 가드" 같은 패턴 키워드에 k=1을 썼지만, 이 단어들은 여러 섹션에 공통으로 나와
    # 정답 청크가 2~3위로 밀리는 경우가 많아(rag_eval.py 측정) k=1 규칙을 없앴다
    return 2  # 기본값


@tool
def search_game_guide(query: str) -> list:
    """
    로스트아크 레이드 관문 정보, 군단장 패턴, 보스 공략, 게임 가이드 문서에서 필요한 정보를 검색합니다.
    정보가 부족하거나 추가 확인이 필요한 패턴이 있다면 키워드를 바꾸어 다시 호출할 수 있습니다.
    
    Args:
        query (str): 
            검색할 공략 키워드 또는 질문 내용. 
            [주의] 사용자가 질문한 핵심 패턴이나 보스 이름, 공략 명칭을 포함하여 구체적으로 작성할 것.
    """
    print("-" * 50)
    print(f"🔍 [Tool Call] search_game_guide 검색어: '{query}'")
    
    if not vector_store:
        return [{"error": "벡터 DB(Chroma)가 초기화되지 않았습니다."}]
    
    try:
        # 💡 [동적 검색 전략] 질문의 성격에 따라 가져올 문서 개수(k)를 유동적으로 결정
        k = decide_k(query)

        print(f"🎯 [동적 k 적용] 설정된 k값: {k}")

        # 💡 as_retriever().invoke() 대신 similarity_search() 직접 호출
        retrieved_docs = vector_store.similarity_search(query=query, k=k)
        
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

