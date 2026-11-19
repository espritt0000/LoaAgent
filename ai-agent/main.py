from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_core.exceptions import (
    ModelRateLimitError,
    ModelAuthenticationError,
    ModelPermissionDeniedError,
    ModelAPIError,
    ModelTimeoutError,
    ModelConnectionError,
)
from agent.agent import app as agent_app

fastapi_app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    reply: str

class HistoryMessage(BaseModel):
    role: str     # "user" 또는 "ai"
    content: str

class HistoryResponse(BaseModel):
    messages: list[HistoryMessage]

# 화면에 다시 보여 줄 최근 메시지 수 (저장소에는 전체가 남아 있음)
HISTORY_LIMIT = 50


def to_text(content) -> str:
    """Gemini는 답변을 [{'type': 'text', 'text': ...}] 같은 리스트로 줄 때가 있어 문자열로 합친다"""
    if isinstance(content, list):
        return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    return content

@fastapi_app.post("/chat", response_model=ChatResponse)
def chat_endpoint(chatRequest: ChatRequest):
    print(f"SpringBoot Request : {chatRequest.message}")

    try:
        initial_state = {
            "messages": [HumanMessage(content=chatRequest.message)]
        }
        thread_id = chatRequest.thread_id
        print("thread_id : " + thread_id)
        config ={"configurable": {"thread_id": thread_id},
                "recursion_limit": 15
        }
        result = agent_app.invoke(initial_state, config=config)

        current_state = agent_app.get_state(config)
        print(f"[DEBUG] 현재 저장된 메시지 개수: {len(current_state.values.get('messages', []))}")

        
        message = result.get("messages",[])
        if message:
            final_message = message[-1]
            # 💡 구글 Gemini 특유의 리스트 형태 응답 처리 (텍스트 덩어리 추출)
            ai_reply = to_text(final_message.content)
        else:
            ai_reply = "답변을 생성하지 못했습니다."

    # LLM 오류는 LangChain 공통 예외(langchain_core.exceptions)로 구분한다 (모델을 바꿔도 그대로 사용 가능)
    except ModelRateLimitError as e:
        print(f"🚦 [LLM Rate Limit] 사용량 제한(429) - {type(e).__name__}")
        ai_reply = "요청이 많아 잠시 이용이 제한됐습니다. 1~2분 후 다시 시도해주세요."
    except (ModelAuthenticationError, ModelPermissionDeniedError) as e:
        # 키 값이 로그에 남지 않도록 예외 이름만 출력한다
        print(f"🔑 [LLM Auth Error] API 키 또는 권한 확인 필요 - {type(e).__name__}")
        ai_reply = "AI 서비스 설정에 문제가 있습니다. 관리자에게 문의해주세요."
    except (ModelAPIError, ModelTimeoutError, ModelConnectionError) as e:
        print(f"🌩️ [LLM Server Error] LLM 서버 장애 또는 연결 문제 - {type(e).__name__}")
        ai_reply = "AI 서비스가 일시적으로 불안정합니다. 잠시 후 다시 시도해주세요."
    except Exception as e:
        print(f"에이전트 실행 중 오류 발생: {type(e).__name__}: {e}")
        ai_reply = "AI 에이전트 처리 중 오류가 발생했습니다."

    return ChatResponse(reply=ai_reply)


@fastapi_app.get("/history", response_model=HistoryResponse)
def history_endpoint(thread_id: str):
    """저장된 대화에서 사용자 질문과 최종 AI 답변만 골라 돌려준다 (새로고침 후 화면 복원용)"""
    state = agent_app.get_state({"configurable": {"thread_id": thread_id}})
    messages = []
    for msg in state.values.get("messages", []):
        if msg.type == "human":
            role = "user"
        elif msg.type == "ai" and not msg.tool_calls:
            role = "ai"
        else:
            continue  # 툴 호출 요청과 툴 결과는 화면에 보여 주지 않는다
        text = to_text(msg.content)
        if text:
            messages.append(HistoryMessage(role=role, content=text))
    print(f"📜 [History] thread_id={thread_id}, 메시지 {len(messages)}개 중 최근 {min(len(messages), HISTORY_LIMIT)}개 반환")
    return HistoryResponse(messages=messages[-HISTORY_LIMIT:])
