from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent.agent import app as agent_app

fastapi_app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    reply: str

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
            raw_content = final_message.content
            
            # 💡 구글 Gemini 특유의 리스트 형태 응답 처리 (텍스트 덩어리 추출)
            if isinstance(raw_content, list):
                print("Context type: list")
                ai_reply = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in raw_content
                )
            else:
                ai_reply = raw_content  # 문자열 응답은 그대로 사용
        else:
            ai_reply = "답변을 생성하지 못했습니다."

    except Exception as e:
        print(f"에이전트 실행 중 오류 발생: {e}")
        ai_reply = "AI 에이전트 처리 중 오류가 발생했습니다."

    return ChatResponse(reply=ai_reply)