import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LOSTARK_API_KEY")

# Stove API 응답을 기다리는 최대 시간(초). 없으면 requests가 끝없이 기다려 FastAPI 작업 스레드가 묶인다
API_TIMEOUT = 10

# 공통 헤더를 생성해주는 함수 또는 상수 정의
def api_get_headers() -> dict:
    return {
        "Accept": "application/json",
        "Authorization": f"bearer {API_KEY}"
    }

# api 상태코드 4xx, 5xx 에러 출력
def handle_api_error(status_code: int, target_name: str = None) -> dict:
    """
    공통 API 상태 코드 에러 핸들러
    - target_name이 있으면 구체적인 에러 메시지 반환
    - target_name이 없으면(None) 일반적인 리소스 에러 메시지 반환
    """
    match status_code:
        case 404:
            if target_name:
                return {"error": f"'{target_name}'을(를) 찾을 수 없습니다."}
            else:
                return {"error": "요청하신 리소스(데이터)를 찾을 수 없습니다."}
                
        case 401 | 403:
            return {"error": "API 인증에 실패했습니다. 토큰을 확인해주세요."}
            
        case 429:
            return {"error": "요청 횟수 제한(Rate Limit)을 초과했습니다. 잠시 후 다시 시도해주세요."}
            
        case code if 500 <= code < 600:
            return {"error": f"API 서버 오류가 발생했습니다. (상태 코드: {code})"}
            
        case _:
            return {"error": f"알 수 없는 에러가 발생했습니다. (상태 코드: {status_code})"}

