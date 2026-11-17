import requests
from langchain_core.tools import tool
from utils.utils import handle_api_error, api_get_headers

@tool
def get_character_profile(character_name: str) -> dict:
    """
    로스트아크 캐릭터의 기본 프로필 정보(아이템 레벨, 전투 레벨, 직업, 전투력, 서버 등)를 조회합니다.
    
    Args:
        character_name (str): 조회할 캐릭터의 닉네임
    """
    url = f"https://developer-lostark.game.onstove.com/armories/characters/{character_name}/profiles"
    
    try:
        response = requests.get(url, headers=api_get_headers())
        
        # 4xx, 5xx 에러인 경우 HTTPError 예외 발생
        response.raise_for_status()
        
        data = response.json()
        if not data:
            return {"error": f"캐릭터 '{character_name}'의 데이터가 비어 있습니다."}
        
        # LLM에게 전달할 핵심 데이터만 추출
        return {
            "서버": data.get("ServerName"),
            "캐릭터명": data.get("CharacterName"),
            "전투레벨": data.get("CharacterLevel"),
            "직업": data.get("CharacterClassName"),
            "아이템레벨": data.get("ItemAvgLevel"),
            "전투력": data.get("CombatPower"),
            "길드명": data.get("GuildName")
        }

    except requests.exceptions.HTTPError as e:
        # utils.py에 만들어둔 공통 에러 핸들러 호출 (target_name 전달)
        return handle_api_error(e.response.status_code, target_name=character_name)
        
    except requests.exceptions.RequestException as e:
        # 네트워크 끊김, 타임아웃 등 통신 자체의 오류
        return {"error": f"네트워크 통신 중 오류가 발생했습니다: {str(e)}"}

