import requests
from langchain_core.tools import tool
from utils.utils import handle_api_error, api_get_headers, API_TIMEOUT

# 캐릭터명 뒤에 붙어 들어올 수 있는 조사 (예: "니나브았아로 칠흑 갈 수 있어?")
# 긴 것부터 확인해야 "으로"를 "로"로 잘못 떼지 않는다
PARTICLES = ["이랑", "에게", "한테", "으로", "로", "랑", "은", "는", "이", "가", "을", "를", "의", "도", "와", "과"]


def strip_particle(name: str):
    """이름 끝의 조사를 하나 뗀 후보를 돌려준다. 뗄 조사가 없거나 남는 이름이 2자 미만이면 None"""
    for particle in PARTICLES:
        if name.endswith(particle) and len(name) - len(particle) >= 2:
            return name[:-len(particle)]
    return None


def request_profile(character_name: str):
    """Stove API로 프로필을 조회해 JSON을 돌려준다. 없는 캐릭터면 Stove가 200 + null을 주므로 None"""
    url = f"https://developer-lostark.game.onstove.com/armories/characters/{character_name}/profiles"
    response = requests.get(url, headers=api_get_headers(), timeout=API_TIMEOUT)
    # 4xx, 5xx 에러인 경우 HTTPError 예외 발생
    response.raise_for_status()
    return response.json()


@tool
def get_character_profile(character_name: str) -> dict:
    """
    로스트아크 캐릭터의 기본 프로필 정보(아이템 레벨, 전투 레벨, 직업, 전투력, 서버 등)를 조회합니다.
    
    Args:
        character_name (str): 조회할 캐릭터의 닉네임
    """
    try:
        data = request_profile(character_name)
        note = None

        # 사용자가 쓴 이름 그대로 먼저 조회하고, 없을 때만 끝의 조사를 떼고 한 번 더 조회한다
        # (조사인지 닉네임의 일부인지는 알 수 없으므로 추측하지 않고 Stove API로 확인. "하늘로" 같은 닉네임도 원래 이름이 먼저 성공)
        if not data:
            stripped = strip_particle(character_name)
            if stripped is None:
                return {"error": f"캐릭터 '{character_name}'을(를) 찾을 수 없습니다."}
            print(f"🔁 [Name Retry] '{character_name}' 없음 → 조사를 뗀 '{stripped}'로 다시 조회")
            data = request_profile(stripped)
            if not data:
                return {"error": f"캐릭터 '{character_name}', '{stripped}' 모두 찾을 수 없습니다. "
                                 f"같은 이름으로 다시 조회하지 말고, 캐릭터명을 확인해 달라고 안내할 것."}
            note = f"'{character_name}'(이)라는 캐릭터가 없어 조사를 뗀 '{stripped}'(으)로 조회한 결과입니다. 답변에 이 사실을 알릴 것."

        # LLM에게 전달할 핵심 데이터만 추출
        result = {
            "서버": data.get("ServerName"),
            "캐릭터명": data.get("CharacterName"),
            "전투레벨": data.get("CharacterLevel"),
            "직업": data.get("CharacterClassName"),
            "아이템레벨": data.get("ItemAvgLevel"),
            "전투력": data.get("CombatPower"),
            "길드명": data.get("GuildName")
        }
        if note:
            result["안내"] = note
        return result

    except requests.exceptions.HTTPError as e:
        # utils.py에 만들어둔 공통 에러 핸들러 호출 (target_name 전달)
        return handle_api_error(e.response.status_code, target_name=character_name)
        
    except requests.exceptions.RequestException as e:
        # 네트워크 끊김, 타임아웃 등 통신 자체의 오류
        return {"error": f"네트워크 통신 중 오류가 발생했습니다: {str(e)}"}
