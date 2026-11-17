import requests
from langchain_core.tools import tool
from dotenv import load_dotenv
from utils.utils import handle_api_error, api_get_headers

@tool
def get_market_options() -> dict:
    """
    로스트아크 거래소(Market)에서 사용할 수 있는 전체 카테고리 코드, 아이템 등급, 티어, 직업 목록 등의 옵션 정보를 조회합니다.
    검색하려는 아이템의 정확한 카테고리 코드를 모를 때 이 툴을 먼저 호출하여 구조를 파악할 수 있습니다.
    """
    url = "https://developer-lostark.game.onstove.com/markets/options"
    
    try:
        response = requests.get(url, headers=api_get_headers())
        
        # 4xx, 5xx 에러 발생 시 예외(HTTPError) 발생
        response.raise_for_status()
        
        data = response.json()
        if not data:
            return {"error": "마켓 옵션 데이터가 비어 있습니다."}
        
        # LLM이 카테고리 구조를 쉽게 파악할 수 있도록 데이터 반환
        return {
            "Categories": data.get("Categories", []),
            "ItemGrades": data.get("ItemGrades", []),
            "ItemTiers": data.get("ItemTiers", []),
            "Classes": data.get("Classes", [])
        }

    except requests.exceptions.HTTPError as e:
        # 전체 리스트나 옵션 조회이므로 target_name을 넘기지 않음 (기본 메시지 출력)
        return handle_api_error(e.response.status_code)
        
    except requests.exceptions.RequestException as e:
        return {"error": f"네트워크 통신 중 오류가 발생했습니다: {str(e)}"}


@tool
def search_market_items(item_name: str, category_code: int) -> list:
    """
    로스트아크 마켓에서 재련 재료(파괴석, 돌파석 등) 또는 특정 아이템의 시세 및 아이디(ID)를 검색합니다.
    
    Args:
        item_name (str): 
            검색할 아이템 이름. 
            [주의] 사용자가 입력한 아이템명을 임의로 축약하거나 변경하지 말고 원문 그대로 입력.
        category_code (int): 검색할 아이템의 카테고리 분류 코드.
    """
    
    url = "https://developer-lostark.game.onstove.com/markets/items"
    
    # API 명세에 따른 요청 Body (Payload) 구성
    payload = {
        "Sort": "GRADE",
        "CategoryCode": category_code,      # 특정 카테고리 코드 지정
        "CharacterClass": None,
        "ItemTier": None,           # 티어 지정 가능
        "ItemGrade": None,          # 등급 지정 가능
        "ItemName": item_name,      # 검색할 아이템 이름
        "PageNo": 1,
        "SortCondition": "ASC"
    }
    print("-" * 50)
    print(f"search_market_items Payload: {payload}")
    try:
        response = requests.post(url, headers=api_get_headers(), json=payload)
        print("response : ", response.json())
        print("-" * 50)
        # 4xx, 5xx 에러 발생 시 예외(HTTPError) 발생
        response.raise_for_status()
        
        data = response.json()
        items = data.get("Items", [])
        
        # LLM이 쉽게 이해할 수 있도록 핵심 데이터만 정제하여 반환
        result = []
        for item in items:
            result.append({
                "아이디(ID)": item.get("Id"),
                "아이템명": item.get("Name"),
                "등급": item.get("Grade"),
                "묶음단위": item.get("BundleCount"),
                "전일평균가": item.get("YDayAvgPrice"),
                "최근거래가": item.get("RecentPrice")                
            })
            
        return result if result else [{"error": f"'{item_name}'에 해당하는 검색 결과가 없습니다."}]

    except requests.exceptions.HTTPError as e:
        # 아이템 검색 중 발생한 에러이므로 target_name에 item_name을 전달하여 구체적인 메시지 반환
        error_dict = handle_api_error(e.response.status_code, target_name=item_name)
        return [error_dict] # 반환 타입이 리스트(list)이므로 딕셔너리를 리스트로 감싸서 반환
        
    except requests.exceptions.RequestException as e:
        return [{"error": f"네트워크 통신 중 오류가 발생했습니다: {str(e)}"}]