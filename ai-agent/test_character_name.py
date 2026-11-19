"""
캐릭터명 뒤에 붙은 조사 처리 테스트

실행 위치: ai-agent 폴더
실행 방법: python -m unittest test_character_name -v
- Stove API 응답을 Mock으로 바꿔 실제 API와 Gemini API를 호출하지 않는다.
- Stove API는 없는 캐릭터에 404가 아니라 200 + null을 돌려준다(2026-11-20 실제 응답으로 확인). Mock도 같은 형태로 만든다.
"""
import unittest
from unittest.mock import patch, MagicMock

from tools import character
from tools.character import get_character_profile, strip_particle

PROFILE = {"ServerName": "니나브", "CharacterName": "니나브았아", "CharacterLevel": 70,
           "CharacterClassName": "바드", "ItemAvgLevel": "1,770.00", "CombatPower": "5,340", "GuildName": "길드"}


def fake_stove(existing):
    """existing에 있는 이름만 프로필을 돌려주는 가짜 Stove API. 호출된 이름을 기록한다"""
    called = []

    def get(url, headers=None):
        name = url.split("/characters/")[1].split("/profiles")[0]
        called.append(name)
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = dict(PROFILE, CharacterName=name) if name in existing else None
        return response
    return get, called


class CharacterNameParticleTest(unittest.TestCase):
    def lookup(self, name, existing):
        get, called = fake_stove(existing)
        with patch.object(character.requests, "get", side_effect=get),                 patch.object(character, "api_get_headers", return_value={}):
            result = get_character_profile.invoke({"character_name": name})
        return result, called

    def test_조사가_붙은_이름은_조사를_떼고_다시_조회한다(self):
        result, called = self.lookup("니나브았아로", existing={"니나브았아"})
        self.assertEqual(called, ["니나브았아로", "니나브았아"])
        self.assertEqual(result["캐릭터명"], "니나브았아")
        self.assertIn("조사를 뗀", result["안내"])  # 다른 이름으로 조회했다는 사실을 답변에 알리게 함

    def test_조사처럼_끝나는_진짜_닉네임은_그대로_조회한다(self):
        result, called = self.lookup("하늘로", existing={"하늘로", "하늘"})
        self.assertEqual(called, ["하늘로"])  # 원래 이름이 먼저 성공하므로 다시 조회하지 않음
        self.assertEqual(result["캐릭터명"], "하늘로")
        self.assertNotIn("안내", result)

    def test_둘_다_없으면_시도한_이름을_모두_알려준다(self):
        result, called = self.lookup("없는캐릭으로", existing=set())
        self.assertEqual(called, ["없는캐릭으로", "없는캐릭"])  # 다시 조회는 한 번만
        self.assertIn("'없는캐릭으로', '없는캐릭' 모두 찾을 수 없습니다", result["error"])

    def test_뗄_조사가_없으면_다시_조회하지_않는다(self):
        result, called = self.lookup("니나브았아", existing=set())
        self.assertEqual(called, ["니나브았아"])
        self.assertIn("찾을 수 없습니다", result["error"])

    def test_조사_떼기_규칙(self):
        self.assertEqual(strip_particle("모코코으로"), "모코코")  # "으로"를 "로"보다 먼저 확인
        self.assertEqual(strip_particle("모코코랑"), "모코코")
        self.assertEqual(strip_particle("모코코이랑"), "모코코")
        self.assertIsNone(strip_particle("니나브았아"))  # 조사로 끝나지 않음
        self.assertIsNone(strip_particle("하로"))  # 떼면 1자만 남아 닉네임이 될 수 없음


if __name__ == "__main__":
    unittest.main()
