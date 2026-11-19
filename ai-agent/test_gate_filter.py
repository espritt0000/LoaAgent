"""
관문 필터 테스트 (tools/guide.py의 관문 번호 찾기, 레이드 고르기, 없는 관문·보스 불일치 안내)

실행 위치: ai-agent 폴더
실행 방법: python -m unittest test_gate_filter -v
- 레이드 표(RAIDS 등)를 테스트용 값으로 바꾸고 벡터 검색은 Mock으로 바꾼다. Ollama·Chroma·Gemini를 호출하지 않는다.
"""
import unittest
from unittest.mock import patch, MagicMock

from tools import guide
from tools.guide import find_gate

RAIDS = {
    "serka.md": {"title": "고통의 마녀 세르카", "overview": "개요 및 입장 정보",
                 "gates": {1: "1관문: 고통의 마녀 세르카 주요 패턴 및 기믹", 2: "2관문: 태초의 그림자 코르부스 툴 라크 주요 패턴 및 기믹"}},
    "mordum.md": {"title": "칠흑, 폭풍의 밤", "overview": "개요 및 입장 정보",
                  "gates": {1: "1관문: 카멘 & 인페르나스 주요 패턴 및 기믹", 2: "2관문: 나이트레아 주요 패턴 및 기믹", 3: "3관문: 모르둠 주요 패턴 및 기믹"}},
}
TITLE_NAMES = {"고통의": "serka.md", "마녀": "serka.md", "세르카": "serka.md", "칠흑": "mordum.md", "폭풍의": "mordum.md"}
BOSS_NAMES = {"세르카": [("serka.md", 1)], "코르부스": [("serka.md", 2)], "카멘": [("mordum.md", 1)],
              "인페르나스": [("mordum.md", 1)], "나이트레아": [("mordum.md", 2)], "모르둠": [("mordum.md", 3)]}


class FindGateTest(unittest.TestCase):
    def test_관문_번호_표현(self):
        self.assertEqual(find_gate("세르카 2관문 패턴"), 2)
        self.assertEqual(find_gate("세르카 2 관문 패턴"), 2)
        self.assertEqual(find_gate("벨가르딘 이관문 공략"), 2)
        self.assertEqual(find_gate("칠흑 두 번째 관문"), 2)
        self.assertEqual(find_gate("칠흑 두번째 관문"), 2)
        self.assertEqual(find_gate("파멸의 성채 첫번째 관문"), 1)
        self.assertEqual(find_gate("칠흑 3관문"), 3)

    def test_관문_번호가_없으면_None(self):
        self.assertIsNone(find_gate("세르카 공략 요약"))
        self.assertIsNone(find_gate("세르카 240줄 패턴"))

    def test_조사_이를_2관문으로_읽지_않는다(self):
        self.assertIsNone(find_gate("세르카이관문"))  # 붙여 쓴 조사 '이'는 관문 번호로 보지 않음 (레이드 전체 검색으로 처리)


class RetrieveTest(unittest.TestCase):
    def setUp(self):
        self.store = MagicMock()
        # 필터 조건을 그대로 돌려줘서 어떤 범위로 검색했는지 확인한다
        self.store.similarity_search.side_effect = lambda query, k, filter: [("검색", k, str(filter))]
        for name, value in (("RAIDS", RAIDS), ("TITLE_NAMES", TITLE_NAMES), ("BOSS_NAMES", BOSS_NAMES), ("vector_store", self.store)):
            p = patch.object(guide, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_관문을_물으면_관문_청크와_개요를_준다(self):
        result = guide.retrieve("세르카 2관문 패턴")
        self.assertEqual(len(result), 2)
        self.assertIn("2관문: 태초의 그림자", result[0][2])  # 관문 청크가 먼저
        self.assertIn("개요 및 입장 정보", result[1][2])

    def test_레이드_제목이_보스_이름보다_우선한다(self):
        # "세르카"는 레이드 제목이자 1관문 보스 → 레이드로 보고 2관문 청크를 준다 (불일치 안내 아님)
        self.assertIsInstance(guide.retrieve("세르카 2관문 240줄"), list)

    def test_관문_번호가_없으면_레이드_문서_안에서만_검색(self):
        result = guide.retrieve("세르카 공략 요약")
        self.assertEqual(result, [("검색", 3, "{'source_file': 'serka.md'}")])

    def test_없는_관문은_검색하지_않고_안내(self):
        result = guide.retrieve("카멘 4관문 공략")
        self.assertIn("3관문까지 있습니다", result)
        self.store.similarity_search.assert_not_called()

    def test_보스와_관문이_맞지_않으면_확인을_요청(self):
        result = guide.retrieve("카멘 2관문 패턴")
        self.assertIn("1관문 보스", result)
        self.assertIn("나이트레아", result)
        self.store.similarity_search.assert_not_called()

    def test_보스와_관문이_맞으면_정상_검색(self):
        self.assertIsInstance(guide.retrieve("나이트레아 2관문 패턴"), list)

    def test_문서에_없는_레이드는_거절(self):
        self.assertIn("찾지 못했습니다", guide.retrieve("발탄 2관문 공략"))

    def test_오타가_있어도_가장_비슷한_레이드를_찾는다(self):
        result = guide.retrieve("세르까 2관문")  # 세르까 ↔ 세르카 0.67
        self.assertIn("serka.md", result[0][2])


if __name__ == "__main__":
    unittest.main()
