"""
라우터 실패 대응(재시도 + general_agent 폴백) 단위 테스트

실행 위치: ai-agent 폴더 (router_eval.py와 같은 위치)
실행 방법: python -m unittest test_router_fallback -v
- router_llm을 가짜 객체(Mock)로 바꿔서 실행하므로 Gemini API를 호출하지 않습니다.
- agent.agent를 import하므로 router_eval.py와 같은 실행 환경(.env, 패키지)이 필요합니다.
"""
import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
from langchain_google_genai.chat_models import GoogleRateLimitError

import agent.agent as agent_module
from agent.agent import router, RouterOutput


def make_state(text="세르카 1관문 기믹 알려줘"):
    return {"messages": [HumanMessage(content=text)]}


class RouterFallbackTest(unittest.TestCase):
    def setUp(self):
        # trimmer가 토큰 계산 중 외부 호출을 하지 않도록 입력을 그대로 돌려주게 바꾼다
        trimmer_patch = patch.object(agent_module, "trimmer")
        mock_trimmer = trimmer_patch.start()
        mock_trimmer.invoke.side_effect = lambda messages: messages
        self.addCleanup(trimmer_patch.stop)

    def run_router_with(self, side_effect):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = side_effect
        with patch.object(agent_module, "router_llm", mock_llm):
            result = router(make_state())
        return result, mock_llm.invoke.call_count

    def test_정상_분류(self):
        result, calls = self.run_router_with([RouterOutput(destination="guide_agent")])
        self.assertEqual(result["next_node"], "guide_agent")
        self.assertEqual(calls, 1)

    def test_첫_시도_예외_후_재시도_성공(self):
        result, calls = self.run_router_with([
            RuntimeError("일시적 API 오류"),
            RouterOutput(destination="market_agent"),
        ])
        self.assertEqual(result["next_node"], "market_agent")
        self.assertEqual(calls, 2)

    def test_두_번_모두_예외면_general_agent(self):
        result, calls = self.run_router_with([
            RuntimeError("API 오류 1"),
            RuntimeError("API 오류 2"),
        ])
        self.assertEqual(result["next_node"], "general_agent")
        self.assertEqual(calls, 2)

    def test_두_번_모두_None이면_general_agent(self):
        result, calls = self.run_router_with([None, None])
        self.assertEqual(result["next_node"], "general_agent")
        self.assertEqual(calls, 2)

    def test_LLM_호출_오류면_재시도_폴백_없이_예외(self):
        # 429처럼 LLM 호출 자체가 실패하면 general_agent로 보내지 않고 바로 올린다
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [GoogleRateLimitError("429 RESOURCE_EXHAUSTED")]
        with patch.object(agent_module, "router_llm", mock_llm):
            with self.assertRaises(GoogleRateLimitError):
                router(make_state())
        self.assertEqual(mock_llm.invoke.call_count, 1)


if __name__ == "__main__":
    unittest.main()
