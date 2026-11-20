"""
LLM 오류 종류별 안내 문구 테스트 (main.py /chat)

실행 위치: ai-agent 폴더
실행 방법: python -m unittest test_llm_error_messages -v
- 라우터 LLM이 오류를 내도록 Mock으로 바꾸고, 실제 LangGraph 그래프와 chat_endpoint를 거쳐
  사용자에게 돌아가는 문구를 확인합니다. Gemini API는 호출하지 않습니다.
"""
import unittest
import httpx
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from langchain_core.exceptions import ModelAPIError
from langchain_google_genai.chat_models import GoogleRateLimitError, GoogleAuthenticationError
from langgraph.checkpoint.memory import MemorySaver

import agent.agent as agent_module
import main
from main import chat_endpoint, ChatRequest


class LLMErrorMessageTest(unittest.TestCase):
    def setUp(self):
        # trimmer가 토큰 계산 중 외부 호출을 하지 않도록 입력을 그대로 돌려주게 바꾼다
        trimmer_patch = patch.object(agent_module, "trimmer")
        mock_trimmer = trimmer_patch.start()
        mock_trimmer.invoke.side_effect = lambda messages: messages
        self.addCleanup(trimmer_patch.stop)

        # 테스트 대화가 실제 대화 기록 파일(SQLite)에 남지 않도록 메모리 저장소로 만든 그래프를 쓴다
        app_patch = patch.object(main, "agent_app", agent_module.workflow.compile(checkpointer=MemorySaver()))
        app_patch.start()
        self.addCleanup(app_patch.stop)

    def ask_with_router_error(self, error, thread_id):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = error
        with patch.object(agent_module, "router_llm", mock_llm):
            response = chat_endpoint(ChatRequest(message="세르카 1관문 기믹 알려줘", thread_id=thread_id))
        return response.reply, mock_llm.invoke.call_count

    def test_429면_사용량_제한_안내(self):
        reply, calls = self.ask_with_router_error(GoogleRateLimitError("429 RESOURCE_EXHAUSTED"), "t-429")
        self.assertEqual(reply, "요청이 많아 잠시 이용이 제한됐습니다. 1~2분 후 다시 시도해주세요.")
        self.assertEqual(calls, 1)  # 라우터가 재시도하지 않음

    def test_하루_한도_429면_남은_시간_안내(self):
        # 2026-11-19 실제 Gemini 3.5 응답에서 필요한 부분만 남긴 메시지
        daily = GoogleRateLimitError(
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', "
            "'violations': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '500'}]}, "
            "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '9354s'}]}}")
        reply, calls = self.ask_with_router_error(daily, "t-429-day")
        self.assertEqual(reply, "오늘 사용할 수 있는 AI 요청을 모두 사용했습니다. 약 2시간 35분 후 다시 시도해주세요.")
        self.assertEqual(calls, 1)

    def test_하루_한도인데_남은_시간이_없으면_초기화_후_안내(self):
        daily = GoogleRateLimitError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        reply, _ = self.ask_with_router_error(daily, "t-429-day-no-delay")
        self.assertEqual(reply, "오늘 사용할 수 있는 AI 요청을 모두 사용했습니다. 한도가 초기화된 뒤 다시 시도해주세요.")

    def test_인증_오류면_설정_문제_안내(self):
        reply, _ = self.ask_with_router_error(GoogleAuthenticationError("401 UNAUTHENTICATED"), "t-401")
        self.assertEqual(reply, "AI 서비스 설정에 문제가 있습니다. 관리자에게 문의해주세요.")

    def test_서버_오류면_일시적_불안정_안내(self):
        reply, _ = self.ask_with_router_error(ModelAPIError("503 UNAVAILABLE"), "t-503")
        self.assertEqual(reply, "AI 서비스가 일시적으로 불안정합니다. 잠시 후 다시 시도해주세요.")

    def test_timeout이면_시간_초과_안내_라우터_재시도_없음(self):
        # 실제 Gemini timeout은 httpx 예외로 올라온다 (ModelError가 아니라서 예전에는 라우터가 재시도·폴백했음)
        reply, calls = self.ask_with_router_error(httpx.ReadTimeout("timed out"), "t-timeout")
        self.assertEqual(reply, "AI 응답이 너무 오래 걸려 중단했습니다. 잠시 후 다시 시도해주세요.")
        self.assertEqual(calls, 1)

    def test_LLM에_timeout과_재시도_횟수가_지정됨(self):
        self.assertEqual(agent_module.llm.timeout, agent_module.LLM_TIMEOUT)
        self.assertEqual(agent_module.llm.max_retries, agent_module.LLM_MAX_RETRIES)

    def test_그_외_오류면_기존_문구(self):
        # 라우터가 LLM 오류가 아닌 예외를 내면 재시도 후 general_agent로 폴백한다.
        # general_agent의 LLM(llm)도 Mock으로 막아, 분류되지 않은 예외가 기존 문구로 처리되는지 확인한다
        mock_general_llm = MagicMock()
        mock_general_llm.invoke.side_effect = RuntimeError("unexpected")
        with patch.object(agent_module, "llm", mock_general_llm):
            reply, calls = self.ask_with_router_error(KeyError("unexpected"), "t-etc")
        self.assertEqual(reply, "AI 에이전트 처리 중 오류가 발생했습니다.")
        self.assertEqual(calls, 2)  # 라우터는 1회 재시도
        self.assertEqual(mock_general_llm.invoke.call_count, 1)  # 폴백으로 general_agent까지 감



class FailedTurnCleanupTest(unittest.TestCase):
    """LLM 오류로 답을 못 만든 턴은 대화 기록에서 빠지는지 확인한다 (답 없는 질문이 다음 맥락에 섞이지 않게)"""

    def setUp(self):
        trimmer_patch = patch.object(agent_module, "trimmer")
        trimmer_patch.start().invoke.side_effect = lambda messages: messages
        self.addCleanup(trimmer_patch.stop)
        self.graph = agent_module.workflow.compile(checkpointer=MemorySaver())
        app_patch = patch.object(main, "agent_app", self.graph)
        app_patch.start()
        self.addCleanup(app_patch.stop)
        self.general = MagicMock()
        # 호출마다 새 메시지를 돌려준다 (같은 객체를 돌려주면 id가 같아 이전 답변을 덮어씀)
        self.general.invoke.side_effect = lambda messages: AIMessage(content="답변")
        general_patch = patch.object(agent_module, "llm", self.general)
        general_patch.start()
        self.addCleanup(general_patch.stop)

    def ask(self, message, router_result=None, router_error=None):
        router = MagicMock()
        if router_error:
            router.invoke.side_effect = router_error
        else:
            router.invoke.return_value = router_result
        with patch.object(agent_module, "router_llm", router):
            return chat_endpoint(ChatRequest(message=message, thread_id="t-cleanup")).reply

    def stored(self):
        return [m.content for m in self.graph.get_state({"configurable": {"thread_id": "t-cleanup"}}).values.get("messages", [])]

    def test_429로_실패한_질문은_기록에서_빠진다(self):
        to_general = SimpleNamespace(steps=[SimpleNamespace(agent="general_agent", task="")])
        self.ask("첫 질문", router_result=to_general)
        self.ask("실패 질문", router_error=GoogleRateLimitError("429 RESOURCE_EXHAUSTED"))
        self.assertEqual(self.stored(), ["첫 질문", "답변"])

        self.ask("다음 질문", router_result=to_general)
        self.assertEqual(self.stored(), ["첫 질문", "답변", "다음 질문", "답변"])

    def test_라우터_이후_에이전트에서_실패해도_그_턴이_통째로_빠진다(self):
        self.general.invoke.side_effect = ModelAPIError("503 UNAVAILABLE")
        reply = self.ask("질문", router_result=SimpleNamespace(steps=[SimpleNamespace(agent="general_agent", task="")]))
        self.assertEqual(reply, "AI 서비스가 일시적으로 불안정합니다. 잠시 후 다시 시도해주세요.")
        self.assertEqual(self.stored(), [])

    def test_synthesizer에서_실패해도_그_턴이_통째로_빠진다(self):
        # 1단계(general_agent)는 답하고, 같은 llm을 쓰는 synthesizer 호출에서 503이 난다
        self.general.invoke.side_effect = [AIMessage(content="1단계 답변"), ModelAPIError("503 UNAVAILABLE")]
        character = MagicMock()
        character.invoke.return_value = AIMessage(content="2단계 답변")
        two_steps = SimpleNamespace(steps=[SimpleNamespace(agent="general_agent", task="인사"),
                                           SimpleNamespace(agent="character_agent", task="캐릭터 조회")])
        with patch.object(agent_module, "character_llm", character):
            reply = self.ask("질문", router_result=two_steps)
        self.assertEqual(reply, "AI 서비스가 일시적으로 불안정합니다. 잠시 후 다시 시도해주세요.")
        self.assertEqual(self.general.invoke.call_count, 2)
        self.assertEqual(self.stored(), [])


if __name__ == "__main__":
    unittest.main()
