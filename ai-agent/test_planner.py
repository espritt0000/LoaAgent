"""
Planner(작업 계획) 단위 테스트 - 에이전트 간 협업 14번

실행 위치: ai-agent 폴더
실행 방법: python -m unittest test_planner -v
- 라우터·에이전트 LLM을 Mock으로 바꾸고 실제 LangGraph 그래프를 MemorySaver로 실행합니다. Gemini API는 호출하지 않습니다.
"""
import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

import agent.agent as agent_module
from agent.agent import router, RouterOutput, PlanStep


def plan_of(*agents):
    return RouterOutput(steps=[PlanStep(agent=a, task=f"{a} 담당 작업") for a in agents])


def fake_agent_llm(name, calls):
    """호출될 때마다 받은 메시지를 기록하고, 툴 호출 없는 답변을 돌려주는 가짜 에이전트 LLM
    general_agent와 synthesizer는 같은 기본 llm을 쓰므로, 시스템 프롬프트로 어느 노드의 호출인지 구분한다"""
    mock = MagicMock()

    def answer(messages):
        caller = name
        if messages and messages[0].content == agent_module.synthesizer_system.content:
            caller = "synthesizer"
        calls.append((caller, messages))
        return AIMessage(content=f"{caller} 답변")
    mock.invoke.side_effect = answer
    return mock


class PlannerTest(unittest.TestCase):
    def setUp(self):
        trimmer = MagicMock()
        trimmer.invoke.side_effect = lambda messages: messages  # 토큰 계산 중 외부 호출 방지
        self.calls = []
        mocks = {"trimmer": trimmer,
                 "guide_llm": fake_agent_llm("guide_agent", self.calls),
                 "character_llm": fake_agent_llm("character_agent", self.calls),
                 "market_llm": fake_agent_llm("market_agent", self.calls),
                 "llm": fake_agent_llm("general_agent", self.calls)}
        for name, mock in mocks.items():
            p = patch.object(agent_module, name, mock)
            p.start()
            self.addCleanup(p.stop)

    def run_graph(self, router_result, question="질문"):
        router_llm = MagicMock()
        router_llm.invoke.return_value = router_result
        graph = agent_module.workflow.compile(checkpointer=MemorySaver())
        with patch.object(agent_module, "router_llm", router_llm):
            result = graph.invoke({"messages": [HumanMessage(content=question)]},
                                  config={"configurable": {"thread_id": "t"}, "recursion_limit": 15})
        return result

    def test_라우터는_같은_에이전트를_한_번만_최대_2단계로_정리한다(self):
        router_llm = MagicMock()
        router_llm.invoke.return_value = plan_of("guide_agent", "guide_agent", "market_agent", "character_agent")
        with patch.object(agent_module, "router_llm", router_llm):
            result = router({"messages": [HumanMessage(content="질문")]})
        self.assertEqual([s["agent"] for s in result["plan"]], ["guide_agent", "market_agent"])
        self.assertEqual(result["next_node"], "guide_agent")
        self.assertEqual(result["step"], 0)

    def test_2단계를_넘어_잘린_부분은_마지막_단계에_안내하게_한다(self):
        router_llm = MagicMock()
        router_llm.invoke.return_value = plan_of("guide_agent", "character_agent", "market_agent")
        with patch.object(agent_module, "router_llm", router_llm):
            result = router({"messages": [HumanMessage(content="질문")]})
        self.assertEqual([s["agent"] for s in result["plan"]], ["guide_agent", "character_agent"])
        self.assertIn("market_agent 담당 작업", result["plan"][-1]["task"])
        self.assertIn("따로 다시 물어봐", result["plan"][-1]["task"])
        self.assertNotIn("처리하지 못하는", result["plan"][0]["task"])

    def test_빈_계획이면_재시도_후_general_agent(self):
        router_llm = MagicMock()
        router_llm.invoke.side_effect = [RouterOutput(steps=[]), RouterOutput(steps=[])]
        with patch.object(agent_module, "router_llm", router_llm):
            result = router({"messages": [HumanMessage(content="질문")]})
        self.assertEqual(result["next_node"], "general_agent")
        self.assertEqual(router_llm.invoke.call_count, 2)

    def test_1단계_계획은_에이전트_하나만_실행하고_프롬프트도_그대로(self):
        result = self.run_graph(plan_of("market_agent"))
        self.assertEqual([name for name, _ in self.calls], ["market_agent"])
        system = self.calls[0][1][0]
        self.assertNotIn("작업 계획", system.content)
        self.assertEqual(result["messages"][-1].content, "market_agent 답변")

    def test_1단계_general_agent도_synthesizer를_거치지_않는다(self):
        result = self.run_graph(plan_of("general_agent"))
        self.assertEqual([name for name, _ in self.calls], ["general_agent"])
        self.assertEqual(result["messages"][-1].content, "general_agent 답변")

    def test_2단계_계획은_순서대로_실행하고_synthesizer가_마지막에_한_번_최종_답변(self):
        result = self.run_graph(plan_of("guide_agent", "character_agent"))
        self.assertEqual([name for name, _ in self.calls], ["guide_agent", "character_agent", "synthesizer"])
        self.assertEqual(result["messages"][-1].content, "synthesizer 답변")
        # 두 단계 에이전트 모두 자기 담당만 정리하라는 같은 역할을 받는다 (합치기는 synthesizer가 맡음)
        first_system, second_system = self.calls[0][1][0], self.calls[1][1][0]
        self.assertIn("1단계 담당", first_system.content)
        self.assertIn("2단계 담당", second_system.content)
        for system in (first_system, second_system):
            self.assertIn("네 담당 부분만", system.content)
            self.assertNotIn("최종 답변을 써라", system.content)

    def test_synthesizer는_질문과_두_단계_결과를_자료로_받는다(self):
        self.run_graph(plan_of("guide_agent", "market_agent"), question="공략이랑 시세 알려줘")
        synth_input = self.calls[-1][1]
        # 시스템 프롬프트 + 자료를 담은 사용자 메시지 하나 (앞 단계 답이 AI 메시지로 들어가지 않음)
        self.assertEqual(len(synth_input), 2)
        self.assertIsInstance(synth_input[1], HumanMessage)
        material = synth_input[1].content
        self.assertIn("[사용자 질문]\n공략이랑 시세 알려줘", material)
        self.assertIn("[1단계 결과 - guide_agent]\nguide_agent 답변", material)
        self.assertIn("[2단계 결과 - market_agent]\nmarket_agent 답변", material)

    def test_synthesizer는_이번_턴_결과만_받는다(self):
        router_llm = MagicMock()
        router_llm.invoke.side_effect = [plan_of("market_agent"), plan_of("guide_agent", "market_agent")]
        graph = agent_module.workflow.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "t"}, "recursion_limit": 15}
        with patch.object(agent_module, "router_llm", router_llm):
            graph.invoke({"messages": [HumanMessage(content="이전 질문")]}, config=config)
            graph.invoke({"messages": [HumanMessage(content="이번 질문")]}, config=config)
        material = self.calls[-1][1][1].content
        self.assertIn("이번 질문", material)
        self.assertNotIn("이전 질문", material)
        self.assertEqual(material.count("market_agent 답변"), 1)  # 이전 턴의 market 답변은 빠진다

    def test_잘린_단계_안내가_synthesizer_자료의_계획에_들어간다(self):
        self.run_graph(plan_of("guide_agent", "character_agent", "market_agent"))
        material = self.calls[-1][1][1].content
        self.assertIn("[작업 계획]", material)
        self.assertIn("따로 다시 물어봐", material)

    def test_2단계_에이전트에는_단계_지시를_사용자_메시지로_덧붙이고_기록에는_남기지_않는다(self):
        result = self.run_graph(plan_of("guide_agent", "character_agent"))
        second_input = self.calls[1][1]
        self.assertIsInstance(second_input[-1], HumanMessage)
        self.assertIn("[작업 계획 2단계]", second_input[-1].content)
        # 저장된 대화에는 질문, 두 단계 답변, 합친 최종 답변만 있다 (자료 메시지는 저장하지 않음)
        self.assertEqual([m.content for m in result["messages"]],
                         ["질문", "guide_agent 답변", "character_agent 답변", "synthesizer 답변"])


if __name__ == "__main__":
    unittest.main()
