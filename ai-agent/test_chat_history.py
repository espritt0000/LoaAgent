"""
대화 기록 영구 저장(SQLite)과 /history 응답 테스트

실행 위치: ai-agent 폴더
실행 방법: python -m unittest test_chat_history -v
- LLM은 Mock으로 바꾸고 임시 SQLite 파일을 쓴다. Gemini API는 호출하지 않는다.
- '새 대화' 버튼용 DELETE /history가 해당 계정의 대화만 지우는지도 확인한다.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

import agent.agent as agent_module
import main


def compile_graph(db_path):
    """서버 시작을 흉내 낸다: SQLite 파일에 새로 연결해 그래프를 만든다"""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return agent_module.workflow.compile(checkpointer=SqliteSaver(conn)), conn


class ChatHistoryPersistenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "history.sqlite")
        router = MagicMock()
        router.invoke.return_value = SimpleNamespace(destination="general_agent")
        general = MagicMock()
        general.invoke.return_value = AIMessage(content="안녕하세요!")
        trimmer = MagicMock()
        trimmer.invoke.side_effect = lambda messages: messages  # 토큰 계산 중 외부 호출 방지
        for name, mock in (("router_llm", router), ("llm", general), ("trimmer", trimmer)):
            p = patch.object(agent_module, name, mock)
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def test_서버를_다시_시작해도_대화가_남는다(self):
        config = {"configurable": {"thread_id": "user:tester"}}
        graph, conn = compile_graph(self.db_path)
        graph.invoke({"messages": [HumanMessage(content="안녕")]}, config=config)
        conn.close()  # 서버 종료

        graph, conn = compile_graph(self.db_path)  # 서버 재시작
        messages = graph.get_state(config).values["messages"]
        conn.close()
        self.assertEqual([m.content for m in messages], ["안녕", "안녕하세요!"])

    def test_계정마다_대화가_따로_저장된다(self):
        graph, conn = compile_graph(self.db_path)
        graph.invoke({"messages": [HumanMessage(content="A의 질문")]}, config={"configurable": {"thread_id": "user:a"}})
        other = graph.get_state({"configurable": {"thread_id": "user:b"}}).values.get("messages", [])
        conn.close()
        self.assertEqual(other, [])

    def test_새_대화를_누르면_내_대화만_지워진다(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        graph = agent_module.workflow.compile(checkpointer=saver)
        mine, other = {"configurable": {"thread_id": "user:a"}}, {"configurable": {"thread_id": "user:b"}}
        graph.invoke({"messages": [HumanMessage(content="A의 질문")]}, config=mine)
        graph.invoke({"messages": [HumanMessage(content="B의 질문")]}, config=other)

        with patch.object(main, "checkpointer", saver):
            main.delete_history_endpoint("user:a")

        self.assertEqual(graph.get_state(mine).values.get("messages", []), [])
        self.assertEqual(len(graph.get_state(other).values["messages"]), 2)  # 다른 계정 대화는 그대로
        graph.invoke({"messages": [HumanMessage(content="새 질문")]}, config=mine)
        self.assertEqual([m.content for m in graph.get_state(mine).values["messages"]], ["새 질문", "안녕하세요!"])
        conn.close()


class HistoryEndpointTest(unittest.TestCase):
    def history_of(self, messages):
        fake_app = MagicMock()
        fake_app.get_state.return_value = SimpleNamespace(values={"messages": messages})
        with patch.object(main, "agent_app", fake_app):
            return [(m.role, m.content) for m in main.history_endpoint("user:tester").messages]

    def test_질문과_최종_답변만_돌려준다(self):
        result = self.history_of([
            HumanMessage(content="세르카 1관문 기믹 알려줘"),
            AIMessage(content="", tool_calls=[{"name": "search_game_guide", "args": {"query": "세르카"}, "id": "c1"}]),
            ToolMessage(content="[검색 결과]", tool_call_id="c1"),
            AIMessage(content=[{"type": "text", "text": "1관문은 "}, {"type": "text", "text": "장판 패턴이 핵심이에요."}]),
        ])
        self.assertEqual(result, [("user", "세르카 1관문 기믹 알려줘"), ("ai", "1관문은 장판 패턴이 핵심이에요.")])

    def test_대화가_없으면_빈_목록(self):
        fake_app = MagicMock()
        fake_app.get_state.return_value = SimpleNamespace(values={})
        with patch.object(main, "agent_app", fake_app):
            self.assertEqual(main.history_endpoint("user:new").messages, [])

    def test_최근_메시지만_돌려준다(self):
        result = self.history_of([HumanMessage(content=str(i)) for i in range(main.HISTORY_LIMIT + 10)])
        self.assertEqual(len(result), main.HISTORY_LIMIT)
        self.assertEqual(result[0][1], "10")


if __name__ == "__main__":
    unittest.main()
