"""
FastAPI 내부 호출용 API Key 테스트

실행 위치: ai-agent 폴더
실행 방법: python -m unittest test_api_key -v
- FastAPI TestClient로 X-API-Key 헤더가 없거나 틀리면 401, 맞으면 통과하는지 확인한다.
- 그래프는 Mock으로 바꿔 Gemini API와 실제 대화 기록 파일을 쓰지 않는다. 키 값은 출력하지 않는다.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

import main


class ApiKeyTest(unittest.TestCase):
    def setUp(self):
        fake_app = MagicMock()
        fake_app.get_state.return_value = SimpleNamespace(values={})
        app_patch = patch.object(main, "agent_app", fake_app)
        app_patch.start()
        self.addCleanup(app_patch.stop)
        self.client = TestClient(main.fastapi_app)

    def get_history(self, headers):
        return self.client.get("/history", params={"thread_id": "user:tester"}, headers=headers)

    def test_키가_없으면_401(self):
        self.assertEqual(self.get_history({}).status_code, 401)

    def test_키가_틀리면_401(self):
        self.assertEqual(self.get_history({"X-API-Key": "wrong-key"}).status_code, 401)

    def test_키가_맞으면_통과(self):
        response = self.get_history({"X-API-Key": main.FASTAPI_API_KEY})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"messages": []})

    def test_질문_삭제_요청도_키가_필요하다(self):
        self.assertEqual(self.client.post("/chat", json={"message": "안녕", "thread_id": "user:tester"}).status_code, 401)
        self.assertEqual(self.client.delete("/history", params={"thread_id": "user:tester"}).status_code, 401)


if __name__ == "__main__":
    unittest.main()
