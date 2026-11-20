"""
공략 문서 증분 동기화(utils/DBupdate.py) 테스트

실행 위치: ai-agent 폴더
실행 방법: python -m unittest test_dbupdate -v
- Chroma와 Ollama 임베딩을 Mock으로 바꾸고 임시 폴더를 쓴다. 실제 벡터 DB는 건드리지 않는다.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils import DBupdate

DOC = "---\ntitle: 테스트 레이드\ncategory: raid\n---\n## 개요\n내용\n## 1관문: 보스\n패턴\n"


class DBUpdateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raid_dir = Path(self.tmp.name) / "raids"
        self.raid_dir.mkdir()
        (self.raid_dir / "test.md").write_text(DOC, encoding="utf-8")
        self.state_path = Path(self.tmp.name) / "sync_state.json"
        self.store = MagicMock()
        for name, value in (("RAID_DATA_DIR", str(self.raid_dir)), ("STATE_FILE_PATH", str(self.state_path)),
                            ("Chroma", MagicMock(return_value=self.store)), ("OllamaEmbeddings", MagicMock())):
            p = patch.object(DBupdate, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_변경된_문서는_기존_청크를_지우고_다시_넣고_상태를_기록한다(self):
        DBupdate.build_vector_db_incremental()
        self.store.delete.assert_called_once_with(where={"source_file": "test.md"})
        self.assertEqual(len(self.store.add_documents.call_args.kwargs["documents"]), 2)
        self.assertIn("test.md", json.loads(self.state_path.read_text(encoding="utf-8")))

    def test_기존_청크_삭제가_실패하면_멈추고_완료로_기록하지_않는다(self):
        self.store.delete.side_effect = RuntimeError("chroma locked")
        with self.assertRaises(RuntimeError):
            DBupdate.build_vector_db_incremental()
        self.store.add_documents.assert_not_called()
        self.assertFalse(self.state_path.exists())  # 다음 실행 때 다시 시도된다

    def test_폴더에서_지운_문서의_청크_삭제가_실패하면_상태에서_빼지_않는다(self):
        self.state_path.write_text(json.dumps({"gone.md": 1.0}), encoding="utf-8")
        self.store.delete.side_effect = RuntimeError("chroma locked")
        with self.assertRaises(RuntimeError):
            DBupdate.build_vector_db_incremental()
        self.assertIn("gone.md", json.loads(self.state_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
