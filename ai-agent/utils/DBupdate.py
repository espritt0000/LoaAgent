import os
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "loa_guides")
CHROMA_PERSIST_DIR = str(BASE_DIR / os.getenv("CHROMA_PERSIST_DIR"))
RAID_DATA_DIR = str(BASE_DIR / os.getenv("RAID_DATA_DIR"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# 파일별 동기화 상태를 저장할 JSON 캐시 파일 경로
STATE_FILE_PATH = str(BASE_DIR / "data" / "sync_state.json")

def load_sync_state():
    if os.path.exists(STATE_FILE_PATH):
        with open(STATE_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_sync_state(state):
    with open(STATE_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def build_vector_db_incremental():
    if not os.path.exists(RAID_DATA_DIR):
        raise FileNotFoundError(f"공략 폴더를 찾을 수 없습니다: {RAID_DATA_DIR}")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

    sync_state = load_sync_state()
    md_files = [f for f in os.listdir(RAID_DATA_DIR) if f.endswith(".md")]
    
    if not md_files:
        print("⚠️ 적재할 마크다운 파일이 없습니다.")
        return vector_store

    current_files = set(md_files)
    tracked_files = set(sync_state.keys())

    # 1. raids 폴더에서 아예 삭제된 파일은 DB에서도 청크 제거
    # 삭제가 실패하면 예외로 멈춘다 (숨기면 청크는 남는데 상태 파일에서는 지워져 다시 시도되지 않음). 지울 청크가 없을 때는 Chroma가 오류를 내지 않는다
    deleted_files = tracked_files - current_files
    for file_name in deleted_files:
        vector_store.delete(where={"source_file": file_name})
        print(f"🗑️ [삭제 반영] '{file_name}' 관련 벡터 데이터가 DB에서 제거되었습니다.")
        if file_name in sync_state:
            del sync_state[file_name]

    updated_count = 0

    for file_name in md_files:
        file_path = os.path.join(RAID_DATA_DIR, file_name)
        file_mtime = os.path.getmtime(file_path)

        # 2. 파일 수정 시간(mtime)이 이전과 동일하다면 변경 없는 파일이므로 스킵!
        if file_name in sync_state and sync_state[file_name] == file_mtime:
            continue

        print(f"🔄 변경 감지됨: [{file_name}] - 벡터 재적재 진행...")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Frontmatter 분리
        frontmatter = {}
        markdown_text = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1]) or {}
                markdown_text = parts[2].strip()

        # 중제목 기준 분할
        headers_to_split_on = [("##", "Header 2")]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        raw_chunks = splitter.split_text(markdown_text)

        # 기존 청크 삭제 후 새로 추가 (Upsert). 실패하면 멈춘다: 섹션 수가 줄었을 때 옛 청크가 남은 채 '동기화 완료'로 기록되지 않도록
        vector_store.delete(where={"source_file": file_name})

        docs = []
        ids = []
        
        doc_title = frontmatter.get("title", "")
        category = frontmatter.get("category", "")

        for i, chunk in enumerate(raw_chunks):
            header_2 = chunk.metadata.get("Header 2", "")

            # 💡 [핵심] 본문 맨 위에 문서 정보 및 섹션 헤더 자동 주입 (Prepend)
            context_prefix = f"[문서 정보]\n- title: {doc_title}\n- category: {category}\n- section: {header_2}\n\n"
            combined_content = context_prefix + chunk.page_content

            # 메타데이터 병합
            combined_metadata = {
                **frontmatter,
                **chunk.metadata,
                "source_file": file_name
            }

            doc = Document(page_content=combined_content, metadata=combined_metadata)
            docs.append(doc)
            
            chunk_id = f"{file_name}_chunk_{i}"
            ids.append(chunk_id)

        if docs:
            vector_store.add_documents(documents=docs, ids=ids)
        
        # 3. 최신 수정 시간으로 상태 갱신
        sync_state[file_name] = file_mtime
        updated_count += 1
        print(f"✨ [{file_name}] 동기화 완료 (반영된 청크 수: {len(docs)}개)")

    # 상태 파일 저장
    save_sync_state(sync_state)

    if updated_count == 0:
        print("💤 변경된 공략 파일이 없어 동기화를 스킵했습니다. (모든 데이터 최신 상태 유지 중)")
    else:
        print(f"✅ 총 {updated_count}개 파일의 스마트 증분 동기화가 완료되었습니다!")

    return vector_store
if __name__ == "__main__":
    print("--- 🚀 스마트 증분 벡터 DB 동기화 시작 ---")
    build_vector_db_incremental()