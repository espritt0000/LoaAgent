# LoaAgent — 로스트아크 멀티 에이전트 AI 어시스턴트

LangGraph 기반 멀티 에이전트로 **경매장 시세 조회 · 캐릭터 정보 조회 · 레이드 공략 RAG 검색**을 하나의 대화창에서 처리하는 개인 프로젝트입니다.
라우터가 질문 의도를 분류해 4개의 전문 에이전트 중 하나로 보내고, 각 에이전트는 Tool Calling으로 Stove 공식 개발자 API나 벡터 DB를 조회해 답변합니다.

- 📄 설계 문서: [Loa Agent Documentation](https://checkered-harmony-550.notion.site/Loa-Agent-Documentation-41589a1f6399802aa32edcfb2cde3efb?source=copy_link)
- 🎬 데모 영상: [링크](https://drive.google.com/file/d/1omvh1UlLLzKgC0GTsXpyI37yDlNoio6c/view)

## 주요 기능

| 에이전트 | 하는 일 | 사용 Tool |
| --- | --- | --- |
| Router | 질문 의도를 4개 경로 중 하나로 분류 (Structured Output) | – |
| Market Agent | 경매장 아이템의 전일 평균가·최근 거래가 조회 | `get_market_options` → `search_market_items` |
| Character Agent | 캐릭터 기본 프로필(서버, 직업, 레벨, 전투력, 길드) 조회 | `get_character_profile` |
| Guide Agent | 레이드 공략 문서 RAG 검색, 정보가 부족하면 재검색 | `search_game_guide` |
| General Agent | 인사·잡담 등 일반 대화 | – |

## 아키텍처

```
브라우저 ──▶ Spring Boot (로그인·세션, MySQL) ──▶ FastAPI /chat ──▶ LangGraph
                                                              ├─ Router
                                                              ├─ Market / Character Agent ──▶ Stove API
                                                              ├─ Guide Agent ──▶ Chroma (BGE-M3)
                                                              └─ General Agent
                                                     (모든 에이전트의 LLM: Gemini)
```

## 기술 스택

- **AI 서버**: Python, FastAPI, LangGraph, LangChain, Gemini (`gemini-3.5-flash-lite`)
- **RAG**: Chroma, BGE-M3 (Ollama 로컬 실행), MarkdownHeaderTextSplitter
- **백엔드**: Java 21, Spring Boot, Spring Security, Spring Data JPA, Spring Session JDBC, Thymeleaf
- **DB**: MySQL
- **외부 API**: Stove 로스트아크 공식 개발자 API

## 폴더 구조

```
LoaAgent/
├─ ai-agent/                  # AI 서버 (FastAPI + LangGraph)
│  ├─ main.py                 # FastAPI /chat 엔드포인트
│  ├─ agent/agent.py          # 라우터·에이전트 노드, 그래프 구성
│  ├─ tools/                  # market.py, character.py, guide.py
│  ├─ utils/
│  │  ├─ DBupdate.py          # 공략 문서 증분 임베딩 (변경된 파일만 재적재)
│  │  └─ utils.py             # Stove API 공통 헤더, 에러 핸들러
│  ├─ data/raids/             # 레이드 공략 Markdown 문서
│  └─ router_eval.py          # 라우터 분기 정확도 테스트 (40문항)
└─ backend/loaagent/          # Spring Boot (인증·세션·화면)
```

## 실행 방법

### 사전 준비
- Python 3.10 이상, JDK 21, MySQL, [Ollama](https://ollama.com)
- API 키: Gemini API Key, Stove 로스트아크 개발자 API Key

### 1. AI 서버 (ai-agent)

```bash
cd ai-agent
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

ollama pull bge-m3                # 임베딩 모델
copy .env.example .env            # .env를 열어 API 키 입력 (macOS/Linux: cp)

python utils/DBupdate.py          # 공략 문서를 Chroma에 적재
uvicorn main:fastapi_app --port 8000
```

### 2. 백엔드 (backend/loaagent)

1. MySQL에 `loaagent_db` 데이터베이스 생성 (테이블은 실행 시 자동 생성)
2. `src/main/resources/application-secret.properties.example`을 `application-secret.properties`로 복사해 비밀번호 입력
3. HTTPS 설정
   - HTTPS로 실행: `keytool -genkeypair -alias springboot -keyalg RSA -storetype PKCS12 -keystore src/main/resources/keystore.p12 -validity 365`로 인증서 생성
   - HTTP로 실행: 환경 변수 `SSL_ENABLED=false` 설정
4. 실행

```bash
cd backend/loaagent
.\gradlew bootRun                 # macOS/Linux: ./gradlew bootRun
```

5. 브라우저에서 `https://localhost:8080` (HTTP 실행 시 `http://`) 접속 → 회원가입 → 로그인 → 채팅 화면

## 라우터 분기 테스트

```bash
cd ai-agent
python router_eval.py
```

에이전트별 10문항, 총 40문항으로 라우터의 분류 결과를 기대 경로와 비교합니다. (2026-11-17 측정: 40/40, 단일 턴 질문 기준)

## 한계와 개선 계획

- 대화 맥락은 메모리(`MemorySaver`)에만 저장되어 서버를 재시작하면 초기화됨
- 라우터 구조화 출력 파싱 실패 시 재시도 미구현
- 벡터 검색만 사용 → 하이브리드 검색(BM25 + 벡터)과 리랭커 도입 검토
