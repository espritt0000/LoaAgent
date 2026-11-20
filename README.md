# LoaAgent — 로스트아크 멀티 에이전트 AI 어시스턴트

LangGraph 기반 멀티 에이전트로 **경매장 시세 조회 · 캐릭터 정보 조회 · 레이드 공략 RAG 검색**을 하나의 대화창에서 처리하는 개인 프로젝트입니다.
라우터가 질문 의도를 분류해 4개의 전문 에이전트 중 하나로 보내고, 각 에이전트는 Tool Calling으로 Stove 공식 개발자 API나 벡터 DB를 조회해 답변합니다.

- 📄 설계 문서: [Loa Agent Documentation](https://checkered-harmony-550.notion.site/Loa-Agent-Documentation-41589a1f6399802aa32edcfb2cde3efb?source=copy_link) — 설계 이유, 측정 방법·조건·해석, 한계는 이 문서에 정리했습니다. README에는 요약과 실행 방법만 둡니다.
- 🎬 데모 영상: [링크](https://drive.google.com/file/d/1omvh1UlLLzKgC0GTsXpyI37yDlNoio6c/view)

## 주요 기능

| 에이전트 | 하는 일 | 사용 Tool |
| --- | --- | --- |
| Router | 질문을 1~2단계 작업 계획(에이전트 + 할 일)으로 나눔 (Structured Output). 대부분 1단계, 두 영역이 필요하면 2단계로 순서대로 실행한 뒤 synthesizer가 합침. 분류 실패 시 1회 재시도 후 General Agent로 폴백 | – |
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
                                                              ├─ General Agent
                                                              └─ Synthesizer (2단계 계획일 때 결과 합치기)
                                                     (모든 에이전트의 LLM: Gemini)
```

### 설계 요약

| 항목 | 내용 | 설계 문서 |
| --- | --- | --- |
| 작업 계획(Planner) | 라우터가 1~2단계 계획을 세우고 에이전트를 순서대로 실행. 2단계면 툴 없는 합치기 노드(synthesizer)가 단계별 결과를 자료로 받아 최종 답변 | 5.2 |
| 라우터 실패 대응 | 분류 실패 시 1회 재시도 → General Agent 폴백. LLM 호출 자체가 실패하면(429·인증·서버 장애) 재시도 없이 중단 | 5.2 |
| LLM 오류 안내 | LangChain 공통 예외로 429(분당/하루 한도 구분)·인증·서버 장애를 나눠 안내, 실패한 턴은 대화 기록에서 제거 | 4.2 |
| 대화 기록 | `SqliteSaver`로 영구 저장, `thread_id`는 Spring이 로그인 계정으로 결정, 새로고침 시 복원, '새 대화' 버튼 | 4.1, 4.2 |
| RAG | `##` 단위 청킹 + 증분 동기화, 동적 k(기본 2, 전체·요약 3), 문서에 없는 레이드·보스는 검색하지 않고 '문서 없음', 관문 번호가 있으면 그 관문 청크 + 개요만 검색(없는 관문은 안내) | 5.5 |
| 컨텍스트 관리 | 모든 노드에서 `trim_messages`(12,000토큰), `recursion_limit=15` | 4.2 |
| 서비스 간 호출 | Spring → FastAPI 연결 3초·응답 60초 제한, 원인별 504/503/502 안내 | 4.1 |
| 보안 | Spring CSRF 보호, FastAPI 모든 요청에 `X-API-Key` 검증 | 4.1 |
| 회원가입 검증 | Bean Validation, 한글 비밀번호 차단(BCrypt 72바이트 제한) | 4.1 |

## 기술 스택

- **AI 서버**: Python, FastAPI, LangGraph, LangChain, Gemini (`gemini-3.5-flash-lite`, `.env`의 `LLM_PROVIDER`·`LLM_MODEL`로 변경 가능)
- **RAG**: Chroma, BGE-M3 (Ollama 로컬 실행), MarkdownHeaderTextSplitter
- **백엔드**: Java 21, Spring Boot, Spring Security, Spring Data JPA, Spring Session JDBC, Thymeleaf
- **DB**: MySQL (회원·세션), SQLite (대화 기록)
- **외부 API**: Stove 로스트아크 공식 개발자 API

## 폴더 구조

```
LoaAgent/
├─ ai-agent/                  # AI 서버 (FastAPI + LangGraph)
│  ├─ main.py                 # FastAPI /chat, /history 엔드포인트
│  ├─ agent/agent.py          # 라우터·에이전트 노드, 그래프 구성
│  ├─ tools/                  # market.py, character.py, guide.py
│  ├─ utils/
│  │  ├─ DBupdate.py          # 공략 문서 증분 임베딩 (변경된 파일만 재적재)
│  │  └─ utils.py             # Stove API 공통 헤더, 에러 핸들러
│  ├─ data/raids/             # 레이드 공략 Markdown 문서 6개 (세르카, 벨가르딘, 칠흑 폭풍의 밤, 파멸의 성채, 최후의 날, 지평의 성당)
│  ├─ router_eval.py          # 라우터 분기 테스트 (단일 턴 40 / multiturn 28 / hard 24 / holdout 10 / compound 12)
│  ├─ compound_eval.py        # 두 영역이 섞인 질문 실제 답변 확인 (그래프 전체 실행, 6문항)
│  ├─ rag_eval.py             # RAG 검색 평가 (검색어 55개 + 관문 10개 + 문서 없는 검색어 10개 + 조건 불일치 4개)
│  ├─ test_router_fallback.py # 라우터 재시도·폴백 단위 테스트 (Mock, API 호출 없음)
│  ├─ test_planner.py         # 작업 계획(Planner) 실행 순서·단계 지시 단위 테스트 (Mock, API 호출 없음)
│  ├─ test_llm_error_messages.py # LLM 오류 종류별 안내 문구 테스트 (Mock, API 호출 없음)
│  ├─ test_chat_history.py    # 대화 기록 영구 저장·/history 테스트 (Mock + 임시 SQLite)
│  ├─ test_api_key.py         # FastAPI 내부 호출용 API Key 테스트 (TestClient)
│  ├─ test_character_name.py  # 캐릭터명 조사 처리 테스트 (Mock, API 호출 없음)
│  └─ test_gate_filter.py     # 공략 검색 관문 필터 테스트 (Mock, 임베딩 호출 없음)
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
copy .env.example .env            # .env를 열어 값 입력 (macOS/Linux: cp)

python utils/DBupdate.py          # 공략 문서를 Chroma에 적재
uvicorn main:fastapi_app --port 8000
```

`.env`의 주요 설정값 (아래 값이 없으면 서버가 시작 단계에서 멈춥니다):

| 설정 | 내용 |
| --- | --- |
| `LLM_PROVIDER`, `LLM_MODEL` | LLM 공급자와 모델 (서비스: `google_genai`, `gemini-3.5-flash-lite`). 코드에 기본값이 없습니다. 다른 공급자는 해당 패키지(예: `langchain-anthropic`)와 API 키가 필요하며 아직 실행해 보지 않았습니다 |
| `CHECKPOINT_DB_PATH` | 대화 기록 SQLite 파일 경로 (예: `data/chat_history.sqlite`, ai-agent 기준). 처음 실행 시 자동 생성, git에는 올라가지 않음 |
| `FASTAPI_API_KEY` | 내부 호출용 키. 백엔드 `fastapi.api-key`와 같은 값이어야 하며, 다르면 채팅 요청이 실패합니다(FastAPI 401 → 화면에는 502 안내). 만들기: `python -c "import secrets; print(secrets.token_urlsafe(32))"` |

### 2. 백엔드 (backend/loaagent)

1. MySQL에 `loaagent_db` 데이터베이스 생성 (테이블은 실행 시 자동 생성)
2. `src/main/resources/application-secret.properties.example`을 `application-secret.properties`로 복사해 비밀번호와 내부 호출용 API Key(AI 서버 `.env`의 `FASTAPI_API_KEY`와 같은 값) 입력
3. HTTPS 설정
   - HTTPS로 실행: `keytool -genkeypair -alias springboot -keyalg RSA -storetype PKCS12 -keystore src/main/resources/keystore.p12 -validity 365`로 인증서 생성
   - HTTP로 실행: 환경 변수 `SSL_ENABLED=false` 설정
4. 실행

```bash
cd backend/loaagent
.\gradlew bootRun                 # macOS/Linux: ./gradlew bootRun
```

5. 브라우저에서 `https://localhost:8080` (HTTP 실행 시 `http://`) 접속 → 회원가입 → 로그인 → 채팅 화면

FastAPI 호출 제한 시간은 `application.properties`의 `fastapi.*`(연결 3초, 응답 60초)이며 환경 변수로 바꿀 수 있습니다.

## 검증 결과 요약

모두 2026-11-19 전후 직접 측정한 값입니다. 측정 조건(모델, 횟수)과 해석, 틀린 사례는 설계 문서 7장에 있습니다.

| 측정 | 결과 | 실행 | 설계 문서 |
| --- | --- | --- | --- |
| 라우터 단일 턴 분기 (40문항) | 40/40 (gemini-3.5·3.1-flash-lite 각 1회) | `python router_eval.py` | 7 |
| 멀티턴 후속 질문 (28건) | 앞 대화와 함께 28/28, 후속 질문만 24/28 | `python router_eval.py multiturn` | 7.2 |
| 단서 없는 후속 질문 (24건, 3.5) | 라우터 프롬프트에 이전 대화 활용 규칙 추가 후 23/23 × 2회 (규칙 추가 전 같은 19건 15·17·16, 앞 대화 없는 지시어 4건은 계속 일반 대화로) | `python router_eval.py hard --ctx-only` | 7.3 |
| 처음 보는 질문 (10건, 3.5) | 규칙을 고칠 때 보지 않은 질문으로 과적합 확인: 10/10 × 3회 | `python router_eval.py holdout` | 7.3 |
| RAG 검색 (검색어 55개, 청크 19개) | 동적 k 적중 47/55, Hit@1 76%, MRR 0.835 | `python rag_eval.py` | 7.4 |
| k=1 규칙 삭제 (검색어 26개, 청크 6개) | 동적 k 적중 21/26 → 24/26 | `python rag_eval.py` | 7.4 |
| 검색어 레이드 이름 유지 규칙 (8문항 × 3회, 3.1) | 이름 유지 9/24 → 17/24 | – | 7.4 |
| 문서에 없는 레이드·보스 거절 | 거절 10/10, 정상 검색어 통과 46/55 | `python rag_eval.py` | 7.4 |
| 관문 필터 (관문 검색어 10개 + 조건 불일치 4개) | 관문 청크 포함 4/10 → 10/10, 없는 관문·보스 불일치 안내 0/4 → 4/4, 기존 검색어 회귀 없음 | `python rag_eval.py` | 7.4 |
| 두 영역이 섞인 질문 - 라우터 계획 (12건, 3.1) | 섞인 질문 9건: 변경 전 모두 둘 중 하나만 → 변경 후 9/9 두 에이전트, 함정 3/3 | `python router_eval.py compound` | 7.5 |
| 두 영역이 섞인 질문 - 실제 답변 (5문항 × 3회, 3.1) | 두 영역 모두 답함: 변경 전 0/5 → 변경 후 11/15 (실패 4건 중 2건은 검색 문제, 1건은 Gemini 503) | `python compound_eval.py` | 7.5 |
| 합치기 노드(synthesizer) (6문항 × 3회, 3.1) | 모든 영역 확인 18/18 → 17/18 (실패 1건은 라우터의 캐릭터명 오해석), 앞 단계 누락 0 → 0, 답변에 "1단계에서" 같은 내부 표현 1 → 0. 실행당 LLM 호출 31 → 36. 라우터 회귀(3.5) 단일 40/40, 섞인 질문 12/12 | `python compound_eval.py` | 7.5 |

`router_eval.py`는 서비스와 같은 `LLM_MODEL` 설정을 쓰므로 실행할 때 값만 바꾸면 모델별로 비교할 수 있습니다(Git Bash 예: `LLM_MODEL=gemini-3.1-flash-lite python router_eval.py`). 다른 공급자 모델과의 비교는 아직 하지 않았습니다.

### 단위 테스트

외부 API를 부르지 않고 Mock·임시 SQLite·가짜 FastAPI로 실패 상황을 재현합니다. (가상환경 활성화 후 실행)

```bash
cd ai-agent
python -m unittest test_router_fallback test_planner test_llm_error_messages test_chat_history test_api_key test_character_name test_gate_filter -v
```

```bash
cd backend/loaagent
.\gradlew test --tests "com.esprit.loaagent.chat.*" --tests "com.esprit.loaagent.member.MemberSignupValidationTest"
```

| 테스트 | 확인 내용 |
| --- | --- |
| `test_router_fallback` | 라우터 재시도·폴백, LLM 호출 오류 시 중단 |
| `test_planner` | 2단계 계획을 순서대로 실행 후 synthesizer 1회(이번 턴 결과만 자료로), 1단계는 synthesizer 없이 기존과 동일, 최대 2단계(넘으면 안내), 빈 계획 폴백 |
| `test_llm_error_messages` | 429·인증·서버 오류·기타 예외별 안내 문구 |
| `test_chat_history` | 재연결 후 대화 유지, 계정별 저장, `/history` 최근 50개, 새 대화(삭제), 같은 계정 동시 질문 시 두 턴 모두 저장 |
| `test_api_key` | 키 없음·틀린 키 401, 맞는 키 통과 |
| `test_gate_filter` | 관문 번호 표현(2관문·이관문·두 번째 관문), 관문 청크 + 개요, 없는 관문·보스와 관문 불일치 안내, 레이드 제목 우선 |
| `test_character_name` | 조사가 붙은 캐릭터명은 조사를 떼고 다시 조회, "하늘로"처럼 조사로 끝나는 진짜 닉네임은 그대로, 둘 다 없으면 시도한 이름 안내, Stove 호출에 timeout 지정 |
| `ChatApiControllerTimeoutTest` | 정상 200, 응답 지연 504, 서버 꺼짐 503, 500 응답 502 |
| `ChatHistoryApiTest` | 브라우저가 보낸 `thread_id` 대신 로그인 계정 값 사용, 조회·삭제 실패 시 502 |
| `CsrfProtectionTest` | 토큰 없는 요청 403, `<meta>` 토큰 헤더로 통과 |
| `MemberSignupValidationTest` | 빈 값·형식·길이 위반, 한글 비밀번호 거부 |

## 한계와 개선 계획

주요 항목만 적었습니다. 전체 목록과 원인은 설계 문서 8장에 있습니다.

- 작업 계획은 최대 2단계라 세 영역이 모두 필요한 질문은 일부만 답하고 나머지는 다시 물어보도록 안내함. 계획은 처음에 한 번 세우고 중간에 바꾸지 않아, 1단계가 실패하면 2단계에서 바로잡지 못함 (캐릭터명에 조사가 붙어 조회에 실패하던 경우는 캐릭터 툴이 조사를 떼고 다시 조회하도록 보완)
- 관문 필터는 LLM이 만든 검색어를 읽으므로, LLM이 검색어에서 레이드 이름을 바꾸거나 관문 번호를 빼면 막지 못함. 관문 번호 외의 조건(난이도 등)은 확인하지 않음
- Guide Agent가 검색어의 레이드 이름을 바꾸는 문제는 프롬프트 규칙으로 줄였지만 남아 있음
- 벡터 검색만 사용 (하이브리드 검색·리랭커 검토)
- 대화는 계정당 하나, 대화 기록은 AI 서버 PC의 SQLite 파일에 있어 서버를 여러 대로 늘리면 공유되지 않음. 같은 계정의 동시 질문은 프로세스 안의 lock으로 순서대로 처리하므로 worker·서버를 여러 개로 늘리면 DB 쪽 잠금이 필요
- Spring 타임아웃은 기다림만 끊고 FastAPI 처리를 취소하지 않음, 응답 제한 60초는 실측값이 아님
- 내부 호출용 API Key는 교체 기능이 없고, Spring → FastAPI 통신은 같은 PC 안이라는 전제로 암호화하지 않음
