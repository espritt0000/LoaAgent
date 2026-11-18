# LoaAgent — 로스트아크 멀티 에이전트 AI 어시스턴트

LangGraph 기반 멀티 에이전트로 **경매장 시세 조회 · 캐릭터 정보 조회 · 레이드 공략 RAG 검색**을 하나의 대화창에서 처리하는 개인 프로젝트입니다.
라우터가 질문 의도를 분류해 4개의 전문 에이전트 중 하나로 보내고, 각 에이전트는 Tool Calling으로 Stove 공식 개발자 API나 벡터 DB를 조회해 답변합니다.

- 📄 설계 문서: [Loa Agent Documentation](https://checkered-harmony-550.notion.site/Loa-Agent-Documentation-41589a1f6399802aa32edcfb2cde3efb?source=copy_link)
- 🎬 데모 영상: [링크](https://drive.google.com/file/d/1omvh1UlLLzKgC0GTsXpyI37yDlNoio6c/view)

## 주요 기능

| 에이전트 | 하는 일 | 사용 Tool |
| --- | --- | --- |
| Router | 질문 의도를 4개 경로 중 하나로 분류 (Structured Output). 분류 실패 시 1회 재시도 후 General Agent로 폴백 | – |
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

Spring Boot → FastAPI 호출에는 연결 3초, 응답 60초 제한 시간을 둡니다(`application.properties`의 `fastapi.*`, 환경 변수로 변경 가능). 시간 초과(504), 연결 실패(503), FastAPI 오류 응답(502)마다 채팅 화면에 다른 안내 문구를 보여 줍니다.

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
│  ├─ router_eval.py          # 라우터 분기 정확도 테스트 (40문항)
│  ├─ test_router_fallback.py # 라우터 재시도·폴백 단위 테스트 (Mock, API 호출 없음)
│  └─ test_llm_error_messages.py # LLM 오류 종류별 안내 문구 테스트 (Mock, API 호출 없음)
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

라우터 실패 대응(재시도 + General Agent 폴백)은 LLM을 Mock으로 바꾼 단위 테스트로 확인합니다. (가상환경 활성화 후 실행)

```bash
cd ai-agent
python -m unittest test_router_fallback -v
```

## LLM 오류 처리

Gemini 오류는 LangChain 공통 예외(`langchain_core.exceptions`)로 구분해 다른 안내 문구를 보여 줍니다. 라우터는 LLM 호출 자체가 실패하면(429, 인증 오류, 서버 장애) 재시도·폴백 없이 바로 중단합니다. SDK가 이미 재시도했고, 폴백 대상인 General Agent도 같은 LLM을 쓰기 때문입니다.

| 오류 | 예외 | 안내 문구 |
| --- | --- | --- |
| 사용량 제한(429) | `ModelRateLimitError` | 요청이 많아 잠시 이용이 제한됐습니다. 1~2분 후 다시 시도해주세요. |
| 인증·권한(401/403) | `ModelAuthenticationError`, `ModelPermissionDeniedError` | AI 서비스 설정에 문제가 있습니다. 관리자에게 문의해주세요. |
| 서버 장애(5xx) 등 | `ModelAPIError`, `ModelTimeoutError`, `ModelConnectionError` | AI 서비스가 일시적으로 불안정합니다. 잠시 후 다시 시도해주세요. |
| 그 외 | – | AI 에이전트 처리 중 오류가 발생했습니다. |

컨텍스트 초과는 모든 노드에서 `trim_messages`(12,000토큰)로 입력을 자르므로 별도로 구분하지 않습니다.

```bash
cd ai-agent
python -m unittest test_llm_error_messages -v
```

## FastAPI 호출 타임아웃 테스트

가짜 FastAPI 서버(JDK 내장 `HttpServer`)로 정상 응답, 응답 지연, 서버 꺼짐, 500 응답을 만들어 상태 코드와 안내 문구를 확인합니다. Spring 컨텍스트를 띄우지 않아 MySQL 없이 실행됩니다.

```bash
cd backend/loaagent
.\gradlew test --tests "com.esprit.loaagent.chat.ChatApiControllerTimeoutTest"
```

## 회원가입 입력값 검증

회원가입 폼은 Bean Validation(`spring-boot-starter-validation`)으로 서버에서 검증하고, 오류가 있으면 저장하지 않고 입력값과 함께 필드별 오류 메시지를 보여 줍니다(비밀번호는 다시 채우지 않음).

| 항목 | 규칙 |
| --- | --- |
| 아이디 | 필수, 4~20자, 영문·숫자만 |
| 비밀번호 | 필수, 8~64자, 영문·숫자·특수문자만(공백·한글 불가) |
| 이메일 | 필수, 이메일 형식 |

비밀번호에 한글을 막는 이유: `BCryptPasswordEncoder`는 72바이트를 넘는 비밀번호에 예외를 던지는데(spring-security-crypto 7.1.2에서 직접 확인), 한글은 1자에 3바이트라 글자 수 제한만으로는 막을 수 없습니다. 1자 1바이트 문자만 허용하고 64자로 제한해 72바이트를 넘지 않게 했습니다.

```bash
cd backend/loaagent
.\gradlew test --tests "com.esprit.loaagent.member.MemberSignupValidationTest"
```

## 한계와 개선 계획

- 대화 맥락은 메모리(`MemorySaver`)에만 저장되어 서버를 재시작하면 초기화됨
- LLM 오류 안내는 FastAPI가 HTTP 200의 답변 문구로 돌려주므로, 채팅 화면에서는 오류 말풍선이 아닌 일반 답변 말풍선으로 보임
- 연결 끊김 같은 네트워크 오류가 `ModelConnectionError`로 변환되는지는 확인하지 못함(langchain-google-genai 4.4.2 소스에서는 HTTP 상태 코드 오류만 공통 예외로 변환). 변환되지 않으면 "그 외" 문구로 처리됨
- Spring 쪽 타임아웃은 기다림만 끊음 → FastAPI는 요청을 끝까지 처리하고 그 답변이 대화 기록(`MemorySaver`)에 남아, 사용자가 보지 못한 답변이 다음 질문의 맥락에 섞일 수 있음
- 응답 제한 시간 60초는 실측값이 아님. Gemini 클라이언트는 기본 `max_retries=6`, `timeout` 미설정(langchain-google-genai 4.4.2 기준)이라 API가 불안정하면 FastAPI 처리 시간이 60초를 넘을 수 있음
- 벡터 검색만 사용 → 하이브리드 검색(BM25 + 벡터)과 리랭커 도입 검토
