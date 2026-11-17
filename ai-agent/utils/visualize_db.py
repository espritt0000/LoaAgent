import os
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 프로젝트 루트 경로 기준 환경변수 로드
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CHROMA_PERSIST_DIR = str(BASE_DIR / os.getenv("CHROMA_PERSIST_DIR"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def visualize_vector_db():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

    # Chroma 컬렉션에서 ID, 임베딩, 메타데이터, 본문 전체 가져오기
    collection = vector_store._collection
    data = collection.get(include=["embeddings", "metadatas", "documents"])

    ids = data.get("ids", [])
    embeddings_list = data.get("embeddings", [])
    metadatas = data.get("metadatas", [])
    documents = data.get("documents", [])

    if not ids or len(embeddings_list) == 0:
        print("⚠️ 시각화할 벡터 데이터가 없습니다.")
        return

    # PCA를 이용해 고차원 임베딩을 2차원으로 축소
    X = np.array(embeddings_list)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # DataFrame 구성 (전체 본문 full_content 포함)
    df_data = []
    for i in range(len(ids)):
        meta = metadatas[i] if metadatas[i] else {}
        df_data.append({
            "idx": i,
            "id": ids[i],
            "source_file": meta.get("source_file", "Unknown"),
            "boss": meta.get("boss", "Unknown"),
            "header": meta.get("Header 2", "General"),
            "full_content": documents[i],  # 👈 전체 청크 본문 저장
            "x": X_2d[i, 0],
            "y": X_2d[i, 1]
        })

    df = pd.DataFrame(df_data)

    # Plotly 산점도 생성 (custom_data에 상세 정보 전달)
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="header",
        custom_data=["idx", "id", "source_file", "boss", "header"],
        hover_data=["id", "source_file"],
        title=f"LoaAgent RAG Vector Space Explorer ({COLLECTION_NAME})"
    )

    # 시각적 가독성 개선
    fig.update_traces(marker=dict(size=14, line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(
        legend_title="Section Header",
        hoverlabel=dict(bgcolor="white", font_size=12),
        clickmode='event+select',
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # Plotly 그래프를 HTML 조각으로 변환
    plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    
    # JavaScript에서 참조할 수 있도록 DataFrame을 JSON으로 변환
    df_json = df.to_json(orient='records', force_ascii=False)

    # 좌우 분할 레이아웃 및 클릭 이벤트 핸들러가 포함된 커스텀 HTML 템플릿 작성
    custom_html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <title>LoaAgent Vector DB Explorer</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-gray-100 h-screen flex flex-col overflow-hidden font-sans">
        <!-- 상단 헤더 -->
        <header class="bg-gray-800 border-b border-gray-700 px-6 py-4 flex justify-between items-center shadow-md">
            <h1 class="text-xl font-bold text-cyan-400">🛡️ LoaAgent RAG Vector Space Explorer</h1>
            <span class="text-sm text-gray-400">총 청크 수: <strong>{len(ids)}개</strong> | 그래프의 점을 클릭하면 우측 패널에서 전체 본문 내용을 확인할 수 있습니다.</span>
        </header>
        
        <!-- 메인 콘텐츠 영역 (좌측: 그래프, 우측: 상세 정보 패널) -->
        <div class="flex flex-1 overflow-hidden">
            <!-- 그래프 컨테이너 -->
            <div class="flex-1 h-full p-4 flex flex-col">
                <div id="plotly-container" class="w-full h-full bg-gray-800 rounded-xl shadow-lg border border-gray-700 p-2 flex items-center justify-center overflow-hidden">
                    {plot_html}
                </div>
            </div>

            <!-- 우측 상세 정보 패널 -->
            <div class="w-[480px] bg-gray-800 border-l border-gray-700 flex flex-col h-full shadow-2xl">
                <div class="p-4 border-b border-gray-700 bg-gray-850 flex justify-between items-center">
                    <h2 class="font-semibold text-lg text-emerald-400">📄 청크 상세 정보</h2>
                    <span id="selected-id" class="text-xs bg-gray-700 text-cyan-300 px-2.5 py-1 rounded font-mono">선택된 청크 없음</span>
                </div>
                
                <div class="p-5 flex-1 overflow-y-auto space-y-4 text-sm leading-relaxed text-gray-300">
                    <div id="welcome-message" class="text-center text-gray-500 mt-32">
                        <p class="text-base mb-2 font-medium text-gray-400">👈 그래프에서 점을 클릭해보세요.</p>
                        <p class="text-xs">해당 벡터와 매핑된 고유 ID, 출처 파일, 소속 헤더 및 전체 마크다운 본문이 출력됩니다.</p>
                    </div>

                    <div id="details-container" class="hidden space-y-4">
                        <div class="bg-gray-900 p-3 rounded-lg border border-gray-700 space-y-2">
                            <div class="flex justify-between text-xs text-gray-400">
                                <span>📁 출처 파일: <strong id="info-file" class="text-gray-200"></strong></span>
                                <span>🎯 보스: <strong id="info-boss" class="text-gray-200"></strong></span>
                            </div>
                            <div class="text-xs text-gray-400">
                                <span>📌 소속 헤더: <strong id="info-header" class="text-cyan-300"></strong></span>
                            </div>
                        </div>

                        <div>
                            <h3 class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">Full Markdown Content</h3>
                            <div id="info-content" class="bg-gray-950 p-4 rounded-lg border border-gray-800 font-mono text-xs whitespace-pre-wrap text-gray-200 leading-relaxed select-all"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 클릭 상호작용을 처리하는 JavaScript -->
        <script>
            const chunkData = {df_json};
            
            // Plotly 그래프 엘리먼트 탐색
            const gd = document.getElementById('plotly-container').getElementsByClassName('plotly-graph-div')[0];
            
            if (gd) {{
                gd.on('plotly_click', function(data) {{
                    if (data.points && data.points.length > 0) {{
                        // customdata에서 index 가져오기
                        const pointIndex = data.points[0].customdata[0];
                        const item = chunkData[pointIndex];
                        
                        if (item) {{
                            // UI 패널에 데이터 주입
                            document.getElementById('welcome-message').classList.add('hidden');
                            document.getElementById('details-container').classList.remove('hidden');
                            
                            document.getElementById('selected-id').innerText = item.id;
                            document.getElementById('info-file').innerText = item.source_file;
                            document.getElementById('info-boss').innerText = item.boss;
                            document.getElementById('info-header').innerText = item.header;
                            document.getElementById('info-content').innerText = item.full_content;
                        }}
                    }}
                }});
            }}
        </script>
    </body>
    </html>
    """

    # HTML 파일로 저장 및 브라우저 열기
    output_html = str(BASE_DIR / "data" / "vector_visualization.html")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(custom_html)
        
    print(f"✨ 인터랙티브 상세 뷰어 HTML 생성 완료: {output_html}")
    
    import webbrowser
    webbrowser.open(output_html)

if __name__ == "__main__":
    print("--- 📊 벡터 DB 2D 인터랙티브 시각화 시작 ---")
    visualize_vector_db()