from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os
from openai import OpenAI
import numpy as np

# ===============================
#  초기 설정
# ===============================
DB_PATH = "data/exchange_helper.db"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
#  HTML (index.html) 라우트
# ===============================
@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

# ===============================
#  DB 관련 함수
# ===============================
def get_all_questions():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, question, answer FROM questions")
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "question": r[1], "answer": r[2]} for r in rows]

# ===============================
#  기존 API (그대로 유지)
# ===============================
@app.get("/categories")
def get_categories():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM categories")
    data = [{"id": row[0], "name": row[1]} for row in cur.fetchall()]
    conn.close()
    return {"categories": data}

@app.get("/questions/{category_id}")
def get_questions(category_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, question FROM questions WHERE category_id=?", (category_id,))
    data = [{"id": row[0], "question": row[1]} for row in cur.fetchall()]
    conn.close()
    return {"questions": data}

@app.get("/answer/{question_id}")
def get_answer(question_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT answer FROM questions WHERE id=?", (question_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="해당 질문을 찾을 수 없습니다.")
    return {"answer": row[0]}

# ===============================
#  🔍 코사인 유사도 기반 검색
# ===============================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.get("/search")
def semantic_search(query: str):
    """OpenAI 임베딩 기반 문맥 검색"""
    try:
        # 1️⃣ 모든 질문 불러오기
        questions = get_all_questions()
        if not questions:
            return {"results": []}

        # 2️⃣ 검색어 임베딩 생성
        query_embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        # 3️⃣ 각 질문 문장 임베딩 생성 + 코사인 유사도 계산
        scored = []
        for q in questions:
            q_embed = client.embeddings.create(
                model="text-embedding-3-small",
                input=q["question"]
            ).data[0].embedding

            similarity = cosine_similarity(
                np.array(query_embed), np.array(q_embed)
            )
            scored.append((q, similarity))

        # 4️⃣ 유사도 순 정렬 + 임계값 필터링 (0.3 이상만)
        threshold = 0.3
        results = [
            item[0] for item in sorted(scored, key=lambda x: x[1], reverse=True)
            if item[1] >= threshold
        ]

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===============================
#  실행 (로컬 테스트용)
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
