from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3, os, json
import numpy as np
from typing import List
from openai import OpenAI

# ----------------------------
# ✅ 기본 설정
# ----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

DB_PATH = "data/exchange_helper.db"
EMBED_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ----------------------------
# ✅ DB 초기화 (최초 실행 시 자동)
# ----------------------------
def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_id INTEGER,
        question TEXT,
        answer TEXT,
        FOREIGN KEY (category_id) REFERENCES categories (id)
    )
    """)

    # 새 테이블: 임베딩 저장용
    cur.execute("""
    CREATE TABLE IF NOT EXISTS question_embeddings (
        question_id INTEGER PRIMARY KEY,
        embedding TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()


# ----------------------------
# ✅ 기존 라우트 (카테고리 / 질문 / 답변)
# ----------------------------
@app.get("/")
def serve_home():
    return FileResponse("index.html")


@app.get("/categories")
def get_categories():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM categories")
    data = [{"id": c[0], "name": c[1]} for c in cur.fetchall()]
    conn.close()
    return {"categories": data}


@app.get("/questions/{category_id}")
def get_questions(category_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, question FROM questions WHERE category_id=?", (category_id,))
    data = [{"id": q[0], "question": q[1]} for q in cur.fetchall()]
    conn.close()
    return {"questions": data}


@app.get("/answer/{question_id}")
def get_answer(question_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT answer FROM questions WHERE id=?", (question_id,))
    ans = cur.fetchone()
    conn.close()
    return {"answer": ans[0] if ans else "답변을 찾을 수 없습니다."}


# ----------------------------
# ✅ OpenAI 임베딩 기능 추가
# ----------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# (1) 모든 질문 임베딩 생성 / 업데이트
@app.post("/embed/rebuild")
def rebuild_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, question FROM questions")
    rows = cur.fetchall()

    if not rows:
        conn.close()
        return {"status": "no_questions"}

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    vectors = embed_texts(texts)

    cur.execute("DELETE FROM question_embeddings")
    for qid, vec in zip(ids, vectors):
        cur.execute(
            "INSERT OR REPLACE INTO question_embeddings (question_id, embedding) VALUES (?, ?)",
            (qid, json.dumps(vec))
        )
    conn.commit()
    conn.close()
    return {"status": "ok", "count": len(ids)}


# (2) 의미기반 검색 (코사인 유사도)
@app.get("/search/semantic")
def semantic_search(query: str, top_k: int = 10):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT q.id, q.question, q.answer, e.embedding
        FROM questions q
        JOIN question_embeddings e ON e.question_id = q.id
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="질문 임베딩이 없습니다. /embed/rebuild 먼저 실행하세요.")

    q_vec = np.array(embed_texts([query])[0], dtype=np.float32)
    results = []

    for qid, qtext, ans, emb_json in rows:
        v = np.array(json.loads(emb_json), dtype=np.float32)
        score = cosine_sim(q_vec, v)
        results.append({"id": qid, "question": qtext, "answer": ans, "score": round(score, 4)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"query": query, "results": results[:top_k]}
