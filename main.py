from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os
from openai import OpenAI
import numpy as np
from pydantic import BaseModel
from typing import List, Dict

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
        questions = get_all_questions()
        if not questions:
            return {"results": []}

        query_embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        scored = []
        for q in questions:
            q_embed = client.embeddings.create(
                model="text-embedding-3-small",
                input=q["question"]
            ).data[0].embedding
            similarity = cosine_similarity(np.array(query_embed), np.array(q_embed))
            scored.append((q, similarity))

        threshold = 0.3
        results = [
            item[0] for item in sorted(scored, key=lambda x: x[1], reverse=True)
            if item[1] >= threshold
        ]
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===============================
#  🤖 AI 챗봇 기능
# ===============================
def embed_text(text: str) -> List[float]:
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def top_k_qa_context(query: str, k: int = 3) -> List[Dict[str, str]]:
    """로컬 DB의 Q/A 중 질의와 가장 가까운 k개 반환"""
    questions = get_all_questions()
    if not questions:
        return []
    q_embed = np.array(embed_text(query))
    scored = []
    for item in questions:
        e = np.array(embed_text(item["question"]))
        sim = cosine_similarity(q_embed, e)
        scored.append((item, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored[:k] if x[1] >= 0.25]

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    """AI 챗봇: 로컬 Q/A 문맥을 참고해 대화형 응답 제공"""
    try:
        ctx_items = top_k_qa_context(req.message, k=3)
        ctx_text = "\n\n".join([f"- Q: {x['question']}\n  A: {x['answer']}" for x in ctx_items]) or "로컬 문맥 없음"

        system_prompt = (
            "당신은 독일 뷔르츠부르크 교환학생 도우미 챗봇입니다. "
            "가능하면 구체적이고 단계별로 답하세요. "
            "로컬 문맥(아래 제공)과 상충되면 로컬 문맥을 우선하세요. "
            "확실하지 않은 정보는 추측하지 말고 '확인 필요'라고 말하세요."
        )

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",
                 "content": f"사용자 질문:\n{req.message}\n\n[로컬 문맥]\n{ctx_text}"}
            ]
        )

        answer = completion.choices[0].message.content.strip()
        return {"answer": answer, "context": ctx_items}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")

# ===============================
#  실행 (로컬 테스트용)
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
