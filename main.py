from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import openai
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

app = FastAPI()
openai.api_key = "YOUR_OPENAI_API_KEY"  # GPT api key 일단공란
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 임시로 전체 허용 (나중엔 특정 도메인만)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    conn = sqlite3.connect("data/exchange_helper.db")
    return conn

@app.get("/")
def serve_home():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


# loading categories
@app.get("/categories")
def get_categories():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM categories")
    data = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]
    conn.close()
    return {"categories": data}

# showing those Q
@app.get("/questions/{category_id}")
def get_questions(category_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, question FROM questions WHERE category_id=?", (category_id,))
    data = [{"id": r[0], "question": r[1]} for r in cur.fetchall()]
    conn.close()
    return {"questions": data}

# Answer from that Q
@app.get("/answer/{question_id}")
def get_answer(question_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT answer FROM questions WHERE id=?", (question_id,))
    result = cur.fetchone()
    conn.close()
    return {"answer": result[0] if result else "해당 질문에 대한 답변이 없습니다."}

# Extra Q 
class AskModel(BaseModel):
    question: str

@app.post("/ask")
async def ask_gpt(data: AskModel):
    user_q = data.question
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "저는 님을 도우러 온 뷔르츠부르크대학 전용 챗봇입니다."},
            {"role": "user", "content": user_q}
        ]
    )
    return {"answer": response["choices"][0]["message"]["content"]}
