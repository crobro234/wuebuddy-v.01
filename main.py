from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sqlite3
from passlib.context import CryptContext
import jwt, datetime

app = FastAPI()

# CORS 허용 (프론트엔드와 통신)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 (index.html 서빙)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

DB_PATH = "data/exchange_helper.db"
SECRET_KEY = "super_secret_key"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# -------------------- DB 초기화 --------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 기존 categories, questions 테이블
    cur.execute("""CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_id INTEGER,
        question TEXT,
        answer TEXT,
        FOREIGN KEY (category_id) REFERENCES categories(id))""")

    # ✅ users 테이블 추가
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password_hash TEXT
    )""")

    conn.commit()
    conn.close()

init_db()

# -------------------- 보안 함수 --------------------
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str):
    return pwd_context.verify(plain, hashed)

def create_token(username: str):
    payload = {
        "sub": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=6)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
# 회원가입
@app.post("/register")
def register(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? OR email=?", (username, email))
    if cur.fetchone():
        raise HTTPException(status_code=400, detail="이미 존재하는 사용자입니다.")
    pw_hash = hash_password(password)
    cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, pw_hash))
    conn.commit()
    conn.close()
    return {"message": "✅ 회원가입 완료!"}

# -------------------- 회원가입 --------------------
@app.post("/register")
def register(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? OR email=?", (username, email))
    if cur.fetchone():
        raise HTTPException(status_code=400, detail="이미 존재하는 사용자입니다.")
    pw_hash = hash_password(password)
    cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)", (username, email, pw_hash))
    conn.commit()
    conn.close()
    return {"message": "✅ 회원가입 완료!"}

# -------------------- 로그인 --------------------
@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row or not verify_password(password, row[0]):
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 틀렸습니다.")
    token = create_token(username)
    return {"message": "로그인 성공", "token": token}

# -------------------- (선택) 내 정보 확인 --------------------
@app.get("/me")
def me(request: Request):
    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(status_code=401, detail="토큰이 없습니다.")
    try:
        token = token.replace("Bearer ", "")
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username = payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="토큰이 만료되었습니다.")
    return {"username": username}

# -------------------- 기존 Q&A API --------------------
@app.get("/categories")
def get_categories():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM categories")
    rows = cur.fetchall()
    conn.close()
    return {"categories": [{"id": r[0], "name": r[1]} for r in rows]}

@app.get("/questions/{category_id}")
def get_questions(category_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, question FROM questions WHERE category_id=?", (category_id,))
    rows = cur.fetchall()
    conn.close()
    return {"questions": [{"id": r[0], "question": r[1]} for r in rows]}

@app.get("/answer/{question_id}")
def get_answer(question_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT answer FROM questions WHERE id=?", (question_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return {"answer": row[0]}
    else:
        raise HTTPException(status_code=404, detail="질문을 찾을 수 없습니다.")
