import sqlite3

def init_db():
    conn = sqlite3.connect("data/exchange_helper.db")
    cur = conn.cursor()

    # 카테고리 테이블
    cur.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT
    )
    """)

    # 질문 테이블
    cur.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_id INTEGER,
        question TEXT,
        answer TEXT,
        FOREIGN KEY (category_id) REFERENCES categories (id)
    )
    """)

    # 예시 데이터
    cur.execute("INSERT INTO categories (name) VALUES ('비자'), ('전화번호')")
    cur.execute("""
    INSERT INTO questions (category_id, question, answer)
    VALUES
    (1, '비자 인터뷰는 어떻게 신청하나요?', '외국인청 홈페이지(Ausländerbehörde)에서 Termin 예약을 하세요.'),
    (1, '비자 수령은 어디서 하나요?', 'Ausländerbehörde 건물 내 Service Point에서 가능합니다.'),
    (2, '독일 번호는 어떻게 만들어요?', 'Aldi Talk 유심을 REWE나 Aldi에서 구매 후 등록하면 됩니다.')
    """)

    conn.commit()
    conn.close()

# 파일 실행하면 DB 자동 생성
if __name__ == "__main__":
    init_db()
    print("✅ DB 초기화 완료!")
