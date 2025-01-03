# .mode csv
# .import train.csv Rate
import sqlite3

# insert into train.csv (name, quote, score) values ('John', 'Hello', 5);
with sqlite3.connect("data.db") as db:
    try:
        cur = db.cursor()

        db.execute("BEGIN TRANSACTION")
        data = cur.execute("""
                INSERT INTO Rate (name, quote, score)
                VALUES (?, ?, ?)
                """, ('John', 'Hello', 5))

        result = data.lastrowid
        if result is not None:
            print(result)
        else:
            print("No data found")
            db.execute("COMMIT")
    except Exception as e:
        db.execute("ROLLBACK")
        print(f"Transaction failed: {e}")
