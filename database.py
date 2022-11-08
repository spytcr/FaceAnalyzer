import sqlite3


class Database:
    UNKNOWN_NAME = 'Неизвестно'
    UNKNOWN_EMOTION = 'Нейтрально'

    def __init__(self, path):
        self.connect = sqlite3.connect(path)
        self.cursor = self.connect.cursor()

    def update_preference(self, key, value):
        self.cursor.execute('''UPDATE preferences SET value = ? WHERE key = ?''', (value, key))
        self.connect.commit()

    def get_preference(self, key):
        return self.cursor.execute('''SELECT value FROM preferences WHERE key = ?''', (key,)).fetchone()[0]

    def get_bool(self, key):
        return self.get_preference(key) == 'True'

    def get_people(self):
        return self.cursor.execute('''SELECT * FROM people''').fetchall()

    def add_person(self, name):
        self.cursor.execute('''INSERT INTO people(name) VALUES(?)''', (name,))
        self.connect.commit()

    def update_person(self, uid, name):
        self.cursor.execute('''UPDATE people SET name = ? WHERE id = ?''', (name, uid))
        self.connect.commit()

    def delete_person(self, uid):
        self.cursor.execute('''DELETE FROM people WHERE id = ?''', (uid,))
        self.connect.commit()

    def get_emotions(self):
        return self.cursor.execute('''SELECT * FROM emotions''').fetchall()

    def add_emotion(self, name):
        self.cursor.execute('''INSERT INTO emotions(title) VALUES(?)''', (name,))
        self.connect.commit()

    def update_emotion(self, uid, name):
        self.cursor.execute('''UPDATE emotions SET title = ? WHERE id = ?''', (name, uid))
        self.connect.commit()

    def delete_emotion(self, uid):
        self.cursor.execute('''DELETE FROM emotions WHERE id = ?''', (uid,))
        self.connect.commit()

    def clear_data(self):
        self.cursor.execute('''DELETE FROM data''')
        self.connect.commit()

    def get_data(self):
        return self.cursor.execute('''SELECT path, people.name, emotions.title FROM data
                                      JOIN people ON people.id = data.person_id
                                      JOIN emotions ON emotions.id = data.emotion_id''').fetchall()

    def insert_data(self, data):
        self.cursor.execute('''INSERT INTO data(path, person_id, emotion_id) 
        VALUES(?, (SELECT id FROM people WHERE name = ?), (SELECT id FROM emotions WHERE title = ?))''', data)
        self.connect.commit()

    def close(self):
        self.connect.close()
