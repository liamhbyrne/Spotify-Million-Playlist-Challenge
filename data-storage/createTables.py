from DBManager import DBManager

class CreateTables(DBManager):
    def __init__(self):
        super().__init__()

    def create(self):
        with self.getCursor() as c:
            c.execute(open("schema.sql").read())
            self._conn.commit()

if __name__ == '__main__':
    c = CreateTables()
    c.create()