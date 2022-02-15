import logging

from . import DBManager
logging.getLogger().setLevel(logging.INFO)

class TableCreator(DBManager):
    def __init__(self):
        super().__init__()

    def create(self):
        with self.getCursor() as c:
            c.execute(open("schema.sql").read())
            self._conn.commit()
            logging.info("Tables created")


if __name__ == '__main__':
    c = TableCreator()
    c.create()
    c.disconnect()
