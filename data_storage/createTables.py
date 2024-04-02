import logging

from DBManager import DBManager

logging.getLogger().setLevel(logging.INFO)


class TableCreator(DBManager):
    def __init__(self):
        super().__init__()

    def create(self):
        """
        Execute the schema.sql file to create the tables.
        """
        cursor = self.get_cursor()

        with open("schema.sql", "r") as f:
            schema_script = f.read()

        cursor.executescript(schema_script)
        self._conn.commit()
        logging.info("Tables created")


if __name__ == "__main__":
    """
    Run this script to create the tables in the database.
    """
    tc = TableCreator()
    tc.create()
    tc.disconnect()
