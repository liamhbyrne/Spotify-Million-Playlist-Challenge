import sqlite3
import logging
import config

logging.getLogger().setLevel(logging.INFO)


class DBManager:
    def __init__(self):
        self._address = config.DB_ADDRESS
        self._conn = self.connect()

    def connect(self):
        if self._address:  # If address has been provided in environment
            try:
                logging.info("Attempting to connect to database.")
                return sqlite3.connect(self._address)
            except sqlite3.Error:
                raise Exception("Failed to connect to DB")
            finally:
                logging.info("Connected to database")
        else:
            raise Exception("SQLITE_ADDRESS not provided in config file")

    def disconnect(self):
        self._conn.close()
        logging.info("Disconnected from DB")

    def get_cursor(self):
        if self._conn is not None:
            return self._conn.cursor()
        raise Exception("Connection is no longer alive")

    def drop_all_tables(self):
        """
        Use with great care.
        """
        cursor = self.get_cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table in tables:
            cursor.execute(f"DROP TABLE {table[0]};")
        self._conn.commit()
        logging.info("All tables dropped")


if __name__ == '__main__':
    """ List all tables in the database and number of rows in each table."""
    db = DBManager()
    cursor = db.get_cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]};")
        count = cursor.fetchone()[0]
        print(f"{table[0]}: {count}")

    db.disconnect()
