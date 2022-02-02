import os
import psycopg2


class DBManager:
    def __init__(self):
        self._address = os.environ.get('DB_ADDRESS')
        self._conn = self.connect()

    def connect(self):
        if self._address:  # If address has been provided in environment
            try:
                return psycopg2.connect(self._address)
            except psycopg2.OperationalError:
                raise Exception("Failed to connect to DB")
        else:
            raise Exception("DB_ADDRESS not provided in environmental variables")

    def getCursor(self):
        if not self._conn.closed:
            return self._conn.cursor()
        raise Exception("Connection is no longer alive")
