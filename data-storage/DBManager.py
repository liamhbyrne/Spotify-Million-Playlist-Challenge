import psycopg2
import logging
import config

logging.getLogger().setLevel(logging.INFO)

class DBManager:
    def __init__(self):
        self._address = config.POSTGRES_ADDRESS
        self._conn = self.connect()

    def connect(self):
        if self._address:  # If address has been provided in environment
            try:
                logging.info("Attempting to connect to database.")
                return psycopg2.connect(self._address)
            except psycopg2.OperationalError:
                raise Exception("Failed to connect to DB")
            finally:
                logging.info("Connected to database")
        else:
            raise Exception("DB_ADDRESS not provided in environmental variables")

    def disconnect(self):
        self._conn.close()
        logging.info("Disconnected from DB")

    def getCursor(self):
        if not self._conn.closed:
            return self._conn.cursor()
        raise Exception("Connection is no longer alive")
