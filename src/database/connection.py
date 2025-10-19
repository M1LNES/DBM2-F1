import duckdb
from contextlib import contextmanager


class DatabaseConnection:
    def __init__(self, db_path):
        self.db_path = db_path
        self._connection = None

    def get_connection(self):
        if self._connection is None:
            self._connection = duckdb.connect(self.db_path)
        return self._connection

    @contextmanager
    def get_cursor(self):
        conn = self.get_connection()
        try:
            yield conn
        except Exception as e:
            print(f"Database error: {e}")
            raise

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None
