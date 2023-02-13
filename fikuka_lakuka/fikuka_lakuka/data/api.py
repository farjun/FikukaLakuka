import sqlite3
from config import config


class DataApi:

    def __init__(self):
        self._con = sqlite3.connect(config.get("general", "db_name"))

    def create_tables(self):
        pass
