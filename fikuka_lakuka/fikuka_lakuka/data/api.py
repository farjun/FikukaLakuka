import sqlite3
from typing import Tuple

import numpy as np
import io
from config import config
from fikuka_lakuka.fikuka_lakuka.models import History
from pathlib import Path
DBS_FOLDER = Path(__file__).parent / "runs"


def adapt_array(arr) -> sqlite3.Binary:
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

class DataApi:
    def __init__(self):
        self.db_path = DBS_FOLDER / config.get("general", "db_name")
        self._db_con = sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES)

        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", convert_array)

        self.create_tables()

    def create_tables(self, force_recreate=False):
        cur = self._db_con.cursor()
        if force_recreate:
            print(f"Dropping all Tables!")
            cur.execute(f"drop table if exists history")

        print(f"Running - create tables if not exists")
        cur.execute(f"create table if not exists history (step int, action int, observation int, agents_locations array)")

    def close(self):
        self._db_con.close()

    def commit(self):
        self._db_con.commit()

    def write_history(self, history :History):
        cur = self._db_con.cursor()
        for i, step in enumerate(history.to_db_obj()):
            cur.execute("insert into history (step, action, observation, agents_locations) values (?,?,?,?)",
                        (i, *step))
        self._db_con.commit()
        cur.close()