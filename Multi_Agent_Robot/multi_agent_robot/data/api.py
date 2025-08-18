import io
import sqlite3
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from config import config
import psutil
from Multi_Agent_Robot.multi_agent_robot.env.history import History

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

_sql_con = None
def get_cur_connection(force_recreate=False):
    db_path = get_game_db_path(config.cur_game, config.seed)
    global _sql_con
    if not force_recreate and _sql_con is not None:
        return _sql_con, db_path
    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)
    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("array", convert_array)
    sql_con = sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES)
    _sql_con = sql_con
    return sql_con, db_path


def get_game_db_path(cur_game, seed):
    return DBS_FOLDER / f"{cur_game}_{seed}"


class DataApi:

    def __init__(self, force_recreate=False):
        self._db_con, self.db_path = get_cur_connection(force_recreate=force_recreate)
        if force_recreate:
            print(f"Dropping all Tables!")
            self.db_path.unlink(missing_ok=True)
            self._db_con, self.db_path = get_cur_connection(force_recreate=force_recreate)
        self.create_run_history_table()
        self.agents = config.get_in_game_context("playing_agents")

    def create_run_history_table(self, schema_name:str = "env"):
        cur = self._db_con.cursor()
        cur.execute(
            f"create table if not exists {schema_name}_history "
            f"({','.join([ f'{col} string' for col in History.TABLE_COLUMNS])})"
        )
        self._db_con.commit()
        cur.close()

    @property
    def env_history_table_name(self):
        return f"env.history"

    def close(self):
        self._db_con.close()

    def commit(self):
        self._db_con.commit()

    def query(self, query: str, commit = False):
        cur = self._db_con.cursor()
        res = cur.execute(query).fetchall()
        if commit:
            self._db_con.commit()
        cur.close()
        return res

    def write_history(self, history: History, schema: str = "env"):
        cur = self._db_con.cursor()
        for i, step in enumerate(history.to_db_obj()):
            cur.execute(f"insert into {schema}_history {History.TABLE_COLUMNS} "
                        f"values ({','.join(['?']*len(History.TABLE_COLUMNS))})", step)
        self._db_con.commit()
        cur.close()

    def write_history_step(self, history_step: list, schema: str = "env"):
        cur = self._db_con.cursor()
        cur.execute(f"insert into {schema}_history {History.TABLE_COLUMNS} "
                    f"values ({','.join(['?']*len(History.TABLE_COLUMNS))})", history_step)
        self._db_con.commit()
        cur.close()

    def get_history(self, schema: str = "env"):
        cur = self._db_con.cursor()
        res = cur.execute(f"select * from {schema}_history").fetchall()
        res = pd.DataFrame(res, columns=History.TABLE_COLUMNS)
        cur.close()
        return res


