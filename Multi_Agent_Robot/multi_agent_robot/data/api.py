import io
import sqlite3
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from config import config

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


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

sql_con = None
class DataApi:

    def __init__(self, force_recreate=False, db_name: str = None, schema: str = "env"):
        self.db_name = db_name or config.get("general", "game_to_run")
        self.db_path = DBS_FOLDER / self.db_name
        if force_recreate:
            print(f"Dropping all Tables!")
            self.db_path.unlink(missing_ok=True)
        self.agents = config.get_in_game_context("playing_agents")
        self._db_con = sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES)
        self.create_run_history_table()

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

    def get_history(self, agent: str = None, as_df=True):
        cur = self._db_con.cursor()
        agents = [it[0] for it in cur.execute(f"select distinct(agent_selection) from {self.env_history_table_name}").fetchall()]
        agents_history = list()
        for agent in agents:
            res = cur.execute(f"select * from {self.env_history_table_name} {f'where agent_selection={agent}' if agent is not None else ''}").fetchall()
            agents_history.append(pd.DataFrame(res, columns=History.TABLE_COLUMNS))

        cur.close()

        return agents_history


