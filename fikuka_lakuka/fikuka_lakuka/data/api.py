import io
import sqlite3
from pathlib import Path
from typing import Tuple

import numpy as np

from config import config
from fikuka_lakuka.fikuka_lakuka.models import History

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
    def __init__(self, force_recreate = False):
        self.db_path = DBS_FOLDER / config.get("general", "db_name")
        self.agents = config.get_in_game_context("playing_agents")
        self._db_con = sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES)

        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", convert_array)

        self.create_tables(force_recreate=force_recreate)

    def create_tables(self, force_recreate=False):
        cur = self._db_con.cursor()
        if force_recreate:
            print(f"Dropping all Tables!")
            cur.execute(f"drop table if exists history")
            for agent in self.agents:
                cur.execute(f"drop table if exists agent_{agent}")

        print(f"Running - create tables if not exists")
        cur.execute(f"create table if not exists history (step int, cur_agent int, action string, observation string, agents_locations string)")
        for agent in self.agents:
            cur.execute(f"create table if not exists agent_{agent} (step int, agent_state array, clustered_state string)")

    def close(self):
        self._db_con.close()

    def commit(self):
        self._db_con.commit()

    def write_history(self, history: History):
        cur = self._db_con.cursor()
        for i, step in enumerate(history.to_db_obj()):
            cur.execute("insert into history (step, cur_agent, action, observation, agents_locations) values (?,?,?,?,?)",
                        (i, *step))
        self._db_con.commit()
        cur.close()

    def write_agent_state(self, agent:str, step: int, agent_state: np.array, clustered_state:str = 'no value'):
        cur = self._db_con.cursor()
        cur.execute(f"insert into agent_{agent} (step, agent_state, clustered_state) values (?,?,?)",
                    (step, agent_state, clustered_state))
        self._db_con.commit()
        cur.close()

    def get_state(self, agent:str, step: int):
        cur = self._db_con.cursor()
        res = cur.execute(f"select * from agent_{agent} where step={step})  ")
        cur.close()
        return res

    def get_all_states(self, agent: str, flatten_states= False)->Tuple[int, np.array, str]:
        cur = self._db_con.cursor()
        res = cur.execute(f"select * from agent_{agent} order by step asc").fetchall()
        if flatten_states:
            res = [it[1].flatten() for it in res]

        cur.close()
        return res
