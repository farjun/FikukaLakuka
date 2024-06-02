import io
import sqlite3
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from config import config

from Multi_Agent_Robot.multi_agent_robot.env.history import History

HISTORY_TABLE_COLMNS = ("step", "cur_agent", "action", "rock_sample_loc", "observation", "agents_locations", "agent_beliefs", "oracle_action", "oracle_beliefs")

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
    def __init__(self, force_recreate=False, db_name: str = None, schema: str = "env"):
        self.db_name = db_name or config.get("general", "db_name")
        self.db_path = DBS_FOLDER / f"{config.cur_game}_{self.db_name}"
        self.agents = config.get_in_game_context("playing_agents")
        self._db_con = sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES)
        self.schema = schema

        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", convert_array)

        self.create_tables(force_recreate=force_recreate)

    def create_tables(self, force_recreate=False):
        cur = self._db_con.cursor()
        if force_recreate:
            print(f"Dropping all Tables!")
            cur.execute(f"drop table if exists {self.history_table_name}")
            for agent in self.agents:
                cur.execute(f"drop table if exists {self.get_agent_table_name(agent)}")

        cur.execute(
            f"create table if not exists {self.history_table_name} ("
                f"step int, "
                f"cur_agent int, "
                f"action string, "
                f"rock_sample_loc string, "
                f"observation string, "
                f"agents_locations string, "
                f"agent_beliefs string,"
                f"oracle_action string,"
                f"oracle_beliefs string)"
        )
        for agent in self.agents:
            cur.execute(f"create table if not exists {self.get_agent_table_name(agent)} (step int, agent_state array, clustered_state string)")

    @property
    def history_table_name(self):
        return f"{self.schema}_history"

    def get_agent_table_name(self, agent_name: str):
        return f"{self.schema}_agent_{agent_name}"

    def close(self):
        self._db_con.close()

    def commit(self):
        self._db_con.commit()

    def write_history(self, history: History):
        cur = self._db_con.cursor()
        for i, step in enumerate(history.to_db_obj()):
            cur.execute(f"insert into {self.history_table_name} {HISTORY_TABLE_COLMNS} values (?,?,?,?,?,?,?,?,?)",
                        (i, *step))
        self._db_con.commit()
        cur.close()

    def write_agent_state(self, agent: str, step: int, agent_state: np.array, clustered_state: str = 'no value'):
        cur = self._db_con.cursor()
        cur.execute(f"insert into {self.get_agent_table_name(agent)} (step, agent_state, clustered_state) values (?,?,?)",
                    (step, agent_state, clustered_state))
        self._db_con.commit()
        cur.close()

    def get_state(self, agent: str, step: int):
        cur = self._db_con.cursor()
        res = cur.execute(f"select * from {self.get_agent_table_name(agent)} where step={step})  ")
        cur.close()
        return res

    def get_all_from_table(self, agent: str, step: int):
        cur = self._db_con.cursor()
        res = cur.execute(f"select * from {self.get_agent_table_name(agent)} where step={step})  ")
        cur.close()
        return res

    def get_all_states(self, agent: str, flatten_states=False) -> Tuple[int, np.array, str]:
        cur = self._db_con.cursor()
        res = cur.execute(f"select * from {self.get_agent_table_name(agent)} order by step asc").fetchall()
        if flatten_states:
            res = [it[1].flatten() for it in res]

        cur.close()
        return res

    def get_history(self, agent: str = None, as_df=True):
        cur = self._db_con.cursor()
        agents = [it[0] for it in cur.execute(f"select distinct(cur_agent) from {self.history_table_name}").fetchall()]
        agents_history = list()
        for agent in agents:
            res = cur.execute(f"select * from {self.history_table_name} {f'where cur_agent={agent}' if agent is not None else ''}").fetchall()
            agents_history.append(pd.DataFrame(res, columns=HISTORY_TABLE_COLMNS))

        cur.close()

        return agents_history
