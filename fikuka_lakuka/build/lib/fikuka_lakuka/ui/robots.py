from fikuka_lakuka.fikuka_lakuka.models.i_state import IState

from fikuka_lakuka.fikuka_lakuka.ui.board import BoardUI
from pathlib import Path
import tkinter as tk

BASE_PATH  = Path(__file__).parent

class RobotsUI(BoardUI):
    IMG_MAPPING = {
        IState.ROBOT2: {"image" : BASE_PATH / "assets" /  "robot1.png"},
        IState.ROBOT1:  {"image" : BASE_PATH / "assets" /  "robot2.png"},
        IState.EMPTY: {"bg" : "gray"},
        IState.START: {"bg" : "black"},
        IState.END: {"bg" : "green"},
        IState.ROCK: {"image" : BASE_PATH / "assets" /  "rock.png"},
    }

    def __init__(self):
        super().__init__(tk.Tk(), RobotsUI.IMG_MAPPING)

if __name__ == '__main__':
    board = RobotsUI()
