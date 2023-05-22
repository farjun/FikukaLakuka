from pathlib import Path
from tkinter import Image
import numpy as np

from config import config
import tkinter as tk

class BoardUI:
    SPOT_SIZE = 32

    def __init__(self, window, img_mapping: dict):
        self.img_mapping = img_mapping
        self.window = window
        self.grid_size = config.get_in_game_context("environment", "grid_size")

        self.canvas = tk.Canvas(window, width=self.grid_size[0]*BoardUI.SPOT_SIZE, height=self.grid_size[1]*BoardUI.SPOT_SIZE, highlightthickness=0)
        self.canvas.pack()

    def show(self, board: np.ndarray):
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                squere_val = board[i][j]
                values = self.img_mapping[squere_val]
                if "image" in values and isinstance(values["image"], (str,Path)):
                    values["image"] = tk.PhotoImage(file=values["image"], master=self.window)
                    self.img_mapping[squere_val] = values

                tk.Label(self.canvas, width=20, height=10, **values).grid(row=i, column=j)


