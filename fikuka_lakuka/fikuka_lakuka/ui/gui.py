import os

import pygame

from fikuka_lakuka.fikuka_lakuka.models.i_state import IState
from fikuka_lakuka.fikuka_lakuka.ui.coord import Coord, Tile

PATH = os.path.split(__file__)[0]
FILE_PATH = os.path.join(PATH, 'assets')


class GuiTile(Tile):
    _borderWidth = 2
    _borderColor = pygame.Color("grey")

    def __init__(self, coord, surface, tile_size=100):
        self.origin = (coord.x * tile_size, coord.y * tile_size)
        self.surface = surface
        self.tile_size = tuple([tile_size] * 2)
        super().__init__(coord)

    def draw(self, img=None, color=pygame.Color("white")):
        rect = pygame.Rect(self.origin, self.tile_size)
        pygame.draw.rect(self.surface, color, rect, 0)  # draw tile
        if img is not None:
            self.surface.blit(img, self.origin)
        pygame.draw.rect(self.surface, GuiTile._borderColor, rect, GuiTile._borderWidth)  # draw border


class GridGui(object):
    _assets = {}

    def __init__(self, x_size, y_size, tile_size):
        self.x_size = x_size
        self.y_size = y_size
        self.n_tiles = x_size * y_size
        self.tile_size = tile_size
        self.assets = {}
        for k, v in self._assets.items():
            self.assets[k] = pygame.transform.scale(pygame.image.load(v), [tile_size] * 2)

        self.w = self.tile_size * self.x_size
        self.h = (self.tile_size * self.y_size) + 50  # size of the taskbar

        pygame.init()
        self.surface = pygame.display.set_mode((self.w, self.h))
        self.surface.fill(pygame.Color("white"))
        self.action_font = pygame.font.SysFont("monospace", 18)
        self.build_board()

    def build_board(self):
        self.board = []
        for idx in range(self.n_tiles):
            tile = GuiTile(self.get_coord(idx), surface=self.surface, tile_size=self.tile_size)
            tile.draw(img=None)
            self.board.append(tile)

    def get_coord(self, idx):
        assert idx >= 0 and idx < self.n_tiles
        return Coord(idx % self.x_size, idx // self.x_size)

    def draw(self, state: IState):
        raise NotImplementedError()

    def render(self, state: IState, msg):
        raise NotImplementedError()

    def task_bar(self, msg):
        assert msg is not None
        txt = self.action_font.render(msg, 2, pygame.Color("black"))
        rect = pygame.Rect((0, self.h - 50 + 5), (self.w, 40))  # 205 for tiger
        pygame.draw.rect(self.surface, pygame.Color("white"), rect, 0)
        self.surface.blit(txt, (self.tile_size // 2, self.h - 50 + 10))  # 210

    @staticmethod
    def _dispatch():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None


class RockGui(GridGui):
    _tile_size = 50
    _assets = dict(
        _ROBOT0=os.path.join(FILE_PATH, "r2d2.png"),
        _ROBOT1=os.path.join(FILE_PATH, "robot1.png"),
        _ROCK=os.path.join(FILE_PATH, "rock.png"),
        _DOOR=os.path.join(FILE_PATH, "door.png")
    )

    def __init__(self, state: IState):
        super().__init__(*state.grid_size, tile_size=self._tile_size)
        self.history = [state.cur_agent_ui_location()]
        self.draw(state)
        pygame.display.update()
        GridGui._dispatch()

    def draw(self, state: IState):
        last_state = self.history[-1]
        cur_pos = state.cur_agent_ui_location()

        self.history.append(cur_pos)
        self.board[last_state].draw()
        self.board[state.get_end_pt(as_ui_idx=True)].draw(
                img=self.assets["_DOOR"])
        for i, rock in enumerate(state.rocks):
            if rock.picked:
                color = pygame.Color('white')
            elif rock.reward > 0:
                color = pygame.Color('green')
            else:
                color = pygame.Color('red')

            self.board[rock.ui_loc].draw(img=self.assets["_ROCK"], color=color)

        for agent_id in range(state.num_of_agents):
            self.board[state.get_agent_location(agent_id, as_ui_idx=True)].draw(
                img=self.assets["_ROBOT" + str(agent_id)])

    def render(self, state: IState, msg=None):
        self.draw(state)
        # self.task_bar(msg)
        pygame.display.update()
        GridGui._dispatch()
