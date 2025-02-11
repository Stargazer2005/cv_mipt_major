from typing import Literal
import cv2
import numpy as np
import enum


Point = tuple[int, int]
Cell = tuple[int, int]
Color = tuple[int, int, int]

WHITE: Color = (255, 255, 255)
BLACK: Color = (0, 0, 0)
RED: Color = (255, 0, 0)

class Direction(enum.Enum):
    UP = (0, -1)
    LEFT = (-1, 0)
    DOWN = (0, 1)
    RIGHT = (1, 0)


def FindWayFromMaze(image: np.ndarray) -> tuple[list[int], list[int]]:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return path: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """

    height, width, _ = image.shape
    # начальное направление
    glob_direction = Direction.DOWN 

    # Размер ячейки - размер входа + 2
    cell_size = len(np.where(np.all(image[0] == WHITE, axis=1))[0]) + 2


    def _FindEntryAndExit() -> tuple[Point, Point]:
        """
        Найти вход и выход лабиринта.

        :return tuple[int, int]: координаты входа и выхода лабиринта соответственно в виде (x, y).
        """

        entry = np.where(
                np.all(image[0] == WHITE, axis=1))[0]
        exit = np.where(
                np.all(image[height - 1] == WHITE, axis=1))[0]
        
        return ((entry[entry.size // 2], 0), (exit[exit.size // 2], height - 1))

 
    def _FillWithColor(point: Point, color: Color) -> np.ndarray:
        """
        Залить область вокруг точки определенным цветом.

        :param point: точка, вокруг которой будет залита область
        :param color: цвет, которым будет залита область
        :return ff_image: изображение с залитой областью
        """

        mask = np.zeros((height+2, width+2), dtype=np.uint8)

        _, ff_image, _, _ = cv2.floodFill(image, mask, point, color)

        return ff_image
    

    entry_c, exit_c = _FindEntryAndExit()
    entry_cell = ((entry_c[0] - 1 - cell_size // 2) // cell_size, 0)
    exit_cell = ((exit_c[0] - 1 - cell_size // 2) // cell_size, (height - cell_size // 2 - 2) // cell_size)
    wall = (entry_c[0] - cell_size // 2, entry_c[1])
    image = _FillWithColor(wall, RED)


    def _GetCellCenter(cell: Cell) -> Point:
        """
        Получить ячейку лабиринта по заданным координатам

        :param point: ячейка лабиринта
        :return: координаты центра (x, y)
        """

        i, j = cell

        x = i * cell_size + cell_size // 2 + 1
        y = j * cell_size + cell_size // 2 + 1

        return (x, y)
    

    def _GetWallColor(cell_center: Point, delta: tuple[int, int]) -> Color:
        return tuple(image[cell_center[1] + delta[1] * cell_size // 2, cell_center[0] + delta[0] * cell_size // 2])
    

    def _GetCellWalls(cell: Cell) -> dict[Direction, Color | None]:
        """
        Получить список стен , которые окружают ячейку лабиринта
        
        :param cell: ячейка лабиринта
        :return: список стен, которые окружают ячейку
        """
        cell_center = _GetCellCenter(cell)
        walls = {}

        for direction in Direction:
            walls[direction] = _GetWallColor(cell_center, direction.value) \
                if not(np.equal(_GetWallColor(cell_center, direction.value), WHITE)).all() else None
            
        return walls


    def _GetCellNeighbours(cell: Cell) -> dict[Direction, Cell | None]:
        """
        Получить список соседних ячеек лабиринта

        :param cell: ячейка лабиринта
        :return: список соседних ячеек лабиринта
        """
        cell_center = _GetCellCenter(cell)
        neighbours = {}
        
        for direction in Direction:
            neighbours[direction] = (cell[0] + direction.value[0], cell[1] + direction.value[1]) \
                if np.equal(_GetWallColor(cell_center, direction.value), WHITE).all() else None

        return neighbours


    def _GetNextCell(cell: Cell, g_direct: Direction) -> tuple[Cell | None, Direction]:
        """
        Получить следующую ячейку лабиринта в пути

        :param cell: ячейка лабиринта
        :param g_direct: текущее направление движения
        :return: следующая ячейка лабиринта в пути и новое направление движения
        """

        def _RotateClockWise(direction: Direction) -> Direction:
            return Direction((-direction.value[1], direction.value[0]))
            
        def _RotateAntiClockWise(direction: Direction) -> Direction:
            return Direction((direction.value[1], -direction.value[0]))
        
        neighbours = _GetCellNeighbours(cell)
        walls = _GetCellWalls(cell)
        candidates = [_RotateClockWise(g_direct), g_direct, _RotateAntiClockWise(g_direct), _RotateClockWise(_RotateClockWise(g_direct))]
        for candidate in candidates:
            if neighbours[candidate] is not None:
                n_walls = _GetCellWalls(neighbours[candidate])
                right_wall = n_walls[_RotateClockWise(candidate)]
                left_wall = n_walls[_RotateAntiClockWise(candidate)]
                if (right_wall == RED):
                    if(left_wall == BLACK):
                        return (neighbours[candidate], candidate)
                elif (right_wall == BLACK):
                    return (neighbours[_RotateClockWise(candidate)], _RotateClockWise(candidate))
                else:
                    if(left_wall == BLACK):
                        return (neighbours[candidate], candidate)
            else:
                if walls[candidate] == BLACK:
                    return (neighbours[_RotateClockWise(candidate)], _RotateClockWise(candidate))

    path = ([entry_c[0], _GetCellCenter(entry_cell)[0]], [entry_c[1], _GetCellCenter(entry_cell)[1]])
    cell = entry_cell
    
    while(cell != exit_cell):
        next_cell = exit_cell
        if (_GetNextCell(cell, glob_direction) is not None):
            next_cell, glob_direction = _GetNextCell(cell, glob_direction)
        path[0].append(_GetCellCenter(next_cell)[0])
        path[1].append(_GetCellCenter(next_cell)[1])
        cell = next_cell

    path[0].append(exit_c[0])
    path[1].append(exit_c[1])

    image = _FillWithColor(wall, BLACK)

    return path