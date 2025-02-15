import cv2
import numpy as np

color_bonds_HSV = {
    "RED": ((0, 50, 70), (9, 255, 255)),
    "BLUE": ((90, 50, 70), (128, 255, 255)),
    "GRAY": ((0, 0, 40), (180, 18, 230)),
}


def FindRoadNumber(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """
    height, width, _ = image.shape

    safe_road_number, car_road_number = None, None

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    road_bonds: dict[int, tuple[int, int]] = {}
    road_mask = cv2.inRange(
        hsv_image, color_bonds_HSV["GRAY"][0], color_bonds_HSV["GRAY"][1]
    )

    road_count = -1
    prev = road_mask[0][0]
    for i in range(1, width):
        if (road_mask[0][i] == 255) and (prev == 0):
            road_count += 1
            road_bonds[road_count] = (i, None)
        if (road_mask[0][i] == 0) and (prev == 255):
            road_bonds[road_count] = (road_bonds[road_count][0], i)
        prev = road_mask[0][i]

    obstacle_mask = cv2.inRange(
        hsv_image, color_bonds_HSV["RED"][0], color_bonds_HSV["RED"][1]
    )
    car_mask = cv2.inRange(
        hsv_image, color_bonds_HSV["BLUE"][0], color_bonds_HSV["BLUE"][1]
    )

    for road_num in road_bonds.keys():
        road_center = (road_bonds[road_num][0] + road_bonds[road_num][1]) // 2
        for i in range(height):
            if obstacle_mask[i, road_center] == 255:
                break
        else:
            safe_road_number = road_num

        for i in range(height):
            if car_mask[i, road_center] == 255:
                car_road_number = road_num

    road_number = safe_road_number if safe_road_number != car_road_number else None

    return road_number
