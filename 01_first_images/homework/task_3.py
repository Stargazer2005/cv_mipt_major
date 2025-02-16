import cv2
import numpy as np


def Rotate(image: np.ndarray, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    angle = angle % 360

    while angle > 90:
        image = Rotate(image, point, 90)
        angle -= 90

    height, width, _ = image.shape

    rad_angle = np.deg2rad(angle)

    new_height = int(np.ceil(height * np.cos(rad_angle) + width * np.sin(rad_angle)))
    new_width = int(np.ceil(width * np.cos(rad_angle) + height * np.sin(rad_angle)))

    pts1 = np.float32([[0, 0], [0, height], [width, 0]])
    pts2 = np.float32(
        [
            [0, width * np.sin(rad_angle)],
            [height * np.sin(rad_angle), new_height],
            [new_width - height * np.sin(rad_angle), 0],
        ]
    )

    M = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, M, (new_width, new_height))

    return image


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """
    # Ваш код
    pass

    return image
