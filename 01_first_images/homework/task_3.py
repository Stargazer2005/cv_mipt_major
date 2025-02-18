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


def FindCorners(image_mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Нахождение углов тедради по маске.

    :param image_mask: маска изображения
    :return: углы тедради
    """
    corners: list[tuple[int, int]] = []

    def _FindCorner(is_transposed: bool, is_reversed: bool) -> None:
        mask = np.transpose(image_mask) if is_transposed else image_mask
        height, width = mask.shape
        for i in range(height):
            k = (height - i - 1) if is_reversed else i
            if max(mask[k]) == 255:
                for j in range(width):
                    if mask[k][j] == 255:
                        corners.append((k, j) if is_transposed else (j, k))
                        break
                break

    for is_reversed in [False, True]:
        for is_transposed in [False, True]:
            _FindCorner(is_transposed, is_reversed)

    return corners
