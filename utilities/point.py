"""
Многомерные точки (вектора) и взаимодействие с ними
"""
import math
import numbers
from typing import Union
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Point:
    """
    Многомерная точка (иммутабельная).

    Определённые операции:
        Сложение точек
        Вычитание точек
        Умножение на число и скалярное произведение точек
        Деление на число
        Возведение в целую неотрицательную степень
        Проверка на равенство
        Приведение к строке
        Вычисление длины
    """
    values: tuple = field(init=False)

    def __init__(self, *args):
        """Инициализатор класса

        Args:
            *args: произвольные объекты
        Raises:
            AttributeError - если не передано ни одного объекта
        """
        if len(args) == 0:
            raise AttributeError("No items given")
        object.__setattr__(self, 'values', args)

    @staticmethod
    def zero(dimension: int) -> "Point":
        """Нулевая точка

        Args:
            dimension: размерность

        Returns:
            Точка вида (0, 0, ..., 0)

        Raises:
            AttributeError - если переданы некорректная размерность
        """
        Point.__dimension_check(dimension)
        return Point(*((0,)*dimension))

    @staticmethod
    def ones(dimension: int) -> "Point":
        """Единичная точка

        Args:
            dimension: размерность

        Returns:
            Точка вида (1, 1, ..., 1)

        Raises:
            AttributeError - если переданы некорректная размерность
        """
        Point.__dimension_check(dimension)
        return Point(*((1,)*dimension))

    @staticmethod
    def unit(dimension: int, axis: int) -> "Point":
        """Единичный орт

        Args:
            dimension: размерность
            axis: ось (от 0 до dimension - 1)

        Returns:
            Единичный орт указанной размерности для указанной оси

        Raises:
            AttributeError - если переданы некорректная ось или размерность
        """
        Point.__dimension_check(dimension)
        if not 0 <= axis < dimension:
            raise AttributeError(f"axis must be in [0, {dimension})")
        values = [0]*dimension
        values[axis] = 1
        return Point(*values)

    @staticmethod
    def __dimension_check(dimension: int):
        """Проверка размерности

        Args:
            dimension: размерность

        Raises:
            AttributeError - если размерность меньше 0
        """
        if not isinstance(dimension, int):
            raise AttributeError("Dimension must be a integer")
        if dimension < 1:
            raise AttributeError("Dimension must be > 0")

    def distance(self, other: "Point") -> float:
        """Евклидово расстояние между точками

        Args:
            other: вторая точка

        Returns:
            :math:`\\sqrt{(X-Y)^2}`
        """
        Point.__check_point(other)
        self.__check_len(other)
        return math.sqrt((self - other) ** 2)

    @staticmethod
    def __check_point(other: "Point"):
        """Проверка точки

        Args:
            other: точка

        Raises:
            TypeError - если other не точка
        """
        if not isinstance(other, Point):
            raise TypeError("Other must be a Point")

    def __check_len(self, other: "Point"):
        """Проверка длины

        Args:
            other: вторая точка

        Raises:
            AttributeError - если other не является точкой
        """
        if len(other) != len(self):
            raise AttributeError(f"Other's length must be ={len(self)}")

    def __add__(self, other: "Point") -> "Point":
        """Сложение точек"""
        Point.__check_point(other)
        self.__check_len(other)
        new_val = zip(self.values, other.values)
        new_val = [x_val + y_val for x_val, y_val in new_val]
        return Point(*new_val)

    def __radd__(self, other: "Point") -> "Point":
        """Сложение точек справа"""
        return self.__add__(other)

    def __mul__(self, other: Union[numbers.Number, "Point"]) \
            -> Union["Point", numbers.Number]:
        """Умножение точки"""
        if isinstance(other, Point):
            return self.__mul_point(other)
        if isinstance(other, numbers.Number):
            return self.__mul_number(other)
        raise TypeError("Other must be a Point or a Number")

    def __rmul__(self, other: Union[numbers.Number, "Point"]) \
            -> Union["Point", numbers.Number]:
        """Умножение точки справа"""
        return self.__mul__(other)

    def __mul_point(self, other: "Point") -> numbers.Number:
        """Умножение на точку"""
        self.__check_len(other)
        pairs = zip(self.values, other.values)
        return sum(x_val * y_val for x_val, y_val in pairs)

    def __mul_number(self, number: numbers.Number) -> "Point":
        """Умножение на число"""
        return Point(*[value*number for value in self.values])

    def __truediv__(self, other: numbers.Number) -> "Point":
        """Целочисленное деление"""
        if isinstance(other, numbers.Number):
            return self.__mul_number(1/other)
        raise AttributeError(f"{other} must be a Number")

    def __sub__(self, other: "Point") -> "Point":
        """Вычитание"""
        return self + other * -1

    def __rsub__(self, other: "Point") -> "Point":
        """Вычитание справа"""
        return self.__sub__(other)

    def __len__(self) -> int:
        """Размерность точки"""
        return len(self.values)

    def __str__(self):
        """Строковое представление"""
        if len(self) == 1:
            return f"({self.values[0]})"
        return str(self.values)

    def __eq__(self, other: "Point") -> bool:
        """Проверка на равенство точек"""
        if not isinstance(other, Point):
            return False
        if len(self) != len(other):
            return False
        for first, second in zip(self.values, other.values):
            if first != second:
                return False
        return True

    def __ne__(self, other: "Point") -> bool:
        """Проверка на неравенство точек"""
        return not self.__eq__(other)

    def __pow__(self, power: int, modulo: int = None) \
            -> Union["Point", numbers.Number]:
        """Возведение в степень"""
        result = 1
        if not isinstance(power, int):
            raise AttributeError("power must be an integer")
        if power < 0:
            raise AttributeError("power must be >= 0")
        if modulo is not None and not isinstance(modulo, int):
            raise AttributeError("modulo must be an integer")
        for _ in range(power):
            result *= self
        return result if modulo is None else result % modulo
