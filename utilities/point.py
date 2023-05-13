"""
Многомерные точки (вектора) и взаимодействие с ними

Классы:
    Point - клас, реализующий многомерную точку и операции с ней
"""
import math
import numbers
from typing import Union
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Point:
    """
    Многомерная точка (иммутабельная).
    В конструкторе принимает произвольное количество объектов

    Поля:
        values - кортеж значений точки
    Методы класса:
        zero(int) - нулевая точка
        ones(int) - единичная точка
        unit(int) - единичный орт
    Методы экземпляра:
        distance(Point) - расстояние между точками

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
        """Конструктор класса

        :param args: произвольное количество значений
        """
        if len(args) == 0:
            raise AttributeError("No items given")
        object.__setattr__(self, 'values', args)

    @staticmethod
    def zero(dimension: int) -> "Point":
        """Нулевая точка

        :param dimension: размерность
        :return: нулевая точка заданной размерности
        """
        Point.__dimension_check(dimension)
        return Point(*((0,)*dimension))

    @staticmethod
    def ones(dimension: int) -> "Point":
        """Единичная точка

        :param dimension: размерность
        :return: единичная точка заданной размерности
        """
        Point.__dimension_check(dimension)
        return Point(*((1,)*dimension))

    @staticmethod
    def unit(dimension: int, axis: int) -> "Point":
        """Единичный орт

        :param dimension: размерность
        :param axis: единичная координата
        :return: Единичный орт указанной размерности для указанной оси
        """
        if not 0 <= axis < dimension:
            raise AttributeError(f"axis must be in [0, {dimension})")
        if dimension < 0:
            raise AttributeError("dimension must be > 0")
        values = [0]*dimension
        values[axis] = 1
        return Point(*values)

    @staticmethod
    def __dimension_check(dimension: int):
        """Проверка размерности

        :param dimension: размерность
        """
        if not isinstance(dimension, int):
            raise AttributeError("Dimension must be a integer")
        if dimension < 1:
            raise AttributeError("Dimension must be > 0")

    def distance(self, other: "Point") -> float:
        """Евклидово расстояние между точками

        :param other: вторая точка
        :return: расстояние
        """
        Point.__check_point(other)
        self.__check_len(other)
        return math.sqrt((self - other) ** 2)

    @staticmethod
    def __check_point(other: "Point"):
        """Проверка точки

        :param other: точка
        """
        if not isinstance(other, Point):
            raise TypeError("Other must be a Point")

    def __check_len(self, other: "Point"):
        """Проверка длины

        :param other: вторая точка
        """
        if len(other) != len(self):
            raise AttributeError(f"Other's length must be ={len(self)}")

    def __add__(self, other: "Point") -> "Point":
        """Сложение точек

        :param other: вторая точка
        :return: новая точка
        """
        Point.__check_point(other)
        self.__check_len(other)
        new_val = zip(self.values, other.values)
        new_val = [x_val + y_val for x_val, y_val in new_val]
        return Point(*new_val)

    def __radd__(self, other: "Point") -> "Point":
        """Сложение точек справа

        :param other: вторая точка
        :return: новая точка
        """
        return self.__add__(other)

    def __mul__(self, other: Union[numbers.Number, "Point"]) \
            -> Union["Point", numbers.Number]:
        """Умножение точки

        :param other: число или точка
        :return: точка или число
        """
        if isinstance(other, Point):
            return self.__mul_point(other)
        if isinstance(other, numbers.Number):
            return self.__mul_number(other)
        raise TypeError("Other must be a Point or a Number")

    def __rmul__(self, other: Union[numbers.Number, "Point"]) \
            -> Union["Point", numbers.Number]:
        """Умножение справа

        :param other: число или точка
        :return: точка или число
        """
        return self.__mul__(other)

    def __mul_point(self, other: "Point") -> numbers.Number:
        """Умножение на точку

        :param other: вторая точка
        :return: число
        """
        self.__check_len(other)
        pairs = zip(self.values, other.values)
        return sum(x_val * y_val for x_val, y_val in pairs)

    def __mul_number(self, number: numbers.Number) -> "Point":
        """Умножение на число

        :param number: число
        :return: точка
        """
        return Point(*[value*number for value in self.values])

    def __truediv__(self, other: numbers.Number) -> "Point":
        """Целочисленное деление

        :param other: число
        :return: точка
        """
        if isinstance(other, numbers.Number):
            return self.__mul_number(1/other)
        raise AttributeError(f"{other} must be a Number")

    def __sub__(self, other: "Point") -> "Point":
        """Вычитание

        :param other: вторая точка
        :return: точка
        """
        return self + other * -1

    def __rsub__(self, other: "Point") -> "Point":
        """Вычитание справа

        :param other: вторая точка
        :return: точка
        """
        return self.__sub__(other)

    def __len__(self) -> int:
        """Длина точки

        :return: длина
        """
        return len(self.values)

    def __str__(self):
        """Строковое представление

        :return: строка вида (x,y,z)
        """
        if len(self) == 1:
            return f"({self.values[0]})"
        return str(self.values)

    def __eq__(self, other: "Point") -> bool:
        """Проверка на равенство точек

        :param other: вторая точка
        :return: булево значение
        """
        if not isinstance(other, Point):
            return False
        if len(self) != len(other):
            return False
        for first, second in zip(self.values, other.values):
            if first != second:
                return False
        return True

    def __ne__(self, other: "Point") -> bool:
        """Проверка на неравенство

        :param other: вторая точка
        :return: булево значение
        """
        return not self.__eq__(other)

    def __pow__(self, power: int, modulo: int = None) \
            -> Union["Point", numbers.Number]:
        """Возведение в степень

        :param power: целая неотрицательная степень
        :param modulo: модуль
        :return: число или точка
        """
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
