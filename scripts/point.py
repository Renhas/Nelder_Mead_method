import math
import numbers
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Point:
    values: tuple = field(init=False)

    def __init__(self, *args):
        object.__setattr__(self, 'values', args)

    @staticmethod
    def zero(dimension: int):
        Point.__dimension_check(dimension)
        return Point((0,)*dimension)

    @staticmethod
    def ones(dimension: int):
        Point.__dimension_check(dimension)
        return Point((1,)*dimension)

    @staticmethod
    def __dimension_check(dimension: int):
        if not isinstance(dimension, int):
            raise AttributeError("Dimension must be a integer")
        if dimension < 1:
            raise AttributeError("Dimension must be > 0")

    def distance(self, other):
        Point.__check_point(other)
        self.__check_len(other)
        return math.sqrt((self - other) ** 2)

    @staticmethod
    def __check_point(other):
        if not isinstance(other, Point):
            raise TypeError("Other must be a Point")

    def __check_len(self, other):
        if len(other) != len(self):
            raise AttributeError(f"Other's length must be ={len(self)}")

    def __add__(self, other):
        Point.__check_point(other)
        self.__check_len(other)
        new_val = zip(self.values, other.values)
        new_val = [x_val + y_val for x_val, y_val in new_val]
        return Point(tuple(new_val))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, Point):
            self.__mul_point(other)
        if isinstance(other, numbers.Number):
            self.__mul_number(other)
        raise TypeError("Other must be a Point or a Number")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul_point(self, other):
        self.__check_len(other)
        pairs = zip(self.values, other.values)
        return sum([x_val * y_val for x_val, y_val in pairs])

    def __mul_number(self, number):
        return Point(tuple(value*number for value in self.values))

    def __sub__(self, other):
        return self + other * -1

    def __rsub__(self, other):
        return self.__sub__(other)

    def __len__(self) -> int:
        return len(self.values)

