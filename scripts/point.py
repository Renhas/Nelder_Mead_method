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
        return Point(*((0,)*dimension))

    @staticmethod
    def ones(dimension: int):
        Point.__dimension_check(dimension)
        return Point(*((1,)*dimension))

    @staticmethod
    def unit(dimension: int, axis: int):
        if axis >= dimension:
            raise AttributeError("axis must be < dimension")
        if dimension < 0:
            raise AttributeError("dimension must be > 0")
        if axis < 0:
            raise AttributeError("axis must be > 0")
        values = [0]*dimension
        values[axis] = 1
        return Point(*values)

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
        return Point(*new_val)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, Point):
            return self.__mul_point(other)
        if isinstance(other, numbers.Number):
            return self.__mul_number(other)
        raise TypeError("Other must be a Point or a Number")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul_point(self, other: "Point"):
        self.__check_len(other)
        pairs = zip(self.values, other.values)
        return sum([x_val * y_val for x_val, y_val in pairs])

    def __mul_number(self, number: numbers.Number):
        return Point(*[value*number for value in self.values])

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return self.__mul_number(1/other)

    def __sub__(self, other):
        return self + other * -1

    def __rsub__(self, other):
        return self.__sub__(other)

    def __len__(self) -> int:
        return len(self.values)

    def __str__(self):
        if len(self) == 1:
            return f"({self.values[0]})"
        return str(self.values)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        if len(self) != len(other):
            return False
        for first, second in zip(self.values, other.values):
            if first != second:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)



