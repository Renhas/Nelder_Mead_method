import numbers
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    values: tuple

    @property
    def dimension(self) -> int:
        return len(self.values)

    def __add__(self, other):
        if not isinstance(other, Point):
            raise TypeError("Other must be a Point")
        if other.dimension != self.dimension:
            raise AttributeError(f"Other's length must be ={self.dimension}")
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
        if other.dimension != self.dimension:
            raise AttributeError(f"Other's length must be ={self.dimension}")
        pairs = zip(self.values, other.values)
        return sum([x_val * y_val for x_val, y_val in pairs])

    def __mul_number(self, number):
        return Point(tuple(value*number for value in self.values))

    def __sub__(self, other):
        return self + other * -1

    def __rsub__(self, other):
        return self.__sub__(other)


