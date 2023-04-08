"""
Метод Нелдера-Мида (метод деформирующегося многогранника, симплекс-метод).
Одна из возможных реализаций с использованием модуля functions

Классы:
    NelderMead - класс, реализующий метод Нелдера-Мида
"""
import random
from typing import Union
import numpy as np
from dataclasses import dataclass, field
from scripts.functions import BaseFunction
from scripts.point import Point


@dataclass(frozen=True)
class Simplex:
    points: tuple = field(init=False)
    function: BaseFunction = field(init=True)

    def __init__(self, function: BaseFunction, *args):
        if not isinstance(function, BaseFunction):
            raise AttributeError("function must be a BaseFunction")
        object.__setattr__(self, "function", function)
        points = self.__create_points(*args)
        object.__setattr__(self, "points", points)

    def __create_points(self, *args):
        points = self.__make_from_args(*args)
        points = self.__make_new(points)
        return self.__sort(points)

    def __make_new(self, points: list):
        size = self.function.dimension
        temp = points.copy()
        if len(temp) == 0:
            temp = [Point.zero(size)]
        points_count = self.function.dimension + 1
        while len(temp) < points_count:
            axis = random.randint(0, size-1)
            new_point = temp[-1] + Point.unit(size, axis)
            value = self.function.calculate(new_point)
            temp.append((new_point, value))
        return temp

    def __make_from_args(self, *args):
        points = list()
        for point in args:
            if self.__check_point(point):
                points.append(point)
            else:
                value = self.function.calculate(point)
                points.append((point, value))
        return points

    def __check_point(self, point):
        size = self.function.dimension
        temp = point
        result = False
        if isinstance(temp, tuple):
            result = True
            temp = point[0]
        if not isinstance(temp, Point):
            raise AttributeError(f"{temp} is not a Point")
        if size != len(temp):
            raise AttributeError(f"{temp} len must be {size}")
        return result

    @staticmethod
    def __sort(points_with_func):
        new_points = points_with_func.copy()
        new_points.sort(key=lambda x: x[1])
        return tuple(new_points)

    @property
    def best(self):
        return self.points[0]

    @property
    def good(self):
        return self.points[-2]

    @property
    def worst(self):
        return self.points[-1]

    def replace(self, index: int,
                new_point: Union[tuple | Point]) -> "Simplex":
        func = self.function
        points = list(self.points)
        if index < -len(points) or index >= len(points):
            raise IndexError(f"index must be in "
                             f"[{-len(points)},{len(points)})")
        if self.__check_point(new_point):
            points[index] = new_point
        else:
            points[index] = (new_point, func.calculate(new_point))
        return Simplex(func, *points)


class NelderMead:
    """Класс, реализующий метод Нелдера-Мида.
    Параметры алгоритма задаются при создании экземпляра класса

    Свойства:
        params - словарь аргументов метода
        function - оптимизироваемая функция в виде sympy.Expr
        simplex - текущий симплекс в виде двумерного списка

    Методы:
        fit - задание функции и начального симплекса
        run - запуск алгоритма
    """

    # pylint: disable=too-many-instance-attributes
    # Nine is reasonable in this case :)
    def __init__(self, *,
                 alpha: float = 1, betta: float = 0.5, gamma: float = 2,
                 max_steps: int = 1000, eps0: float = 0.001,
                 max_blank: int = 10, eps1: float = 0.001):
        """Инициализатор метода

        :param alpha: коэффициент отражения
        :param betta: коэффициент сжатия
        :param gamma: коэффициент растяжения
        :param max_steps: максимальной количество итераций
        :param eps0: предельная дисперсия точек
        :param max_blank: максимальное число "бесполезных" итераций
        :param eps1: минимальная разница между "полезными" итерациями
        """
        self.__alpha = alpha
        self.__betta = betta
        self.__gamma = gamma
        self.__max_steps = max_steps
        self.__eps0 = eps0
        self.__eps1 = eps1
        self.__max_blank = max_blank
        self.__simplex: Simplex = None
        self.__function: BaseFunction = None
        self.__check_args()
        self.__current_blank = 0
        self.__last_value = 0

    @property
    def params(self) -> dict:
        """Параметры метода

        :return: словарь с названиями и значения параметров
        """
        return {"alpha": self.__alpha,
                "betta": self.__betta,
                "gamma": self.__gamma,
                "max_steps": self.__max_steps,
                "eps0": self.__eps0,
                "eps1": self.__eps1,
                "max_blank": self.__max_blank}

    @property
    def function(self) -> BaseFunction:
        """Оптимизируемая функция

        :return: экземпляр класса BaseFunction
        """
        return self.__function

    @property
    def simplex(self) -> Simplex:
        """Текущий симплекс

        :return: двумерный список точек
        """
        # return [list(point) for point, _ in self.__simplex.points]
        return self.__simplex

    def fit(self, function: BaseFunction, *args) -> None:
        if not isinstance(function, BaseFunction):
            raise AttributeError("function not an instance of BaseFunction")
        self.__function = function
        self.__simplex = Simplex(function, *args)

    def run(self, *, action=None) -> float:
        if not isinstance(self.__function, BaseFunction):
            raise AttributeError("No function in class, use fit method")
        iteration = 0
        self.__current_blank = 0
        size = self.__function.dimension
        while iteration <= self.__max_steps:
            sim = self.__simplex
            points = [point for point, _ in sim.points]

            # Вычисление необходимых точек
            zero = Point.zero(size)
            centroid = 1 / len(points[:-1]) * sum(points[:-1], start=zero)
            best, good, worst = sim.best, sim.good, sim.worst
            self.__last_value = best[1]
            # Отражение
            reflected = self.__reflection(centroid)
            # Выбор замены
            if reflected[1] < best[1]:
                # Растяжение
                self.__expansion(centroid, reflected)
            elif reflected[1] < good[1]:
                self.__simplex = sim.replace(-1, reflected)
            else:
                if reflected[1] < worst[1]:
                    self.__simplex = sim.replace(-1, reflected)
                # Сжатие
                self.__contraction(centroid)
            # Опциональное действие
            if callable(action):
                action(self)
            # Условие останова
            if self.__stop():
                break
            iteration += 1
        return self.__simplex.best[1]

    def __reflection(self, centroid) -> tuple:
        """Операция отражения

        :param centroid: центр масс

        :return: кортеж из отражённой точки и значения функции
        """
        worst = self.__simplex.worst
        reflected = (1 + self.__alpha) * centroid - self.__alpha * worst[0]
        return reflected, self.__function.calculate(reflected)

    def __expansion(self, centroid, reflected) -> None:
        """Операция растяжения. Автоматически заменяет худшую точку

        :param centroid: центр масс

        :param reflected: отражённая точка
        """
        expanded = (1-self.__gamma) * centroid + self.__gamma * reflected[0]
        value = self.__function.calculate(expanded)
        if value <= reflected[1]:
            self.__simplex = self.__simplex.replace(-1, (expanded, value))
        else:
            self.__simplex = self.__simplex.replace(-1, reflected)

    def __contraction(self, centroid) -> None:
        """Операция сжатия. Заменяет худшую точку.
        Если необходимо, начинает глобальное сжатие

        :param centroid: центр масс
        """
        worst = self.__simplex.worst
        condensed = (1-self.__betta) * centroid + self.__betta * worst[0]
        value = self.__function.calculate(condensed)
        if value < worst[1]:
            self.__simplex = self.__simplex.replace(-1, (condensed, value))
        else:
            self.__global_contraction()

    def __global_contraction(self) -> None:
        """Операция глобального сжатия"""
        best = self.__simplex.best[0]
        for index, (point, _) in enumerate(self.__simplex.points[1:]):
            new_point = best + (point - best)/2
            # new_value = self.__function.calculate(list(new_point))
            self.__simplex = self.__simplex.replace(index, new_point)

    def __stop(self) -> bool:
        """Условия останова

        :return: булево значение
        """
        current_value = self.__simplex.best[1]
        if abs(current_value - self.__last_value) < self.__eps1:
            self.__current_blank += 1
        else:
            self.__current_blank = 0
        if self.__current_blank == self.__max_blank:
            return True
        dispersion = self.__dispersion()
        return dispersion < self.__eps0

    def __dispersion(self) -> float:
        """ Вычисление дисперсии точек, как в numpy, но для класса Point

        :return: неотрицательное вещественное число
        """
        points = [point for (point, _) in self.__simplex.points]
        size = self.__function.dimension
        mean = 1/len(points) * sum(points, Point.zero(size))
        mean = np.mean(mean.values)
        dispersion = 0
        for point in points:
            temp_p = point - Point.ones(size) * mean
            dispersion += temp_p * temp_p * 1/size
        return dispersion * 1 / len(points)

    def __check_args(self):
        """Проверка параметров метода"""
        if self.__alpha < 0:
            raise AttributeError("alpha should be >= 0")
        if self.__betta < 0:
            raise AttributeError("betta should be >= 0")
        if self.__gamma < 0:
            raise AttributeError("gamma should be >= 0")
        if not isinstance(self.__max_steps, int):
            raise AttributeError("max_steps should be integer")
        if self.__eps0 < 0:
            raise AttributeError("eps0 should be > 0")

    def __str__(self) -> str:
        """Строковое представление текущего состояния метода

        :return: строка с информацией о параметрах, функции и симплексе
        """
        return f"Params: alpha = {self.__alpha}, betta = {self.__betta}," \
               f" gamma = {self.__gamma}\n" \
               f"steps = {self.__max_steps}, accuracy = {self.__eps0}\n" \
               f"Function: {self.__function.expr}\n" \
               f"Simplex: {self.__simplex}"
