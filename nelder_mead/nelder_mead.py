"""
Метод Нелдера-Мида (метод деформирующегося многогранника, симплекс-метод).
"""
from typing import Union, Callable, Tuple, List
from dataclasses import dataclass, field
import numpy as np
from utilities.functions import BaseFunction
from utilities.point import Point


@dataclass(frozen=True)
class Simplex:
    """Класс симплекса метода Нелдера-Мида"""
    points: Tuple[Tuple[Point, float]] = field(init=False)
    function: BaseFunction = field(init=True)

    def __init__(self, function: BaseFunction, *args):
        """Инициализатор класса

        Args:
            function: функция для вычисления значения в точках
            *args: произвольное количество начальных :class:`~utilities.point.Point`

        Raises:
            AttributeError: если function не является BaseFunction
        """
        if not isinstance(function, BaseFunction):
            raise AttributeError("function must be a BaseFunction")
        object.__setattr__(self, "function", function)
        points = self.__create_points(*args)
        object.__setattr__(self, "points", points)

    @property
    def best(self) -> Tuple[Point, float]:
        """Первая точка симплекса"""
        return self.points[0]

    @property
    def good(self) -> Tuple[Point, float]:
        """Предпоследняя точка симплекса"""
        return self.points[-2]

    @property
    def worst(self) -> Tuple[Point, float]:
        """Последняя точка симплекса"""
        return self.points[-1]

    def sort(self) -> "Simplex":
        """Сортирует список пар исходя из значений функции

        Returns:
            Новый, отсортированный симплекс
        """
        new_points = list(self.points)
        new_points.sort(key=lambda x: x[1])
        return Simplex(self.function, *new_points)

    def replace(self, index: int,
                new_point: Union[Tuple[Point, float] | Point]) -> "Simplex":
        """Замещает одну точку на другую.
         Если передаётся просто точка, для неё вычисляется значение

        Args:
            index: индекс заменяемой точки
            new_point: новая точка или кортеж (Точка, Значение)

        Returns:
            Новый симплекс с заменённой точкой

        Raises:
            IndexError: если индекс за пределами [-n, n],
             где n - размерность симплекса
        """
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

    def __create_points(self, *args) -> Tuple[Tuple[Point, float]]:
        """Создание точек для симплекса

        Args:
            *args: произвольное количество точек или пар

        Returns:
            Готовый кортеж из пар (Точка, Значение)
        """
        points = self.__make_from_args(*args)
        points = self.__make_new(points)
        return tuple(points)

    def __make_new(self, points: List[Tuple[Point, float]]) \
            -> List[Tuple[Point, float]]:
        """Создание новых точек.\n
        Каждая последующая точка получатся из предыдущей путём сдвига на 1
        в одном из направлений из возможных для данной размерности.
        Если переданный список пуст, то генерация начинается с нулевой точки

        Args:
            points: список уже готовых пар

        Returns:
            Полностью готовый список пар
        """
        size = self.function.dimension
        temp = points.copy()
        if len(temp) == 0:
            zero = Point.zero(size)
            value = self.function.calculate(zero)
            temp = [(zero, value)]
        points_count = self.function.dimension + 1
        axis = 0
        while len(temp) < points_count:
            new_point = temp[-1][0] + Point.unit(size, axis)
            value = self.function.calculate(new_point)
            temp.append((new_point, value))
            axis += 1
            if axis >= size:
                axis = 0
        return temp

    def __make_from_args(self, *args) -> List[Tuple[Point, float]]:
        """Создание необходимых пар из переданных точек.\n
        Переданы могут быть как сами точки, так и готовая пара.
        Пары не изменяются, для одиночных точек вычисляется значение

        Args:
            *args: произвольное количество точек

        Returns:
            Список пар
        """
        points = []
        for point in args:
            if self.__check_point(point):
                points.append(point)
            else:
                value = self.function.calculate(point)
                points.append((point, value))
        return points

    def __check_point(self, point: Union[Point, Tuple[Point, float]]) -> bool:
        """Проверка точки. Если передана пара, то проверяется точка из пары

        Args:
            point: точка или пара (Точка, Значение)

        Returns:
            True, если передана пара. False, иначе

        Raises:
            AttributeError: если проверяемый объект не является точкой
             или его длина не корректна
        """
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


class NelderMead:
    """Класс, реализующий метод Нелдера-Мида.
    При инициализации экземпляра класса, задаются только параметры алгоритма,
    что позволяет использовать один экземпляр с разными математическими функциями.
    """

    # pylint: disable=too-many-instance-attributes
    # Nine is reasonable in this case :)
    def __init__(self, *,
                 alpha: float = 1, betta: float = 0.5, gamma: float = 2,
                 max_steps: int = 1000, eps0: float = 0.001,
                 max_blank: int = 10, eps1: float = 0.001):
        """Инициализатор класса

        Args:
            alpha: коэффициент отражения
            betta: коэффициент сжатия
            gamma: коэффициент растяжения
            max_steps: максимальной количество итераций
            eps0: предельная дисперсия точек
            max_blank: максимальное число "бесполезных" итераций
            eps1: минимальная разница между "полезными" итерациями
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
        """Параметры метода"""
        return {"alpha": self.__alpha,
                "betta": self.__betta,
                "gamma": self.__gamma,
                "max_steps": self.__max_steps,
                "eps0": self.__eps0,
                "eps1": self.__eps1,
                "max_blank": self.__max_blank}

    @property
    def function(self) -> BaseFunction:
        """Оптимизируемая функция"""
        return self.__function

    @property
    def simplex(self) -> Simplex:
        """Текущий симплекс"""
        return self.__simplex

    def fit(self, function: BaseFunction, *args):
        """Инициализация оптимизируемой функции и начального симплекса.

        Args:
            function: оптимизируемая функция
            *args: произвольное количество :class:`~utilities.point.Point` начального симплекса
        """
        if not isinstance(function, BaseFunction):
            raise AttributeError("function not an instance of BaseFunction")
        self.__function = function
        self.__simplex = Simplex(function, *args)

    def run(self, *, action: Callable = None) -> float:
        """Запуск метода

        Args:
            action: опциональное действие в конце каждой итерации.
             Должен принимать экземпляр
             :class:`~nelder_mead.nelder_mead.NelderMead`

        Returns:
            Найденный минимум
        Raises:
            AttributeError - в случае, когда нет экземпляра
             :class:`~utilities.functions.BaseFunction`
        """
        if not isinstance(self.__function, BaseFunction):
            raise AttributeError("No function in class, use fit method")
        iteration = 0
        self.__current_blank = 0
        self.__simplex = self.__simplex.sort()
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
            self.__simplex = self.__simplex.sort()
            # Опциональное действие
            if callable(action):
                action(self)
            # Условие останова
            if self.__stop():
                break
            iteration += 1
            self.__last_value = self.__simplex.best[1]
        return self.__simplex.best[1]

    def __reflection(self, centroid: Point) -> Tuple[Point, float]:
        """Операция отражения

        Args:
            centroid: центр масс

        Returns:
            Кортеж из отражённой точки и значения функции
        """
        worst = self.__simplex.worst
        reflected = (1 + self.__alpha) * centroid - self.__alpha * worst[0]
        return reflected, self.__function.calculate(reflected)

    def __expansion(self, centroid: Point, reflected: Tuple[Point, float]):
        """Операция растяжения. Автоматически заменяет худшую точку

        Args:
            centroid: центр масс
            reflected: отражённая точка со значением функции в ней
        """
        expanded = (1-self.__gamma) * centroid + self.__gamma * reflected[0]
        value = self.__function.calculate(expanded)
        if value <= reflected[1]:
            self.__simplex = self.__simplex.replace(-1, (expanded, value))
        else:
            self.__simplex = self.__simplex.replace(-1, reflected)

    def __contraction(self, centroid: Point):
        """Операция сжатия. Заменяет худшую точку.
        Если необходимо, начинает глобальное сжатие

        Args:
            centroid: центр масс
        """
        worst = self.__simplex.worst
        condensed = (1-self.__betta) * centroid + self.__betta * worst[0]
        value = self.__function.calculate(condensed)
        if value < worst[1]:
            self.__simplex = self.__simplex.replace(-1, (condensed, value))
        else:
            self.__global_contraction()

    def __global_contraction(self):
        """Операция глобального сжатия"""
        best = self.__simplex.best[0]
        for index, (point, _) in enumerate(self.__simplex.points[1:]):
            new_point = best + (point - best)/2
            # new_value = self.__function.calculate(list(new_point))
            self.__simplex = self.__simplex.replace(index, new_point)

    def __stop(self) -> bool:
        """Условия останова

        Returns:
            `True`, если метод нужно остановить. `False`, иначе
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
        """Вычисление дисперсии точек, как в numpy, но для класса Point

        Returns:
            Неотрицательное вещественное число
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
        """Проверка параметров метода

        Returns:
            AttributeError - если какой-то из параметров не соответствует ограничениям.
        """
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

        Returns:
            Строка с информацией о параметрах, функции и симплексе
        """
        return f"Params: alpha = {self.__alpha}, betta = {self.__betta}," \
               f" gamma = {self.__gamma}\n" \
               f"steps = {self.__max_steps}, accuracy = {self.__eps0}\n" \
               f"Function: {self.__function.expr}\n" \
               f"Simplex: {self.__simplex}"
