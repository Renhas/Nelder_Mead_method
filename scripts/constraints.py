"""
Модуль, реализующий функционал ограничений

Классы:
    Constraint - абстрактный базовый класс всех ограничений
    Equality - ограничение-равенство
    Inequality - ограничение-неравенство
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import sympy as sm
from scripts.functions import BaseFunction
from scripts.point import Point


@dataclass(frozen=True)
class Constraint(ABC):
    """Абстрактный класс, базовый для всех ограничений, иммутабельный

    Поля:
        function - функция-ограничение
    Свойства:
        error_func - функция штрафа
    Методы:
        error(Point) - величина штрафа в точке
        check(Point) - проверка на выполнимость
    """
    function: BaseFunction = field(init=False)

    def __init__(self, function: BaseFunction):
        """Конструктор класса

        :param function: функция-ограничение
        """
        if not isinstance(function, BaseFunction):
            raise AttributeError("function must be a BaseFunction")
        object.__setattr__(self, "function", function)

    @property
    @abstractmethod
    def error_func(self) -> BaseFunction:
        """Функция штрафа

        :return: функция
        """
        pass

    def error(self, point: Point) -> float:
        """Величина штрафа в заданной точке

        :param point: точка
        :return: значение штрафа
        """
        return self.error_func.calculate(point)

    def check(self, point: Point) -> bool:
        """Проверка на выполнимость в заданной точке

        :param point: точка
        :return: булево значение
        """
        return self.error(point) == 0


class Equality(Constraint):
    """Класс для ограничения-равенства вида f(x) = 0"""
    @property
    def error_func(self) -> BaseFunction:
        """Функция штрафа |f(x)|

        :return: функция
        """
        func = self.function.expr
        new_func = abs(func)
        return BaseFunction(new_func, self.function.variables)


class Inequality(Constraint):
    """Класс для ограничения неравенства вида f(x) <= 0"""
    @property
    def error_func(self) -> BaseFunction:
        """Функция штрафа max(0, f(x))

        :return: функция
        """
        func = self.function.expr
        new_func = sm.Max(0, func)
        return BaseFunction(new_func, self.function.variables)
