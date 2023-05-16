"""
Модуль, реализующий функционал математических ограничений
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import sympy as sm
from utilities.functions import BaseFunction
from utilities.point import Point


@dataclass(frozen=True)
class Constraint(ABC):
    """Абстрактный класс, базовый для всех ограничений, иммутабельный"""
    function: BaseFunction = field(init=False)

    def __init__(self, function: BaseFunction):
        """Инициализатор класса

        Args:
            function: функция-ограничение

        Raises:
            AttributeError - если передан не экземпляр
             :class:`~utilities.functions.BaseFunction`
        """
        if not isinstance(function, BaseFunction):
            raise AttributeError("function must be a BaseFunction")
        object.__setattr__(self, "function", function)

    @property
    @abstractmethod
    def error_func(self) -> BaseFunction:
        """Функция штрафа, построенная на основе функции-ограничения"""

    def error(self, point: Point) -> float:
        """Величина штрафа в заданной точке"""
        return self.error_func.calculate(point)

    def check(self, point: Point) -> bool:
        """Проверка на выполнимость в заданной точке

        Args:
            point: проверяемая точка

        Returns:
            True, если ошибка равна 0. False, иначе
        """
        return self.error(point) == 0


class Equality(Constraint):
    """Ограничение-равенство вида :math:`f(x)=0`"""
    @property
    def error_func(self) -> BaseFunction:
        """Функция штрафа :math:`|f(x)|`"""
        func = self.function.expr
        new_func = abs(func)
        return BaseFunction(new_func, self.function.variables)


class Inequality(Constraint):
    r"""Ограничение-неравенство вида :math:`f(x)\leq0`"""
    @property
    def error_func(self) -> BaseFunction:
        """Функция штрафа :math:`max(0,f(x))`"""
        func = self.function.expr
        new_func = sm.Max(0, func)
        return BaseFunction(new_func, self.function.variables)
