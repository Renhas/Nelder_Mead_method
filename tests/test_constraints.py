"""
Модуль для тестирования классов-ограничений

Классы:
    TestEquality - равенства
    TestInequality - неравенства
"""
import pytest
import sympy as sm
from scripts.constraints import Equality, Inequality
from scripts.functions import BaseFunction
from scripts.point import Point

x_var, y_var = sm.symbols("x,y")
function_first = BaseFunction(x_var*y_var, (x_var, y_var))
function_second = BaseFunction(x_var**2 - y_var**2, (x_var, y_var))


class TestEquality:
    """Класс для тестирования ограничений-равенств

    Методы:
        test_error_func(BaseFunction) - вид штрафной функции
        test_error(BaseFunction, Point, float, float) - подсчёт ошибки
        test_check(BaseFunction, Point, bool) - проверка ограничения
    """
    @pytest.mark.parametrize(
        "func", [
            function_first,
            function_second
        ]
    )
    def test_error_func(self, func: BaseFunction):
        """Тестирование штрафной функции

        :param func: функция-ограничение
        """
        equality = Equality(func)
        expr = sm.Abs(func.expr)
        assert equality.error_func == BaseFunction(expr, func.variables)

    @pytest.mark.parametrize(
        ("func", "point", "expected", "accuracy"), [
            (function_first, Point(0, 0), 0, 0),
            (function_first, Point(1, 1), 1, 0),
            (function_first, Point(-3, 1), 3, 0)
        ]
    )
    def test_error(self, func: BaseFunction, point: Point, expected: float,
                   accuracy: float):
        """Тестирование подсчёта штрафа

        :param func: функция-ограничение
        :param point: точка
        :param expected: ожидаемое значение
        :param accuracy: точность
        """
        equality = Equality(func)
        assert equality.error(point) == pytest.approx(expected, abs=accuracy)

    @pytest.mark.parametrize(
        ("func", "point", "expected"), [
            (function_first, Point(0, 0), True),
            (function_first, Point(1, 1), False)
        ]
    )
    def test_check(self, func: BaseFunction, point: Point, expected: bool):
        """Тестирование проверки ограничения

        :param func: функция-ограничение
        :param point: точка
        :param expected: ожидаемое значение
        """
        equality = Equality(func)
        assert equality.check(point) == expected


class TestInequality:
    """Класс для тестирования ограничений-неравенств

    Методы:
        test_error_func(BaseFunction) - вид штрафной функции
        test_error(BaseFunction, Point, float, float) - подсчёт ошибки
        test_check(BaseFunction, Point, bool) - проверка ограничения
    """
    @pytest.mark.parametrize(
        "func", [
            function_first,
            function_second
        ]
    )
    def test_error_func(self, func: BaseFunction):
        """Тестирование штрафной функции

        :param func: функция-ограничение
        """
        equality = Inequality(func)
        expected = BaseFunction(sm.Max(0, func), func.variables)
        assert equality.error_func == expected

    @pytest.mark.parametrize(
        ("func", "point", "expected", "accuracy"), [
            (function_first, Point(0, 0), 0, 0),
            (function_first, Point(1, 1), 1, 0),
            (function_first, Point(-3, 1), 0, 0)
        ]
    )
    def test_error(self, func: BaseFunction, point: Point, expected: float,
                   accuracy: float):
        """Тестирование подсчёта штрафа

        :param func: функция-ограничение
        :param point: точка
        :param expected: ожидаемое значение
        :param accuracy: точность
        """
        equality = Inequality(func)
        assert equality.error(point) == pytest.approx(expected, abs=accuracy)

    @pytest.mark.parametrize(
        ("func", "point", "expected"), [
            (function_first, Point(0, 0), True),
            (function_first, Point(1, 1), False),
            (function_first, Point(-3, 1), True)
        ]
    )
    def test_check(self, func: BaseFunction, point: Point, expected: bool):
        """Тестирование проверки ограничения

        :param func: функция-ограничение
        :param point: точка
        :param expected: ожидаемое значение
        """
        equality = Inequality(func)
        assert equality.check(point) == expected
