"""
Модуль для тестирования библиотеки математических функций

Классы:
    TestBaseFunction - тестирование базового класса
    TestPolynomial - тестирование класса полиномиальных функций
    TestRosenbroke - тестирование функции Розенброка
    TestHimmelblau - тестирование функции Химмельблау
"""
import typing
import pytest
import sympy as sm
from scripts.functions import BaseFunction, Polynomial, Rosenbroke
from scripts.functions import Himmelblau
from scripts.point import Point

# Символы для символьных выражений
variables = sm.symbols("x1:10")
x_var, y_var = sm.symbols("x,y")


class TestBaseFunction:
    """Класс тестирования базовой функции

    Методы:
        test_create(sympy.Expr, tuple) - тестирование создания экземпляра\n
        test_calculate(sympy.Expr, tuple, list, tuple) - тестирование
        вычисления функции\n
        test_str(sympy.Expr, tuple, str) - тестирование
        строкового представления функции
    """

    @pytest.mark.parametrize(
        ("expression", "expr_var"), [
            (sm.sin(variables[0] + sm.cos(variables[1])),
             (variables[0], variables[1])),
            pytest.param(0, (), marks=pytest.mark.xfail(strict=True)),
            pytest.param(sm.sin(0) + variables[1] * 2, (),
                         marks=pytest.mark.xfail(strict=True)),
            pytest.param(variables[1], 0,
                         marks=pytest.mark.xfail(strict=True)),
            pytest.param(variables[1], [1, 3],
                         marks=pytest.mark.xfail(strict=True)),
            pytest.param(variables[1], (33, "2"),
                         marks=pytest.mark.xfail(strict=True)),
            pytest.param(variables[1], (variables[0],),
                         marks=pytest.mark.xfail(strict=True)),
            pytest.param(variables[1], [variables[1]],
                         marks=pytest.mark.xfail(strict=True)),

        ]
    )
    def test_create(self, expression: sm.Expr, expr_var: tuple):
        """Тестирование создания экземпляра класса

        :param expression: символьное выражение
        :param expr_var: кортеж символов-переменных
        """
        func = BaseFunction(expression, expr_var)
        assert func.expr == expression
        assert func.dimension == len(expr_var)
        assert func.variables == expr_var

    @pytest.mark.parametrize(
        ("expression", "expr_var", "test_input", "expected"), [
            (2.0 * sm.ln(variables[0]), (variables[0],), [sm.E], (2.0, 1)),
            (x_var ** 2 + x_var * y_var + y_var ** 2 - 6 * x_var - 9 * y_var,
             (x_var, y_var), [1, 0], (-5, 1)),
            (variables[0] ** 2 + variables[1] ** 3, (variables[0], variables[1]),
             [1, -1], (0, 1)),
            (2.0 * sm.ln(variables[0]), (variables[0],), Point(sm.E), (2.0, 1)),
            (x_var ** 2 + x_var * y_var + y_var ** 2 - 6 * x_var - 9 * y_var,
             (x_var, y_var), Point(1, 0), (-5, 1))
        ]

    )
    def test_calculate(self, expression: sm.Expr, expr_var: tuple,
                       test_input: list, expected: tuple):
        """Тестирования вычисления функции

        :param expression: символьное выражение
        :param expr_var: кортеж символов-переменных
        :param test_input: значения переменных
        :param expected: кортеж с ожидаемым значением и точностью сравнения
        """
        func = BaseFunction(expression, expr_var)
        result = func.calculate(test_input)
        assert result == pytest.approx(expected[0], abs=expected[1])

    @pytest.mark.parametrize(
        ("expression", "expr_var", "view"), [
            (2.0 * sm.ln(variables[0]), (variables[0],), "2.0*log(x1)")
        ]
    )
    def test_str(self, expression: sm.Expr, expr_var: tuple, view: str):
        """Тестирование строкового представления функции

        :param expression: символьное выражение
        :param expr_var: кортеж символов-переменных
        :param view: ожидаемое строковое представление
        """
        func = BaseFunction(expression, expr_var)
        assert str(func) == view

    @pytest.mark.parametrize(
        ("first", "second", "expected"), [
            (Rosenbroke(), Rosenbroke(), True),
            (Rosenbroke(), Himmelblau(), False),
            (BaseFunction(sm.E * variables[0], (variables[0],)),
             BaseFunction(sm.E * variables[1], (variables[1],)), False),
            (Rosenbroke(), 34, False)
        ]
    )
    def test_equal(self, first: BaseFunction, second: typing.Any,
                   expected: bool):
        """Тестирование сравнения функций

        :param first: Первая функция
        :param second: Вторая функция
        :param expected: Ожидаемый результат
        """
        assert (first == second) == expected

    @pytest.mark.parametrize(
        ("functions", "expected"), [
            ([Rosenbroke(), Rosenbroke()],
             BaseFunction(2*(1-x_var)**2 + 200*(y_var - x_var**2)**2,
                          (x_var, y_var))),
            ([BaseFunction(variables[0] ** 2, (variables[0],)),
             BaseFunction(sm.E ** 2, ()),
             BaseFunction(sm.pi / variables[1], (variables[1],)),
             BaseFunction(variables[2] + 1, (variables[2],))],
             BaseFunction(variables[0]**2 + sm.E**2+sm.pi/variables[1] +
                          variables[2] + 1,
                          (variables[0], variables[1], variables[2]))),
            ([Rosenbroke(), 35 + 7 ** 12],
             BaseFunction((1-x_var)**2+100*(y_var - x_var**2)**2 + 35 + 7**12,
                          (x_var, y_var)))

        ]
    )
    def test_sum(self, functions: list, expected: BaseFunction):
        """Тестирование суммирования функций

        :param functions: список функций для сложения
        :param expected: ожидаемая функция
        """
        new_function = sum(functions)
        assert new_function.expr == expected.expr
        assert new_function.variables == expected.variables


class TestPolynomial:
    """Тестирование функции произвольных полиномов

    Методы:
        test_create(list, sympy.Expr, int) - тестирование создания экземпляра
        test_calculate(list, list, float, float) - тестирование вычисления
    """

    @pytest.mark.parametrize(
        ("coefficients", "view", "variable_count"), [
            ([[0, 2, 3], [0, 3, 4]],
             2 * variables[0] + 3 * variables[0] ** 2 +
             3 * variables[1] + 4 * variables[1] ** 2,
             2)
        ]
    )
    def test_create(self, coefficients: list, view: sm.Expr,
                    variable_count: int):
        """Тестирование создания экземпляра класса

        :param coefficients: коэффициенты полинома
        :param view: символьный вид полинома
        :param variable_count: количество переменных
        """
        poly = Polynomial(coefficients)
        assert poly.expr == view
        assert poly.dimension == variable_count

    @pytest.mark.parametrize(
        ("coefficients", "test_input", "test_output", "accuracy"), [
            ([[0, 2, 3], [0, 3, 4]], [0, 0], 0.0, 1),
            ([[0, 2, 3], [0, 3, 4]], [1, 2], 27.0, 1),
            ([[0, 1], [0, 2, 1]], [1, 1], 4.0, 1)
        ]
    )
    def test_calculate(self, coefficients: list, test_input: list,
                       test_output: float, accuracy: float):
        """Тестирование вычисления функции

        :param coefficients: двумерный список коэффициентов
        :param test_input: значения переменных
        :param test_output: ожидаемое значение
        :param accuracy: точность сравнения
        """
        poly = Polynomial(coefficients)
        result = poly.calculate(test_input)
        assert result == pytest.approx(test_output, abs=accuracy)


class TestRosenbroke:
    """Тестирование функции Розенброка

    Методы:
        test_create() - тестирование создания функции
        test_calculate(list, float, float) - тестирование вычисления функции
    """

    def test_create(self):
        """Тестирование создания функции Розенброка"""
        assert Rosenbroke().expr == (1 - x_var) ** 2 + 100 * (y_var - x_var ** 2) ** 2
        assert Rosenbroke().dimension == 2

    @pytest.mark.parametrize(
        ("test_input", "test_output", "accuracy"), [
            ([1, 1], 0, 1)
        ]
    )
    def test_calculate(self, test_input, test_output, accuracy):
        """Тестирование вычисления функции Розенброка

        :param test_input: значения переменных
        :param test_output: ожидаемый результат
        :param accuracy: точность сравнения
        """
        result = Rosenbroke().calculate(test_input)
        assert result == pytest.approx(test_output, accuracy)


class TestHimmelblau:
    """Тестирование функции Химмельблау

    Методы:
        test_create() - тестирование создания экземпляра
        test_calculate(list, float, float) - тестирование вычисления
    """

    def test_create(self):
        """Тестирование создания функции Химмельблау"""
        expected = (x_var ** 2 + y_var - 11) ** 2 + (x_var + y_var ** 2 - 7) ** 2
        assert Himmelblau().expr == expected
        assert Himmelblau().dimension == 2

    @pytest.mark.parametrize(
        ("test_input", "test_output", "accuracy"), [
            ([-0.270845, -0.923039], 181.617, 0.005),
            ([3, 2], 0, 0),
            pytest.param([0, 0], 0, 0, marks=pytest.mark.xfail(strict=True)),
            ([-2.805118, 3.131312], 0, 0.1)
        ]
    )
    def test_calculate(self, test_input, test_output, accuracy):
        """Тестирование вычисления функции Химмельблау

        :param test_input: значения переменных
        :param test_output: ожидаемый результат
        :param accuracy: точность сравнения
        """
        result = Himmelblau().calculate(test_input)
        assert result == pytest.approx(test_output, abs=accuracy)
