"""
Модуль тестирования для алгоритма решения задачи условной оптимизации

Функции:
    data_first() - набор данных для первой тестовой задачи (простая задача)
    data_second() - набор данных для второй тестовой задачи (труднодифференцируемая функция)
    data_third() - набор данных для третьей тестовой задачи (несколько ограничений)
Классы:
    TestConditional - тест для тестирования ConditionalNelderMead
"""
import pytest
import sympy as sm
from nelder_mead.conditional_nm import ConditionalNelderMead
from nelder_mead.nelder_mead import NelderMead
from utilities import Equality, Inequality
from utilities.functions import BaseFunction, Polynomial
from utilities.point import Point


def data_first():
    """Входные данные для первой тестовой задачи:
        (x1 - 1)**2 + (x2)**2 -> min

        x1 + x2 - 0.5 <= 0

        Решение: X* = (0.75, -0.25), Q(X*) = 0.125
    :return: кортеж из: параметры алгоритма, метод Нелдера-Мида, целевая функция
        кортеж ограничений, стартовая точка, ожидаемое значение и точность,
        ожидаемая точка и точность.
    """
    method = NelderMead()
    func = Polynomial([[1, -2, 1], [0, 0, 1]])
    constr_func = Polynomial([[-0.5, 1], [0, 1]])
    constraints = (Inequality(constr_func), )
    true_value = (0.125, 0.001)
    true_point = (Point(0.75, -0.25), 0.01)
    start_point = Point(0, 0)
    params = {}
    return params, method, func, constraints, start_point,\
        true_value, true_point


def data_second():
    """Входные данные для второй тестовой задачи:
        |x1 + 0.5| + |x2 + 1| + |x3 + 2|-> min

        (x1)**2 + (x2)**2 + (x3)**2 - 1 <= 0

        Решение: X* = (-0.5, -sqrt(3/2)/2, -sqrt(3/2)/2), Q(X*) = 1.78
    :return: кортеж из: параметры алгоритма, метод Нелдера-Мида, целевая функция
        кортеж ограничений, стартовая точка, ожидаемое значение и точность,
        ожидаемая точка и точность.
    """
    method = NelderMead()
    variables = sm.symbols("x1:4")
    expr = sm.Abs(variables[0]+0.5) + sm.Abs(variables[1]+1)
    expr += sm.Abs(variables[2]+2)
    func = BaseFunction(expr, variables)
    constr_func = Polynomial([[-1, 0, 1], [0, 0, 1], [0, 0, 1]])
    constraints = (Inequality(constr_func), )
    true_value = (1.78, 0.01)
    true_point = Point(-0.5, -(3**0.5/(2 * 2**0.5)), -(3**0.5/(2 * 2**0.5)))
    true_point = (true_point, 0.1)
    start_point = Point(0, 0, 0)
    params = {}
    return params, method, func, constraints, start_point,\
        true_value, true_point


def data_third():
    """Входные данные для второй тестовой задачи:
        (x1)**2 + (x2)**2 + (x3)**2 -> min

        2*x1 - x2 + x3 - 5 <= 0
        x1 + x2 + x3 - 3 = 0

        Решение: X* = (1, 1, 1), Q(X*) = 3
    :return: кортеж из: параметры алгоритма, метод Нелдера-Мида, целевая функция,
        кортеж ограничений, стартовая точка, ожидаемое значение и точность,
        ожидаемая точка и точность.
    """
    method = NelderMead()
    func = Polynomial([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    constr_first = Polynomial([[-5, 2], [0, -1], [0, 1]])
    constr_second = Polynomial([[-3, 1], [0, 1], [0, 1]])
    constraints = (Inequality(constr_first), Equality(constr_second))
    true_value = (3, 0.1)
    true_point = Point(1, 1, 1), 0.1
    start_point = Point(0, 0, 0)
    params = {"max_steps": 5}
    return params, method, func, constraints, start_point, \
        true_value, true_point


class TestConditional:
    """Класс для тестирования ConditionalNelderMead

    Методы:
        test_create(dict) - создание

        test_fit(dict, NelderMead, BaseFunction, tuple) - инициализация метода и задачи

        test_run(dict, NelderMead, BaseFunction, tuple, Point,
        tuple, tuple) - работа алгоритма
    """
    @pytest.mark.parametrize(
        "params", [
            ({"eps": 0.0, "betta": 1.5}),
            pytest.param({"test": 1.0}, marks=pytest.mark.xfail(strict=True)),
            ({}),
            pytest.param({"eps": -0.1}, marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_create(self, params: dict):
        """Тестирование создания алгоритма

        :param params: параметры в виде словаря
        """
        method = ConditionalNelderMead(**params)
        actual = method.parameters
        for key, element in params.items():
            assert actual[key] == element

    @pytest.mark.parametrize(
        ("params", "nm_method", "func", "constraints"), [
            data_first()[:4],
            data_second()[:4]
        ]
    )
    def test_fit(self, params: dict, nm_method: NelderMead,
                 func: BaseFunction, constraints: tuple):
        """Тестирование инициализации метода и задачи

        :param params: параметры алгоритма
        :param nm_method: метода Неледра-Мида
        :param func: целевая функция
        :param constraints: ограничения
        """
        method = ConditionalNelderMead(**params)
        method.fit(nm_method, func, *constraints)
        assert method.nm_method.params == nm_method.params
        assert method.function == func
        assert method.constraints == constraints

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize(
        ("params", "nm_method", "func", "constraints",
         "start_point", "true_value", "true_point"), [
            data_first(),
            data_second(),
            data_third()
        ]
    )
    def test_run(self, params: dict, nm_method: NelderMead, func: BaseFunction,
                 constraints: tuple, start_point: Point, true_value: tuple,
                 true_point: tuple):
        """Тестирование работы алгоритма

        :param params: параметры алгоритма
        :param nm_method: метод Неледра-Мида
        :param func: целевая функция
        :param constraints: ограничения
        :param start_point: стартовая точка
        :param true_value: оптимальное значение функции
        :param true_point: оптимальное решение
        """
        method = ConditionalNelderMead(**params)
        method.fit(nm_method, func, *constraints)
        result = method.run(start_point)
        assert result[0].values == pytest.approx(true_point[0].values,
                                                 abs=true_point[1])
        assert result[1] == pytest.approx(true_value[0], abs=true_value[1])
