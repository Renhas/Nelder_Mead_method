"""
Тестовый модуль для метода Нелдера-Мида

Классы:
    TestNelderMead - класс для тестирования метода
"""
import pytest
import sympy as sm
from bin.neldermead import NelderMead
from bin.functions import Rosenbroke, BaseFunction
from bin.functions import Himmelblau

# Символы для символьных выражений
x_var, y_var = sm.symbols("x,y")


class TestNelderMead:
    """Класс для тестирования метода Нелдера-Мида

    Методы:
        test_create(dict) - тестирование корректного создания экземпляра класса\n
        test_fit(dict, BaseFunction, list, list) - тестирование инициализации
        функции и начального симплекса\n
        test_run(dict, BaseFunction, list, tuple) - тестирование работы метода
        test_answer(dict, BaseFunction, list, tuple) - тестирование решения
    """
    @pytest.mark.parametrize(
        "params", [
            ({"alpha": 10, "betta": 0.2, "gamma": 0.003}),
            pytest.param({"zeta": 10, "min_steps": 100},
                         marks=pytest.mark.xfail(strict=True)),
            ({})

        ]
    )
    def test_create(self, params: dict):
        """Тестирование инициализатора экземпляра класса.

        :param params: словарь параметров
        """
        method = NelderMead(**params)
        for key, value in params.items():
            assert method.params[key] == value

    @pytest.mark.parametrize(
        ("params", "function", "simplex", "expected"), [
            ({}, Rosenbroke(), [[10, 9], [10, -2], [21, 1]],
             [[10, 9], [10, -2], [21, 1]])

        ]
    )
    def test_fit(self, params: dict, function: BaseFunction, simplex: list,
                 expected: list):
        """Тестирование функции инициализации оптимизируемой функции
         и начального симплекса

        :param params: словарь параметров
        :param function: оптимизируемая функция
        :param simplex: начальный симплекс
        :param expected: ожидаемый симплекс
        """
        method = NelderMead(**params)
        method.fit(function, simplex)
        assert method.function == function
        assert method.simplex == expected

    @pytest.mark.parametrize(
        ("params", "function", "simplex", "expected"), [
            ({"max_steps": 10},
             BaseFunction(expr=x_var**2 + x_var*y_var + y_var**2
                               - 6*x_var - 9*y_var, var=(x_var, y_var)),
             [[0, 0], [1, 0], [0, 1]], (-20.99, 0.01)),
            ({"eps0": 0.0001}, Rosenbroke(), [[10, 9], [10, -2], [21, 1]],
             (0, 0.0005)),
            ({}, Himmelblau(), [[-1.5, 0.5], [-4, 2.5], [-4.5, 5]], (0, 0.01))


        ]
    )
    def test_run(self, params: dict, function: BaseFunction, simplex: list,
                 expected: tuple):
        """Тестирование работы метода

        :param params: словарь параметров модели
        :param function: оптимизируемая функция
        :param simplex: начальный симплекс
        :param expected: кортеж из ожидаемого значения и точности сравнения
        """
        method = NelderMead(**params)
        method.fit(function, simplex)
        result = method.run()
        assert result == pytest.approx(expected[0], abs=expected[1])

    @pytest.mark.parametrize(
        ("params", "function", "simplex", "expected"), [
            ({"max_steps": 10},
             BaseFunction(expr=x_var ** 2 + x_var * y_var + y_var ** 2
                               - 6 * x_var - 9 * y_var, var=(x_var, y_var)),
             [[0, 0], [1, 0], [0, 1]], ([1, 4], 0.1)),
            ({"eps0": 0.0001}, Rosenbroke(), [[10, 9], [10, -2], [21, 1]],
             ([1, 1], 0.01)),
            ({}, Himmelblau(), [[-1.5, 0.5], [-4, 2.5], [-4.5, 5]],
             ([-2.805, 3.131], 0.001))

        ]
    )
    def test_answer(self, params: dict, function: BaseFunction, simplex: list,
                    expected: tuple):
        """Тестирование правильности найденного решения

        :param params: словарь параметров
        :param function: оптимизируемая функция
        :param simplex: начальный симплекс
        :param expected: кортеж из ожидаемого симплекса и точности сравнения
        """
        method = NelderMead(**params)
        method.fit(function, simplex)
        method.run()
        result = method.simplex[0]
        assert result == pytest.approx(expected[0], abs=expected[1])
