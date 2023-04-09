"""
Тестовый модуль для метода Нелдера-Мида

Классы:
    TestNelderMead - класс для тестирования метода
    TestSimplex - класс для тестирования Simplex
"""
import pytest
import sympy as sm
from scripts.nelder_mead import NelderMead, Simplex
from scripts.point import Point
from scripts.functions import Rosenbroke, BaseFunction
from scripts.functions import Himmelblau

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
            ({}),
            pytest.param({"alpha": -1, "betta": -15},
                         marks=pytest.mark.xfail(strict=True)),
            ({"alpha": 0}),
            pytest.param({"max_steps": 1.5},
                         marks=pytest.mark.xfail(strict=True)),
            pytest.param({"alpha": "23"},
                         marks=pytest.mark.xfail(strict=True)),
            pytest.param({"eps0": -100},
                         marks=pytest.mark.xfail(strict=True))

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
        ("params", "function", "simplex"), [
            ({}, Rosenbroke(), [Point(10, 9), Point(10, -2), Point(21, 1)])


        ]
    )
    def test_fit(self, params: dict, function: BaseFunction, simplex: list):
        """Тестирование функции инициализации оптимизируемой функции
         и начального симплекса

        :param params: словарь параметров
        :param function: оптимизируемая функция
        :param simplex: начальный симплекс
        """
        method = NelderMead(**params)
        sim = Simplex(function, *simplex)
        method.fit(function, *simplex)
        assert method.function == function
        assert method.simplex == sim

    @pytest.mark.parametrize(
        ("params", "function", "simplex", "expected"), [
            ({"max_steps": 10},
             BaseFunction(expr=x_var**2 + x_var*y_var + y_var**2
                               - 6*x_var - 9*y_var, var=(x_var, y_var)),
             [Point(0, 0), Point(1, 0), Point(0, 1)], (-20.99, 0.01)),
            ({"eps0": 0.0001}, Rosenbroke(),
             [Point(10, 9), Point(10, -2), Point(21, 1)], (0, 0.0005)),
            ({}, Himmelblau(),
             [Point(-1.5, 0.5), Point(-4, 2.5), Point(-4.5, 5)], (0, 0.01))


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
        method.fit(function, *simplex)
        result = method.run()
        assert result == pytest.approx(expected[0], abs=expected[1])

    @pytest.mark.parametrize(
        ("params", "function", "simplex", "expected"), [
            ({"max_steps": 10},
             BaseFunction(expr=x_var ** 2 + x_var * y_var + y_var ** 2
                               - 6 * x_var - 9 * y_var, var=(x_var, y_var)),
             [Point(0, 0), Point(1, 0), Point(0, 1)], (Point(1, 4), 1)),
            ({"eps0": 0.0001}, Rosenbroke(), [Point(10, 9), Point(10, -2), Point(21, 1)],
             (Point(1, 1), 1)),
            ({}, Himmelblau(),
             [Point(-1.5, 0.5), Point(-4, 2.5), Point(-4.5, 5)],
             (Point(-2.805, 3.131), 0.001))

        ]
    )
    def test_answer(self, params: dict, function: BaseFunction, simplex: list,
                    expected: tuple):
        """Тестирование правильности найденного решения

        :param params: словарь параметров
        :param function: оптимизируемая функция
        :param simplex: начальный симплекс
        :param expected: кортеж из ожидаемой точки и точности сравнения
        """
        method = NelderMead(**params)
        method.fit(function, *simplex)
        method.run()
        result = method.simplex.best[0].values
        assert result == pytest.approx(expected[0].values, abs=expected[1])


class TestSimplex:
    """Класс для тестирования функционала Simplex

    Метода:
        test_create(BaseFunction, list, tuple)

        test_create_from_one()

        test_three_points(BaseFunction, list, tuple)

        test_replace(BaseFunction, list, int, Point, tuple)
    """
    @pytest.mark.parametrize(
        ("function", "points", "expected"), [
            (Rosenbroke(), [Point(0, 0), Point(1, 1), Point(0, 1)],
             ((Point(1, 1), 0), (Point(0, 0), 1), (Point(0, 1), 101))),
            pytest.param(Rosenbroke(), [Point(0, 0), Point(1, 1), Point(0, 1)],
                         ((Point(0, 0), 1), (Point(1, 1), 0),
                         (Point(0, 1), 101)),
                         marks=pytest.mark.xfail(strict=True)),
            (Rosenbroke(),
             [(Point(0, 0), 1), (Point(1, 1), 0), (Point(0, 1), 101)],
             ((Point(1, 1), 0), (Point(0, 0), 1), (Point(0, 1), 101))),
        ]
    )
    def test_create(self, function: BaseFunction, points: list,
                    expected: tuple):
        """Тестирование создания симплекса

        :param function: символьная функция
        :param points: список точек
        :param expected: ожидаемый кортеж точек вместе со значением функции
        """
        sim = Simplex(function, *points)
        sim = sim.sort()
        assert sim.function == function
        assert sim.points == expected

    def test_create_from_one(self):
        """Тестирование создания симплекса из одной точки"""
        function = BaseFunction(x_var * 2, (x_var, ))
        point = Point(0)
        sim = Simplex(function, point)
        sim = sim.sort()
        assert sim.points == ((Point(0), 0), (Point(1), 2))

    @pytest.mark.parametrize(
        ("function", "points", "three_points"), [
            (Rosenbroke(), [(Point(0, 0), 1), (Point(1, 1), 0), (Point(0, 1), 101)],
             ((Point(1, 1), 0), (Point(0, 0), 1), (Point(0, 1), 101)))
        ]
    )
    def test_three_points(self, function: BaseFunction, points: list,
                          three_points: tuple):
        """Тестирование точек best, good, worst

        :param function: символьная функция
        :param points: список точек
        :param three_points: кортеж с best, good, worst
        """
        sim = Simplex(function, *points)
        sim = sim.sort()
        assert sim.best == three_points[0]
        assert sim.good == three_points[-2]
        assert sim.worst == three_points[-1]

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize(
        ("function", "points", "index", "new_point", "expected"), [
            (Rosenbroke(), [(Point(0, 0), 1), (Point(1, 1), 0), (Point(0, 1), 101)],
             2, Point(0.5, 0.5**2),
             ((Point(1, 1), 0), (Point(0.5, 0.5**2), 0.25), (Point(0, 0), 1)))
        ]
    )
    def test_replace(self, function: BaseFunction, points: list,
                     index: int, new_point: Point, expected: tuple):
        """Тестирование замещения точки

        :param function: символьная функция
        :param points: список точек
        :param index: индекс заменяемой точки
        :param new_point: новая точка
        :param expected: ожидаемый кортеж точек вместе со значением функции
        """
        sim = Simplex(function, *points)
        sim = sim.replace(index, new_point)
        sim = sim.sort()
        assert sim.points == expected
