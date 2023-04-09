"""
Модуль для тестирования функционала point.py

Классы:
    TestPoint - тестирование класса Point
"""
import pytest
from scripts.point import Point


class TestPoint:
    """Класс для тестирования Point

    Методы:
        test_create(list, tuple)

        test_zero(int, tuple)

        test_ones(int, tuple)

        test_unit(int, int, tuple)

        test_distance(Point, Point, float, float)

        test_len()

        test_eq()

        test_add()

        test_sub()

        test_mul()

        test_truediv()

        test_pow()

        test_str()
    """
    @pytest.mark.parametrize(
        ("values", "expected"), [
            ([3, 6, 7, 8], (3, 6, 7, 8)),
            pytest.param([], tuple(), marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_create(self, values: list, expected: tuple):
        """Тестирование создания точки

        :param values: список координат
        :param expected: ожидаемые координаты
        """
        point = Point(*values)
        assert point.values == expected

    @pytest.mark.parametrize(
        ("dimension", "expected"), [
            (3, (0, 0, 0)),
            pytest.param(0, (), marks=pytest.mark.xfail(strict=True)),
            pytest.param(-1, (), marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_zero(self, dimension: int, expected: tuple):
        """ Тестирование создания нулевой точки

        :param dimension: размерность
        :param expected: ожидаемые координаты
        """
        assert Point.zero(dimension).values == expected

    @pytest.mark.parametrize(
        ("dimension", "expected"), [
            (2, (1, 1)),
            pytest.param(0, (), marks=pytest.mark.xfail(strict=True)),
            pytest.param(-1, (), marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_ones(self, dimension: int, expected: tuple):
        """Тестирование создания единичной точки

        :param dimension: размерность
        :param expected: ожидаемые координаты
        """
        assert Point.ones(dimension).values == expected

    @pytest.mark.parametrize(
        ("dimension", "axis", "expected"), [
            (2, 1, (0, 1)),
            pytest.param(0, 1, (), marks=pytest.mark.xfail(strict=True)),
            pytest.param(-1, 1, (), marks=pytest.mark.xfail(strict=True)),
            pytest.param(2, -1, (), marks=pytest.mark.xfail(strict=True)),
            pytest.param(2, 2, (), marks=pytest.mark.xfail(strict=True)),
        ]
    )
    def test_unit(self, dimension: int, axis: int, expected: tuple):
        """Тестирование создания единичного орта

        :param dimension: размерность
        :param axis: единичная координата
        :param expected: ожидаемые координаты
        """
        assert Point.unit(dimension, axis).values == expected

    @pytest.mark.parametrize(
        ("first", "second", "expected", "precision"), [
            (Point(0, 0), Point(3, 4), 5, 1),
            (Point(1, 2), Point(1, 3), 1, 1),
            (Point(-2, -3), Point(-5, -3), 3, 1),
            (Point(0, 0), Point(0, 0), 0, 1),
            pytest.param(Point(2, 3), Point(3, 1, 1), 0, 0,
                         marks=pytest.mark.xfail(strict=True)),
            pytest.param(Point(3, 2), 56, 0, 0,
                         marks=pytest.mark.xfail(strict=True))
        ]
    )
    def test_distance(self, first: Point, second: Point,
                      expected: float, precision: float):
        """Тестирование вычисления расстояния

        :param first: первая точка
        :param second: вторая точка
        :param expected: ожидаемое значение
        :param precision: точность сравнения
        """
        assert first.distance(second) == pytest.approx(expected, abs=precision)

    def test_len(self):
        """Тестирование вычисления размерности точки"""
        assert len(Point(3, 1, 2)) == 3

    def test_eq(self):
        """Тестирование сравнений"""
        assert Point(1, 3) == Point(1, 3)
        assert Point(1, 3) != Point(3, 1)

    def test_add(self):
        """Тестирование сложения"""
        assert Point(3, 2) + Point(1, 3) == Point(4, 5)

    def test_mul(self):
        """Тестирование умножения"""
        assert Point(2, 3) * 3 == Point(6, 9)
        assert Point(2, 3) * Point(1, 1) == 5

    def test_truediv(self):
        """Тестирование деления"""
        assert Point(4, 4) / 2 == Point(2, 2)

    def test_pow(self):
        """Тестирование возведения в степень"""
        assert Point(2, 3) ** 2 == 13
        assert Point(2, 3) ** 3 == Point(26, 39)

    def test_str(self):
        """Тестирование строкового представления"""
        assert str(Point(2, 3)) == "(2, 3)"
