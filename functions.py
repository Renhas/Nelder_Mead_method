"""
Сборник математических функций

Классы:
    BaseFunction - базовый класс для произвольной функции\n
    Polynomial - класс для произвольного многомерного полинома\n
    Rosenbroke - класс с двумерной функцией Розенброка\n
    Himmelblau - класс с функцией Химмельблау
"""
from typing import Union
import sympy as sm



class BaseFunction:
    """
    Базовый класс, предоставляющий основной функционал для любых математических функций

    Методы:
        calculate(list) -> float | sympy.Expr
    Свойства:
        expr - символьное выражение в виде sympy.Expr\n
        dimension - размерность пространства переменных
        variables - кортеж переменных из выражения
    """
    def __init__(self, expr: sm.Expr, var: tuple):
        """
        Инициализатор класса

        :param expr: символьное выражение в виде sympy.Expr
        :param var: кортеж переменных типа sympy.symbols
        """
        self.__expression = expr
        self.__variables = var
        self.__dimension = len(var)

    def __str__(self) -> str:
        """
        Строковое представление выражения

        :return: строка с выражением
        """
        return str(self.__expression)

    def calculate(self, data: list) -> Union[float, sm.Expr]:
        """
        Подстановка чисел или символов вместо переменных,
        а также последующее вычисление или упрощение выражения

        :param data: список значений

        :return: число или символьное выражение с учётом подстановок
        """
        if len(data) != self.__dimension:
            raise AttributeError("Data length must be the same as the dimension")
        return self.__expression.subs(zip(self.__variables, data))

    @property
    def expr(self) -> sm.Expr:
        """
        Возвращает символьное выражение

        :return: символьное выражение
        """
        return self.__expression

    @property
    def dimension(self) -> int:
        """
        Возвращает размерность пространства переменных

        :return: целое число
        """
        return self.__dimension

    @property
    def variables(self) -> tuple:
        """
        Кортеж символьных переменных в виде sympy.symbols

        :return: кортеж символов
        """
        return self.__variables


class Polynomial(BaseFunction):
    """
    Класс, позволяющий создать произвольный n-мерный полином.

    Наследует от BaseFunction весь функционал с переопределением инициализатора
    """
    def __init__(self, coefficients: list):
        """Инициализатор класса. Создаёт переменные вида x1, x2, x3...

        :param coefficients: В общем случае - двумерный список коэффициентов.
            Каждый внутренний список соответствует одной переменной,
            а конкретный список соответствует коэффициентам по степеням
            переменной в порядке возрастания
        """
        variables = sm.symbols(f"x1:{len(coefficients)+1}")
        expr = 0 * variables[0]
        for var_index, variable in enumerate(coefficients):
            for val_index, value in enumerate(variable):
                expr += value * variables[var_index] ** val_index
        super().__init__(expr, variables)


class Rosenbroke(BaseFunction):
    """Класс реализующий двумерную функцию Розенброка.
    Наследует функционал BaseFunction и переопределяет инициализатор
    """
    def __init__(self):
        """Инициализатор, создающий функцию Розенброка"""
        variables = sm.symbols("x, y")
        expr = (1 - variables[0])**2 + 100*(variables[1] - variables[0]**2)**2
        super().__init__(expr, variables)


class Himmelblau(BaseFunction):
    """Класс, реализующий функцию Химмельблау.
    Наследует функционал BaseFunction и переопределяет инициализатор
    """
    def __init__(self):
        """Инициализатор, создающий функцию Химмельблау"""
        x_var, y_var = sm.symbols("x, y")
        expr = (x_var**2 + y_var - 11) ** 2 + (x_var + y_var**2 - 7) ** 2
        super().__init__(expr, (x_var, y_var))
