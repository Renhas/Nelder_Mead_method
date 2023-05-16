"""
Сборник математических функций на основе :mod:`sympy`
"""
import numbers
from typing import Union, Sequence
import sympy as sm
from utilities.point import Point


class BaseFunction:
    """Базовый класс, предоставляющий основной функционал
     для любых математических функций
    """
    def __init__(self, expr: sm.Expr, var: Sequence[sm.Symbol]):
        """Инициализатор класса

        Args:
            expr:символьное выражение
            var: кортеж переменных
        """
        self.__expression = expr
        self.__variables = var
        self.__dimension = len(var)
        self.__check_args()

    @property
    def expr(self) -> sm.Expr:
        """Символьное выражение"""
        return self.__expression

    @property
    def dimension(self) -> int:
        """Размерность пространства переменных"""
        return self.__dimension

    @property
    def variables(self) -> Sequence[sm.Symbol]:
        """Символьные переменные"""
        return self.__variables

    def __check_args(self):
        """Проверка входных параметров

        Raises:
            AttributeError - если параметры не корректны
        """
        expr = self.__expression
        var = self.__variables
        if not isinstance(expr, sm.Expr):
            raise AttributeError("expr should be an sympy.Expr")
        if len(var) == 0 and len(expr.atoms(sm.Symbol)) != 0:
            raise AttributeError("In var should be at least one element")
        if not isinstance(var, tuple):
            raise AttributeError("var should be a tuple")
        for variable in var:
            if variable not in expr.atoms(sm.Symbol):
                raise AttributeError("var should contain"
                                     " sympy.Symbol from expr")

    def calculate(self, data: Union[list, Point]) -> Union[float, sm.Expr]:
        """Подстановка чисел или символов вместо переменных,
        а также последующее вычисление или упрощение выражения

        Args:
            data: список значений или одно значение

        Returns:
            Число или символьное выражение с учётом подстановок

        Raises:
            AttributeError - если передано данных меньше, чем размерность функции
        """
        if len(data) != self.__dimension:
            raise AttributeError("Data length must be the same as the dimension")
        if isinstance(data, Point):
            data = data.values
        return self.__expression.subs(zip(self.__variables, data))

    def __str__(self) -> str:
        """Строковое представление выражения"""
        return str(self.__expression)

    def __add__(self, other: Union["BaseFunction", numbers.Number]) \
            -> "BaseFunction":
        """Сложение двух функций или функции и числа"""
        new_expr = self.__expression
        new_var = list(self.__variables)
        if isinstance(other, numbers.Number):
            new_expr += other
        elif isinstance(other, BaseFunction):
            new_expr += other.expr
            for var in other.variables:
                if var not in new_var:
                    new_var.append(var)
        else:
            raise AttributeError(f"can't add {type(other)} to {type(self)}")
        return BaseFunction(new_expr, tuple(new_var))

    def __radd__(self, other: Union["BaseFunction", numbers.Number]) \
            -> "BaseFunction":
        """Сложение справа"""
        return self.__add__(other)

    def __mul__(self, other: Union["BaseFunction", numbers.Number]) \
            -> "BaseFunction":
        """Умножение функции на число или функцию"""
        new_expr = self.__expression
        new_var = list(self.__variables)
        if isinstance(other, numbers.Number):
            new_expr *= other
        elif isinstance(other, BaseFunction):
            new_expr *= other.expr
            for var in other.variables:
                if var not in new_var:
                    new_var.append(var)
        else:
            raise AttributeError(f"can't add {type(other)} to {type(self)}")
        return BaseFunction(new_expr, tuple(new_var))

    def __rmul__(self, other: Union["BaseFunction", numbers.Number]) \
            -> "BaseFunction":
        """Умножение функции справа на число или функцию"""
        return self.__mul__(other)

    def __eq__(self, other: "BaseFunction") -> bool:
        """Проверка на равенство двух функций"""
        if not isinstance(other, BaseFunction):
            return False
        if len(other.variables) != len(self.__variables):
            return False
        expr = other.expr == self.__expression
        var = other.variables == self.__variables
        return expr and var

    def __ne__(self, other: "BaseFunction") -> bool:
        """Проверка на не равенство"""
        return not self.__eq__(other)


class Polynomial(BaseFunction):
    """
    Класс, позволяющий создать произвольный n-мерный полином"""
    def __init__(self, coefficients: Sequence[Sequence[float]]):
        """Инициализатор класса. Создаёт переменные вида :math:`x_1, x_2, x_3\\dots`

        Args:
            coefficients: Коэффициенты в порядке возрастания степеней переменной.
        """
        variables = sm.symbols(f"x1:{len(coefficients)+1}")
        expr = 0 * variables[0]
        for var_index, variable in enumerate(coefficients):
            for val_index, value in enumerate(variable):
                expr += value * variables[var_index] ** val_index
        super().__init__(expr, variables)


class Rosenbroke(BaseFunction):
    """Класс реализующий двумерную функцию Розенброка"""
    def __init__(self):
        """Инициализатор класса"""
        variables = sm.symbols("x, y")
        expr = (1 - variables[0])**2 + 100*(variables[1] - variables[0]**2)**2
        super().__init__(expr, variables)


class Himmelblau(BaseFunction):
    """Класс, реализующий функцию Химмельблау"""
    def __init__(self):
        """Инициализатор класса"""
        x_var, y_var = sm.symbols("x, y")
        expr = (x_var**2 + y_var - 11) ** 2 + (x_var + y_var**2 - 7) ** 2
        super().__init__(expr, (x_var, y_var))
