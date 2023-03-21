import sympy as sm


class BaseFunction:

    def __init__(self, expr: sm.Expr, var: set):
        self.__expression = expr
        self.__variables = var
        self.__dimension = len(var)

    def __str__(self):
        return str(self.__expression)

    def calculate(self, data: list):
        if len(data) != self.__dimension:
            raise AttributeError("Data length must be the same as the dimension")
        return self.__expression.subs(zip(self.__variables, data))

    @property
    def expr(self):
        return self.__expression

    @property
    def dimension(self):
        return self.__dimension

    @property
    def variables(self):
        return self.__variables
