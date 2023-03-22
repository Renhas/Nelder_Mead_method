import sympy as sm


class BaseFunction:

    def __init__(self, expr: sm.Expr, var: tuple):
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


class Polynomial(BaseFunction):
    def __init__(self, coefficients: list):
        variables = sm.symbols(f"x1:{len(coefficients)+1}")
        expr = 0 * variables[0]
        for var_index, variable in enumerate(coefficients):
            for val_index, value in enumerate(variable):
                expr += value * variables[var_index] ** val_index
        super().__init__(expr, variables)


class Rosenbroke(BaseFunction):
    def __init__(self):
        variables = sm.symbols("x, y")
        expr = (1 - variables[0])**2 + 100*(variables[1] - variables[0]**2)**2
        super().__init__(expr, variables)


class Himmelblau(BaseFunction):
    def __init__(self):
        x_var, y_var = sm.symbols("x, y")
        expr = (x_var**2 + y_var - 11) ** 2 + (x_var + y_var**2 - 7) ** 2
        super().__init__(expr, (x_var, y_var))
