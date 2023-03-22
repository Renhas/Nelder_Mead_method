import pytest
import sympy as sm
from functions import BaseFunction, Polynomial, Rosenbroke


variables = sm.symbols("x1:10")
x_var, y_var = sm.symbols("x,y")


class TestBaseFunction:
    @pytest.mark.parametrize(
        ("expression", "expr_var"), [
            (sm.sin(variables[0] + sm.cos(variables[1])), {variables[0], variables[1]}),

        ]
    )
    def test_create(self, expression, expr_var):
        func = BaseFunction(expression, expr_var)
        assert func.expr == expression
        assert func.dimension == len(expr_var)
        assert func.variables == expr_var

    @pytest.mark.parametrize(
        ("expression", "expr_var",  "test_input", "test_output"), [
            (2.0*sm.ln(variables[0]), {variables[0]}, [sm.E], 2.0),
            (x_var**2 + x_var*y_var + y_var**2 - 6*x_var - 9*y_var,
             {x_var, y_var}, [1, 0], -5)
        ]

    )
    def test_calculate(self, expression, expr_var, test_input, test_output):
        func = BaseFunction(expression, expr_var)
        assert func.calculate(test_input) == test_output

    @pytest.mark.parametrize(
        ("expression", "expr_var", "view"), [
            (2.0*sm.ln(variables[0]), {variables[0]}, "2.0*log(x1)")
        ]
    )
    def test_str(self, expression, expr_var, view):
        func = BaseFunction(expression, expr_var)
        assert str(func) == view


class TestPolynomial:
    @pytest.mark.parametrize(
        ("coefficients", "view", "variable_count"), [
            ([[0, 2, 3], [0, 3, 4]],
             2 * variables[0] + 3 * variables[0] ** 2 +
             3 * variables[1] + 4 * variables[1] ** 2,
             2)
        ]
    )
    def test_create(self, coefficients, view, variable_count):
        poly = Polynomial(coefficients)
        assert poly.expr == view
        assert poly.dimension == variable_count

    @pytest.mark.parametrize(
        ("coefficients", "test_input", "test_output"), [
            ([[0, 2, 3], [0, 3, 4]], [0, 0], 0.0),
            ([[0, 2, 3], [0, 3, 4]], [1, 2], 27.0),
            ([[0, 1], [0, 2, 1]], [1, 1], 4.0)
        ]
    )
    def test_calculate(self, coefficients, test_input, test_output):
        poly = Polynomial(coefficients)
        assert poly.calculate(test_input) == test_output


class TestRosenbroke:

    def test_create(self):
        x_var, y_var = sm.symbols("x, y")
        assert Rosenbroke().expr == (1 - x_var)**2 + 100*(y_var - x_var**2)**2
        assert Rosenbroke().dimension == 2

    @pytest.mark.parametrize(
        ("test_input", "test_output"), [
            ([1, 1], 0)
        ]
    )
    def test_calculate(self, test_input, test_output):
        assert Rosenbroke().calculate(test_input) == test_output

