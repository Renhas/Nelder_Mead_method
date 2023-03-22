import pytest
import sympy as sm
from neldermead import NelderMead
from functions import Polynomial, Rosenbroke, BaseFunction

x_var, y_var = sm.symbols("x,y")


class TestNelderMead:
    @pytest.mark.parametrize(
        "params", [
            ({"alpha": 10, "betta": 0.2, "gamma": 0.003}),
            pytest.param({"zeta": 10, "min_steps": 100},
                         marks=pytest.mark.xfail(strict=True)),
            ({})

        ]
    )
    def test_create(self, params: dict):
        method = NelderMead(**params)
        for key, value in params.items():
            assert method.params[key] == value

    @pytest.mark.parametrize(
        ("params", "function", "simplex", "expected"), [
            ({}, Rosenbroke(), [[10, 9], [10, -2], [21, 1]],
             [[10, 9], [10, -2], [21, 1]])

        ]
    )
    def test_fit(self, params, function, simplex, expected):
        method = NelderMead(**params)
        method.fit(function, simplex)
        assert method.function == function
        assert method.simplex == expected

    @pytest.mark.parametrize(
        ("params", "function", "simplex", "expected"), [
            ({"max_steps": 10},
             BaseFunction(expr=x_var**2 + x_var*y_var + y_var**2
                               - 6*x_var - 9*y_var, var={x_var, y_var}),
             [[0, 0], [1, 0], [0, 1]], (-20.99, 0.01)),
            ({"eps0": 0.0001}, Rosenbroke(), [[10, 9], [10, -2], [21, 1]],
             (0, 0.0005)),


        ]
    )
    def test_run(self, params, function, simplex, expected):
        method = NelderMead(**params)
        method.fit(function, simplex)
        result = method.run()
        assert result == pytest.approx(expected[0], abs=expected[1])
