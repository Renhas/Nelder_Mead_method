import pytest
from nelder_mead import Nelder_Mead
from functions import Polynomial, Rosenbroke


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
        method = Nelder_Mead(**params)
        for key, value in params.items():
            assert method.params[key] == value

    @pytest.mark.parametrize(
        ("params", "function", "simplex", "expected"), [
            ({}, Rosenbroke(), [[10, 9], [10, -2], [21, 1]],
             (0.6*10**(-6), 0.1))

        ]
    )
    def test_run(self, params, function, simplex, expected):
        method = Nelder_Mead()
        assert method.Run(function, simplex) == pytest.approx(expected[0], abs=expected[1])
