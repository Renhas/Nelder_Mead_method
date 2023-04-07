import pytest
from scripts.functions import Polynomial

class TestSimplex:
    @pytest.mark.parametrize(
        ("points", "function", "expected"), [
            #([], Polynomial([[1, 1]]), )
        ]
    )
    def test_init(self, points, function, expected):
        pass
