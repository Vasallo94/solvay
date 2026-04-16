"""Tests for custom tools."""

import pytest

from solvay.tools.dimensional import check_dimensions
from solvay.tools.python_exec import python_exec, reset_exec_state


class TestCheckDimensions:
    def test_consistent_units(self) -> None:
        # 9.81 kg*m/s^2 is dimensionally a newton (force = mass * acceleration)
        result = check_dimensions("9.81 * kilogram * meter / second**2", "newton")
        assert result.ok

    def test_inconsistent_units(self) -> None:
        result = check_dimensions("9.81 * meter", "second")
        assert not result.ok

    def test_symbolic_expression(self) -> None:
        # 1 kg*m/s^2 is exactly 1 newton — pure unit expression, no free symbols
        result = check_dimensions(
            "1 * kilogram * meter / second**2",
            "newton",
        )
        assert result.ok

    def test_dimensionless(self) -> None:
        result = check_dimensions("3.14", "1")
        assert result.ok


class TestPythonExec:
    @pytest.fixture(autouse=True)
    def _reset_state(self) -> None:
        reset_exec_state()

    def test_simple_expression(self) -> None:
        result = python_exec("2 + 2")
        assert result.last_expr_repr == "4"
        assert result.stderr == ""

    def test_print_output(self) -> None:
        result = python_exec("print('hello')")
        assert "hello" in result.stdout

    def test_sympy_available(self) -> None:
        result = python_exec("import sympy; print(sympy.sqrt(4))")
        assert "2" in result.stdout

    def test_numpy_available(self) -> None:
        result = python_exec("import numpy as np; print(np.array([1,2,3]).sum())")
        assert "6" in result.stdout

    def test_timeout_enforcement(self) -> None:
        result = python_exec(
            "import time; time.sleep(60)",
            timeout_seconds=2,
        )
        assert result.stderr != ""  # should contain timeout error

    def test_state_persistence_across_calls(self) -> None:
        r1 = python_exec("x = 42")
        assert r1.stderr == ""
        r2 = python_exec("print(x)")
        assert "42" in r2.stdout

    def test_error_handling(self) -> None:
        result = python_exec("1 / 0")
        assert "ZeroDivisionError" in result.stderr
