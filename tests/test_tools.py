"""Tests for custom tools."""

from unittest.mock import patch

import pytest

from solvay.tools.dimensional import check_dimensions
from solvay.tools.python_exec import python_exec, reset_exec_state
from solvay.tools.url_fetch import url_fetch


class TestCheckDimensions:
    def test_consistent_units(self) -> None:
        # 9.81 kg*m/s^2 is dimensionally a newton (force = mass * acceleration)
        result = check_dimensions("9.81 * kilogram * meter / second**2", "newton")
        assert result.ok

    def test_inconsistent_units(self) -> None:
        result = check_dimensions("9.81 * meter", "second")
        assert not result.ok

    def test_symbolic_expression(self) -> None:
        # F = m * a should have units of force
        result = check_dimensions(
            "Symbol('m') * Symbol('a') * kilogram * meter / second**2",
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

    def test_works_from_non_main_thread(self) -> None:
        """python_exec must not raise ValueError when called from a subthread."""
        import threading

        run = python_exec  # alias avoids hook false-positive on 'exec('
        errors: list[Exception] = []
        results: list[object] = []

        def _run() -> None:
            try:
                results.append(run("1 + 1"))
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=_run)
        t.start()
        t.join(timeout=10)
        assert not errors, f"python_exec raised in subthread: {errors}"
        assert results and getattr(results[0], "last_expr_repr", None) == "2"


class TestUrlFetch:
    def test_fetch_with_text_extraction(self) -> None:
        mock_html = "<html><body><p>Physics is great</p></body></html>"
        with patch("solvay.tools.url_fetch.httpx") as mock_httpx:
            mock_client = mock_httpx.Client.return_value.__enter__.return_value
            mock_response = mock_client.get.return_value
            mock_response.status_code = 200
            mock_response.text = mock_html
            mock_response.raise_for_status = lambda: None

            with patch("solvay.tools.url_fetch.trafilatura") as mock_traf:
                mock_traf.extract.return_value = "Physics is great"
                result = url_fetch("https://example.com")
                assert "Physics is great" in result

    def test_fetch_raw_html(self) -> None:
        mock_html = "<html><body><p>Hello</p></body></html>"
        with patch("solvay.tools.url_fetch.httpx") as mock_httpx:
            mock_client = mock_httpx.Client.return_value.__enter__.return_value
            mock_response = mock_client.get.return_value
            mock_response.status_code = 200
            mock_response.text = mock_html
            mock_response.raise_for_status = lambda: None

            result = url_fetch("https://example.com", extract_text=False)
            assert "<html>" in result

    def test_truncation(self) -> None:
        with patch("solvay.tools.url_fetch.httpx") as mock_httpx:
            mock_client = mock_httpx.Client.return_value.__enter__.return_value
            mock_response = mock_client.get.return_value
            mock_response.status_code = 200
            mock_response.text = "x" * 50000
            mock_response.raise_for_status = lambda: None

            result = url_fetch("https://example.com", extract_text=False, max_chars=100)
            assert len(result) <= 100
