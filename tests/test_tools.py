"""Tests for custom tools."""

from solvay.tools.dimensional import check_dimensions


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
