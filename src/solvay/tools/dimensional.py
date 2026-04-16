"""Dimensional analysis tool using sympy.physics.units."""

from __future__ import annotations

from solvay.schemas import DimCheckResult


def check_dimensions(expression: str, expected_unit: str) -> DimCheckResult:
    """Check dimensional consistency of a physics expression.

    Args:
        expression: A string representing a sympy expression with units
            (e.g., "9.81 * kilogram * meter / second**2").
        expected_unit: The expected unit as a sympy unit name
            (e.g., "newton", "meter", "1" for dimensionless).

    Returns:
        DimCheckResult with ok=True if dimensions match.
    """
    import sympy
    from sympy.physics.units import convert_to
    from sympy.physics.units.systems import SI

    # Build a namespace with all SI units and common symbols
    namespace: dict[str, object] = {}
    namespace.update(vars(sympy))
    namespace.update(vars(sympy.physics.units))
    namespace["Symbol"] = sympy.Symbol

    try:
        expr = sympy.sympify(expression, locals=namespace)
        target = sympy.sympify(expected_unit, locals=namespace)

        converted = convert_to(expr, target, unit_system=SI)
        simplified = sympy.simplify(converted)

        # Check if conversion succeeded (no leftover unit mismatch)
        # If convert_to cannot convert, it returns the original expression unchanged
        ratio = sympy.simplify(simplified / target) if target != 1 else simplified
        is_number = ratio.is_number if hasattr(ratio, "is_number") else False

        return DimCheckResult(
            ok=bool(is_number),
            actual_unit=str(converted),
            simplified=str(simplified),
            notes=None,
        )
    except Exception as e:
        return DimCheckResult(
            ok=False,
            actual_unit="error",
            simplified="error",
            notes=f"Dimensional check failed: {e}",
        )
