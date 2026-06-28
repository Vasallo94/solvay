"""Unit tests for the physics_checklist tool."""

from __future__ import annotations

import ast

import pytest

from solvay.tools.physics_checklist import physics_checklist

_ALL_DOMAINS = ["em", "mechanics", "quantum", "thermo", "waves", "relativity", "other"]
_GENERIC_CHECK_NAMES = {"dimensional_analysis", "energy_conservation", "limiting_cases"}


class TestPhysicsChecklist:
    def test_em_domain_returns_checks(self) -> None:
        result = physics_checklist("em", knowns=["E", "B"], unknowns=["J"])
        assert isinstance(result, dict)
        assert "checks" in result
        assert "domain_notes" in result
        checks = result["checks"]
        assert isinstance(checks, list)
        assert len(checks) > 0

    def test_em_has_boundary_condition_check(self) -> None:
        result = physics_checklist("em", knowns=["sigma"], unknowns=["J"])
        check_names = {c["name"] for c in result["checks"]}
        assert "boundary_condition_j_dot_n" in check_names

    def test_mechanics_domain_returns_checks(self) -> None:
        result = physics_checklist("mechanics", knowns=["m", "v0"], unknowns=["x(t)"])
        checks = result["checks"]
        check_names = {c["name"] for c in checks}
        # Generic checks must be present.
        assert _GENERIC_CHECK_NAMES.issubset(check_names)
        # Domain-specific checks must also be present.
        assert "momentum_conservation" in check_names
        assert "limiting_mass_cases" in check_names

    def test_quantum_domain_returns_checks(self) -> None:
        result = physics_checklist("quantum", knowns=["V(x)"], unknowns=["psi(x)"])
        check_names = {c["name"] for c in result["checks"]}
        assert "normalization" in check_names
        assert "hermiticity" in check_names
        assert "correspondence_principle" in check_names

    def test_thermo_domain_returns_checks(self) -> None:
        result = physics_checklist("thermo", knowns=["T", "P"], unknowns=["S"])
        check_names = {c["name"] for c in result["checks"]}
        assert "second_law" in check_names
        assert "temperature_limits" in check_names

    def test_unknown_domain_returns_generic_fallback(self) -> None:
        result = physics_checklist("astrology", knowns=[], unknowns=["fate"])
        checks = result["checks"]
        check_names = {c["name"] for c in checks}
        # Only the three generic checks should be present.
        assert check_names == _GENERIC_CHECK_NAMES
        assert "domain_notes" in result
        assert "astrology" in result["domain_notes"]

    def test_check_fields_are_complete(self) -> None:
        result = physics_checklist("em", knowns=["E"], unknowns=["J"])
        for check in result["checks"]:
            assert "name" in check, f"Missing 'name' in check: {check}"
            assert "description" in check, f"Missing 'description' in check: {check}"
            assert "python_code" in check, f"Missing 'python_code' in check: {check}"
            assert isinstance(check["name"], str) and check["name"]
            assert isinstance(check["description"], str) and check["description"]
            assert isinstance(check["python_code"], str) and check["python_code"]

    def test_python_code_is_syntactically_valid(self) -> None:
        result = physics_checklist("em", knowns=["E"], unknowns=["J"])
        for check in result["checks"]:
            code = check["python_code"]
            try:
                ast.parse(code)
            except SyntaxError as exc:
                pytest.fail(f"Check '{check['name']}' has invalid Python code: {exc}\n\n{code}")

    def test_all_domains_produce_dimensional_check(self) -> None:
        for domain in _ALL_DOMAINS:
            result = physics_checklist(domain, knowns=[], unknowns=[])
            check_names = {c["name"] for c in result["checks"]}
            assert "dimensional_analysis" in check_names, (
                f"Domain '{domain}' is missing the 'dimensional_analysis' check"
            )
