"""Physics checklist tool: generates domain-specific verification checks."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Domain notes: common pitfalls per physics subdomain
# ---------------------------------------------------------------------------

_DOMAIN_NOTES: dict[str, str] = {
    "em": (
        "Electromagnetism problems often fail due to missing boundary conditions on "
        "normal and tangential field components. Always verify current continuity "
        "(div J = 0 in steady state) and check energy balance via the Poynting vector."
    ),
    "mechanics": (
        "Newtonian mechanics errors frequently arise from missing internal forces in "
        "momentum sums or sign errors in torque calculations. Verify limiting mass "
        "cases (m -> 0 and m -> inf) to catch unphysical divergences."
    ),
    "quantum": (
        "Quantum mechanics solutions must be normalised; unnormalised states lead to "
        "probability violations. Operators representing observables must be Hermitian "
        "and should reproduce classical results in the large-quantum-number limit "
        "(correspondence principle)."
    ),
    "thermo": (
        "Thermodynamics solutions must respect the second law: entropy never decreases "
        "in an isolated system. Check temperature limits -- expressions should remain "
        "physical as T -> 0 and T -> inf."
    ),
    "waves": (
        "Wave problems require the dispersion relation to be consistent with the wave "
        "equation being solved. Energy flux (Poynting vector or acoustic intensity) "
        "must be conserved across any interface."
    ),
    "relativity": (
        "Relativistic calculations must yield Lorentz-invariant scalars for physical "
        "observables. Always verify that the non-relativistic limit (v << c) recovers "
        "the expected Newtonian expression."
    ),
}

# ---------------------------------------------------------------------------
# Generic checks -- always included regardless of domain
# ---------------------------------------------------------------------------

_GENERIC_CHECKS: list[dict[str, str]] = [
    {
        "name": "dimensional_analysis",
        "description": (
            "Verify that every term in the final expression carries consistent "
            "physical dimensions."
        ),
        "python_code": (
            "import sympy\n"
            "from sympy.physics.units import (\n"
            "    meter, kilogram, second, ampere, kelvin, convert_to\n"
            ")\n"
            "from sympy.physics.units.systems import SI\n"
            "\n"
            "# TODO: replace 'expr' with the symbolic expression to check\n"
            "# and 'expected_unit' with the target unit.\n"
            "expr = sympy.sympify('1')  # placeholder\n"
            "expected_unit = meter      # placeholder\n"
            "\n"
            "converted = convert_to(expr, expected_unit, unit_system=SI)\n"
            "ratio = sympy.simplify(converted / expected_unit)\n"
            "assert ratio.is_number, f'Dimensional mismatch: {converted}'\n"
        ),
    },
    {
        "name": "energy_conservation",
        "description": (
            "Confirm that the total energy (or Hamiltonian) is conserved over the "
            "time evolution described in the solution."
        ),
        "python_code": (
            "import sympy\n"
            "\n"
            "t = sympy.Symbol('t')\n"
            "\n"
            "# TODO: replace E_total with the energy expression as a function of t.\n"
            "E_total = sympy.sympify('1')  # placeholder -- must be a sympy expr in t\n"
            "\n"
            "dE_dt = sympy.diff(E_total, t)\n"
            "assert sympy.simplify(dE_dt) == 0, (\n"
            "    f'Energy is not conserved: dE/dt = {dE_dt}'\n"
            ")\n"
        ),
    },
    {
        "name": "limiting_cases",
        "description": (
            "Check that the solution reproduces known analytic results in well-understood "
            "limiting cases (e.g., massless limit, large separation, zero frequency)."
        ),
        "python_code": (
            "import sympy\n"
            "\n"
            "x = sympy.Symbol('x', positive=True)\n"
            "\n"
            "# TODO: replace 'result_expr' with the symbolic result to test,\n"
            "# 'limit_var' with the variable to take the limit in,\n"
            "# and 'limit_val' with the value to approach.\n"
            "result_expr = sympy.sympify('x')  # placeholder\n"
            "limit_var = x                      # placeholder\n"
            "limit_val = 0                      # placeholder\n"
            "expected_limit = sympy.sympify('0')  # placeholder\n"
            "\n"
            "actual_limit = sympy.limit(result_expr, limit_var, limit_val)\n"
            "assert sympy.simplify(actual_limit - expected_limit) == 0, (\n"
            "    f'Limiting case failed: limit = {actual_limit}, '\n"
            "    f'expected {expected_limit}'\n"
            ")\n"
        ),
    },
]

# ---------------------------------------------------------------------------
# Domain-specific checks
# ---------------------------------------------------------------------------

_DOMAIN_CHECKS: dict[str, list[dict[str, str]]] = {
    "em": [
        {
            "name": "boundary_condition_j_dot_n",
            "description": (
                "Verify that the normal component of the current density J is "
                "continuous across every material interface (boundary condition "
                "J1.n = J2.n in steady state)."
            ),
            "python_code": (
                "import numpy as np\n"
                "\n"
                "# TODO: replace J1_normal and J2_normal with the normal components\n"
                "# of the current density on each side of the interface.\n"
                "J1_normal = 0.0  # placeholder (A/m^2)\n"
                "J2_normal = 0.0  # placeholder (A/m^2)\n"
                "\n"
                "assert np.isclose(J1_normal, J2_normal, rtol=1e-6), (\n"
                "    f'Boundary condition violated: J1.n={J1_normal}, J2.n={J2_normal}'\n"
                ")\n"
            ),
        },
        {
            "name": "current_conservation_div_j",
            "description": (
                "Confirm that div J = 0 everywhere in the steady-state region, "
                "i.e., no net charge accumulation inside the conductor."
            ),
            "python_code": (
                "import sympy\n"
                "\n"
                "x, y, z = sympy.symbols('x y z')\n"
                "\n"
                "# TODO: replace Jx, Jy, Jz with the current density components.\n"
                "Jx = sympy.sympify('0')  # placeholder\n"
                "Jy = sympy.sympify('0')  # placeholder\n"
                "Jz = sympy.sympify('0')  # placeholder\n"
                "\n"
                "div_J = (\n"
                "    sympy.diff(Jx, x)\n"
                "    + sympy.diff(Jy, y)\n"
                "    + sympy.diff(Jz, z)\n"
                ")\n"
                "assert sympy.simplify(div_J) == 0, (\n"
                "    f'Current not conserved: div J = {div_J}'\n"
                ")\n"
            ),
        },
        {
            "name": "poynting_energy_balance",
            "description": (
                "Verify that the Poynting energy flux integrated over the surface "
                "equals the power dissipated inside the volume (Poynting's theorem)."
            ),
            "python_code": (
                "import numpy as np\n"
                "\n"
                "# TODO: replace surface_integral and volume_dissipation with\n"
                "# the computed values (W) for your geometry.\n"
                "surface_integral = 0.0   # placeholder (W)\n"
                "volume_dissipation = 0.0  # placeholder (W)\n"
                "\n"
                "assert np.isclose(surface_integral, volume_dissipation, rtol=1e-6), (\n"
                "    f\"Poynting balance failed: S_surface={surface_integral} W, \"\n"
                "    f\"P_dissipated={volume_dissipation} W\"\n"
                ")\n"
            ),
        },
    ],
    "mechanics": [
        {
            "name": "momentum_conservation",
            "description": (
                "Check that the total linear momentum of the system is constant when "
                "no external forces act, or changes at the correct rate when they do."
            ),
            "python_code": (
                "import sympy\n"
                "\n"
                "t = sympy.Symbol('t')\n"
                "\n"
                "# TODO: replace p_total with the total momentum as a sympy function of t,\n"
                "# and F_external with the net external force expression.\n"
                "p_total = sympy.sympify('0')    # placeholder (kg*m/s)\n"
                "F_external = sympy.sympify('0') # placeholder (N)\n"
                "\n"
                "dp_dt = sympy.diff(p_total, t)\n"
                "residual = sympy.simplify(dp_dt - F_external)\n"
                "assert residual == 0, (\n"
                "    f'Momentum not conserved: dp/dt - F_ext = {residual}'\n"
                ")\n"
            ),
        },
        {
            "name": "limiting_mass_cases",
            "description": (
                "Confirm that the result approaches physically sensible values when "
                "mass tends to zero and when mass tends to infinity."
            ),
            "python_code": (
                "import sympy\n"
                "\n"
                "m = sympy.Symbol('m', positive=True)\n"
                "\n"
                "# TODO: replace 'result_expr' with the expression that depends on m.\n"
                "result_expr = sympy.sympify('1')  # placeholder\n"
                "\n"
                "limit_zero = sympy.limit(result_expr, m, 0)\n"
                "limit_inf = sympy.limit(result_expr, m, sympy.oo)\n"
                "\n"
                "# TODO: assert expected limits below.\n"
                "print(f'm->0 limit: {limit_zero}')\n"
                "print(f'm->inf limit: {limit_inf}')\n"
                "assert limit_zero.is_finite, f'Unphysical divergence at m->0: {limit_zero}'\n"
            ),
        },
    ],
    "quantum": [
        {
            "name": "normalization",
            "description": (
                "Verify that the wave function integrates to unity over all space: "
                "integral |psi|^2 dx = 1."
            ),
            "python_code": (
                "import sympy\n"
                "\n"
                "x = sympy.Symbol('x', real=True)\n"
                "\n"
                "# TODO: replace psi with the wave function expression.\n"
                "psi = sympy.sympify('0')  # placeholder\n"
                "\n"
                "norm_sq = sympy.integrate(\n"
                "    sympy.conjugate(psi) * psi,\n"
                "    (x, -sympy.oo, sympy.oo),\n"
                ")\n"
                "assert sympy.simplify(norm_sq - 1) == 0, (\n"
                "    f'Wave function not normalised: integral |psi|^2 = {norm_sq}'\n"
                ")\n"
            ),
        },
        {
            "name": "hermiticity",
            "description": (
                "Confirm that the operator representing the observable is Hermitian "
                "(self-adjoint), guaranteeing real eigenvalues."
            ),
            "python_code": (
                "import sympy\n"
                "from sympy import Matrix\n"
                "\n"
                "# TODO: replace H_matrix with the matrix representation of the operator.\n"
                "H_matrix = Matrix([[1, 0], [0, 1]])  # placeholder (identity)\n"
                "\n"
                "H_dag = H_matrix.conjugate().transpose()\n"
                "assert H_matrix == H_dag, (\n"
                "    f'Operator is not Hermitian: H - H_dag = {H_matrix - H_dag}'\n"
                ")\n"
            ),
        },
        {
            "name": "correspondence_principle",
            "description": (
                "Check that expectation values of position and momentum follow "
                "classical equations of motion in the large quantum number limit."
            ),
            "python_code": (
                "import sympy\n"
                "\n"
                "n, hbar = sympy.symbols('n hbar', positive=True)\n"
                "\n"
                "# TODO: replace E_n with the energy eigenvalue expression.\n"
                "E_n = sympy.sympify('n')  # placeholder\n"
                "\n"
                "# Energy spacing should vanish as n -> inf for correspondence.\n"
                "dE = sympy.diff(E_n, n)\n"
                "limit_large_n = sympy.limit(dE / E_n, n, sympy.oo)\n"
                "print(f'Relative energy spacing as n->inf: {limit_large_n}')\n"
                "# For a harmonic oscillator this is 0; assert domain-specific expectation.\n"
            ),
        },
    ],
    "thermo": [
        {
            "name": "second_law",
            "description": (
                "Verify that the entropy change of the universe (system + surroundings) "
                "is non-negative for the described process."
            ),
            "python_code": (
                "import sympy\n"
                "\n"
                "# TODO: replace dS_system and dS_surroundings with the computed\n"
                "# entropy changes (J/K) for system and surroundings.\n"
                "dS_system = sympy.sympify('0')       # placeholder (J/K)\n"
                "dS_surroundings = sympy.sympify('0') # placeholder (J/K)\n"
                "\n"
                "dS_universe = dS_system + dS_surroundings\n"
                "assert sympy.simplify(dS_universe) >= 0, (\n"
                "    f'Second law violated: dS_universe = {dS_universe} J/K'\n"
                ")\n"
            ),
        },
        {
            "name": "temperature_limits",
            "description": (
                "Check that thermodynamic quantities remain physical as temperature "
                "approaches absolute zero (T -> 0) and as T -> infinity."
            ),
            "python_code": (
                "import sympy\n"
                "\n"
                "T = sympy.Symbol('T', positive=True)\n"
                "\n"
                "# TODO: replace quantity with the thermodynamic expression to check.\n"
                "quantity = sympy.sympify('1')  # placeholder\n"
                "\n"
                "limit_zero = sympy.limit(quantity, T, 0, '+')\n"
                "limit_inf = sympy.limit(quantity, T, sympy.oo)\n"
                "\n"
                "print(f'T->0  limit: {limit_zero}')\n"
                "print(f'T->inf limit: {limit_inf}')\n"
                "assert limit_zero.is_finite, (\n"
                "    f'Unphysical behaviour at T->0: {limit_zero}'\n"
                ")\n"
            ),
        },
    ],
    "waves": [
        {
            "name": "dispersion_consistency",
            "description": (
                "Verify that the wave vector k and angular frequency omega satisfy "
                "the dispersion relation of the medium."
            ),
            "python_code": (
                "import sympy\n"
                "\n"
                "k, omega, c = sympy.symbols('k omega c', positive=True)\n"
                "\n"
                "# TODO: replace lhs and rhs with the two sides of the dispersion\n"
                "# relation (e.g., omega**2 == c**2 * k**2 for free waves).\n"
                "lhs = omega**2        # placeholder\n"
                "rhs = c**2 * k**2    # placeholder\n"
                "\n"
                "residual = sympy.simplify(lhs - rhs)\n"
                "assert residual == 0, (\n"
                "    f'Dispersion relation not satisfied: lhs - rhs = {residual}'\n"
                ")\n"
            ),
        },
        {
            "name": "energy_flux_conservation",
            "description": (
                "Confirm that the time-averaged energy flux (intensity) is conserved "
                "across any interface or waveguide cross-section."
            ),
            "python_code": (
                "import numpy as np\n"
                "\n"
                "# TODO: replace flux_in and flux_out with the computed intensities\n"
                "# (W/m^2) on the incident and transmitted sides of the interface.\n"
                "flux_in = 0.0   # placeholder (W/m^2)\n"
                "flux_out = 0.0  # placeholder (W/m^2)\n"
                "\n"
                "assert np.isclose(flux_in, flux_out, rtol=1e-6), (\n"
                "    f'Energy flux not conserved: in={flux_in}, out={flux_out}'\n"
                ")\n"
            ),
        },
    ],
    "relativity": [
        {
            "name": "lorentz_invariance",
            "description": (
                "Verify that the computed observable is a Lorentz scalar (invariant "
                "under boosts), i.e., the value is the same in all inertial frames."
            ),
            "python_code": (
                "import sympy\n"
                "\n"
                "# Minkowski metric signature (-,+,+,+)\n"
                "eta = sympy.diag(-1, 1, 1, 1)\n"
                "\n"
                "# TODO: replace four_vector with the 4-vector to contract.\n"
                "four_vector = sympy.Matrix([1, 0, 0, 0])  # placeholder (ct, x, y, z)\n"
                "\n"
                "invariant = (four_vector.T * eta * four_vector)[0, 0]\n"
                "print(f'Lorentz invariant (should be real constant): {invariant}')\n"
                "assert invariant.is_number, (\n"
                "    f'Invariant depends on frame variables: {invariant}'\n"
                ")\n"
            ),
        },
        {
            "name": "non_relativistic_limit",
            "description": (
                "Confirm that setting v/c -> 0 (or beta -> 0) recovers the "
                "classical Newtonian expression."
            ),
            "python_code": (
                "import sympy\n"
                "\n"
                "beta = sympy.Symbol('beta', positive=True)  # v/c\n"
                "c = sympy.Symbol('c', positive=True)\n"
                "\n"
                "# TODO: replace relativistic_expr with the relativistic expression,\n"
                "# and classical_expr with the expected Newtonian limit.\n"
                "relativistic_expr = sympy.sympify('1')  # placeholder\n"
                "classical_expr = sympy.sympify('1')      # placeholder\n"
                "\n"
                "nr_limit = sympy.limit(relativistic_expr, beta, 0)\n"
                "residual = sympy.simplify(nr_limit - classical_expr)\n"
                "assert residual == 0, (\n"
                "    f'Non-relativistic limit wrong: got {nr_limit}, '\n"
                "    f'expected {classical_expr}'\n"
                ")\n"
            ),
        },
    ],
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SUPPORTED_DOMAINS = frozenset(_DOMAIN_CHECKS.keys())


def physics_checklist(
    domain: str,
    knowns: list[str],
    unknowns: list[str],
) -> dict[str, object]:
    """Generate a list of domain-specific verification checks for a physics solution.

    Args:
        domain: Physics subdomain. One of "em", "mechanics", "quantum",
            "thermo", "waves", "relativity". Unknown values fall back to
            the generic checks only.
        knowns: List of known quantities in the problem (for context).
        unknowns: List of quantities the solution must determine (for context).

    Returns:
        A dict with two keys:
            "checks": list of check dicts, each with "name", "description",
                and "python_code" (valid, parseable Python).
            "domain_notes": str with common errors for this domain.
    """
    domain_key = domain.lower()

    domain_specific = _DOMAIN_CHECKS.get(domain_key, [])
    checks = list(_GENERIC_CHECKS) + list(domain_specific)

    domain_notes = _DOMAIN_NOTES.get(
        domain_key,
        (
            f"No domain-specific notes for '{domain}'. "
            "Apply generic checks: dimensional analysis, energy conservation, "
            "and limiting cases."
        ),
    )

    return {
        "checks": checks,
        "domain_notes": domain_notes,
    }
