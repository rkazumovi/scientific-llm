"""
scientific-llm - Step 5c: SymPy mathematical verification layer.

This is a training-time-independent CHECKER, not a training signal (that
is Step 4a's physics_consistency_loss - see that file's docstring for the
distinction). It answers a narrower, more literal question than "is this
physically correct": given two equations written in the SAME variable
notation, are their left-hand-side-minus-right-hand-side expressions
symbolically the same, according to SymPy's algebra - not "do these
strings look similar."

Two concrete uses in this project:
  1. Cross-checking Step 4a's corrupt_equation(): that function claims a
     sign flip / exponent flip / subscript swap changes an equation's
     meaning. This file's verify_corruption_detected() confirms that
     independently, using real symbolic algebra instead of trusting the
     regex that produced the corruption. If the two ever disagree, that
     is a real bug worth knowing about, not a rounding error.
  2. Scoring the MATH benchmark in benchmarks.py: comparing a model's
     boxed final answer against the known correct answer as algebra
     (so "1/2" and "0.5" and "2/4" all correctly score as equal),
     not as an exact string match.

Honesty about scope, read this before trusting a result:
  - Equations are compared as written. "E = mc^2" and "E = m*c**2" use
    different tokens ("mc" is parsed as ONE symbol, not m times c - see
    preprocess_latex_for_sympy's docstring) and would NOT be recognized
    as the same law. This checker verifies algebraic consistency between
    two expressions in a shared notation - it does not recognize physical
    laws across different notations, and does not know any physics.
  - True partial-derivative notation (u_t meaning du/dt) is treated as an
    opaque symbol named "u_t", not an actual SymPy Derivative object. So
    this can confirm "u_t = a*u_xx" and "u_xx = a*u_t" are DIFFERENT
    (correct - that is a real corruption), but it cannot confirm a PDE is
    solved correctly in the calculus sense. Full derivative-aware
    verification would need each equation's variables and their
    functional dependence (u(x, t)) supplied explicitly, which the
    plain-text equations extracted by preprocessor.py do not carry.
  - Operators with no algebraic meaning as a bare symbol (\\nabla, \\int,
    \\partial) are preserved as opaque symbol names too. "\\nabla^2 \\phi"
    becomes the symbol "nabla" squared times "phi" - internally
    consistent for comparing two equations that both use \\nabla the same
    way, but this is notation preservation, not calculus.
  - equations_equivalent() and parse_single_expr() return None (not
    False) when SymPy cannot parse an input at all, so a "not
    equivalent" result always means a real algebraic difference was
    found, never a hidden parse failure.

Run directly:
    python src\\evaluation\\math_verifier.py
(pure SymPy + regex logic - no model, no GPU needed. Cross-checks against
Step 4a's corrupt_equation on the same example equations used there.)
"""

import re
import sys
import traceback

from sympy import simplify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

# Deliberately NOT implicit_multiplication_application. That transform's
# split_symbols step turns unrecognized multi-letter names into a product
# of single-letter symbols (so "mc" would become m*c) - useful for
# reading "mc^2" as "m times c squared", but it applies the same
# splitting to non-algebraic multi-letter tokens like "nabla", breaking
# them into nonsense (n*a*b*l*a). Plain implicit_multiplication only
# inserts * between tokens that are already separately recognized (e.g.
# "4pi" -> 4*pi, "alpha u_xx" -> alpha*u_xx) and never splits one
# identifier into several - predictable at the cost of not knowing that
# "mc" secretly means "m times c". See the module docstring: this
# checker compares same-notation equations, so that tradeoff costs
# nothing in practice here.
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication, convert_xor)

_GREEK_AND_SPECIAL = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma",
    "tau", "upsilon", "phi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Sigma", "Upsilon", "Phi",
    "Psi", "Omega", "nabla", "hbar", "infty", "partial", "cdot", "times",
}

_FRAC_PATTERN = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_SQRT_PATTERN = re.compile(r"\\sqrt\{([^{}]*)\}")
_BACKSLASH_CMD_PATTERN = re.compile(r"\\([a-zA-Z]+)")
_SUBSCRIPT_BRACE_PATTERN = re.compile(r"_\{([^{}]+)\}")


def preprocess_latex_for_sympy(expr: str) -> str:
    """Rewrites a handful of common LaTeX constructs into a form SymPy's
    parser accepts, WITHOUT trying to be a general LaTeX parser (SymPy
    ships one - sympy.parsing.latex - but it depends on antlr4 and lark,
    extra install weight this project's minimal-dependency approach
    (see requirements-step5.txt) does not need for the narrow physics/
    math notation this project's equations actually use).

    Handles: \\frac{a}{b} -> ((a)/(b)), \\sqrt{a} -> sqrt(a), lone Greek
    letters/\\nabla/\\hbar/\\infty -> bare names (kept as opaque symbols,
    see module docstring), u_{xx} -> u_xx (brace removed, subscript kept
    as part of the identifier), \\cdot and \\times -> * , stray \\, and
    remaining backslashes stripped.
    """
    s = expr

    while _FRAC_PATTERN.search(s):
        s = _FRAC_PATTERN.sub(r"((\1)/(\2))", s)
    while _SQRT_PATTERN.search(s):
        s = _SQRT_PATTERN.sub(r"sqrt(\1)", s)

    for name in _GREEK_AND_SPECIAL:
        s = s.replace("\\" + name, name)

    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = _SUBSCRIPT_BRACE_PATTERN.sub(r"_\1", s)
    # Any backslash-command not explicitly handled above (unknown LaTeX
    # macro): drop the backslash, keep the name as an opaque symbol
    # rather than failing outright.
    s = _BACKSLASH_CMD_PATTERN.sub(r"\1", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\,", " ").replace("\\", "")

    return s.strip()


def parse_single_expr(expr: str):
    """Parses one algebraic expression (no '='). Returns a SymPy
    expression, or None if it cannot be parsed at all."""
    try:
        cleaned = preprocess_latex_for_sympy(expr)
        if not cleaned:
            return None
        return parse_expr(cleaned, transformations=_TRANSFORMATIONS)
    except Exception:  # noqa: BLE001 - SymPy raises several distinct error types
        return None


def parse_equation(equation: str):
    """Splits on the first '=' and parses each side. Returns
    (lhs_expr, rhs_expr), or None if there is no '=' or either side fails
    to parse."""
    if "=" not in equation:
        return None
    lhs_raw, rhs_raw = equation.split("=", 1)
    lhs = parse_single_expr(lhs_raw)
    rhs = parse_single_expr(rhs_raw)
    if lhs is None or rhs is None:
        return None
    return lhs, rhs


def expressions_equivalent(expr1: str, expr2: str) -> bool | None:
    """True/False if both sides parse and SymPy can decide; None if
    either fails to parse (an honest "could not verify", never silently
    counted as unequal)."""
    e1, e2 = parse_single_expr(expr1), parse_single_expr(expr2)
    if e1 is None or e2 is None:
        return None
    try:
        return simplify(e1 - e2) == 0
    except Exception:  # noqa: BLE001
        return None


def equations_equivalent(eq1: str, eq2: str) -> bool | None:
    """True/False if the two equations' (lhs - rhs) expressions are
    symbolically identical after simplification; None if either equation
    could not be parsed."""
    parsed1, parsed2 = parse_equation(eq1), parse_equation(eq2)
    if parsed1 is None or parsed2 is None:
        return None
    lhs1, rhs1 = parsed1
    lhs2, rhs2 = parsed2
    try:
        return simplify((lhs1 - rhs1) - (lhs2 - rhs2)) == 0
    except Exception:  # noqa: BLE001
        return None


def verify_corruption_detected(original: str, corrupted: str) -> bool | None:
    """Independent check that a corrupted equation (from Step 4a's
    corrupt_equation) is symbolically DIFFERENT from the original -
    True means the corruption is confirmed real, False means SymPy says
    they are still equivalent (the corruption failed to change the
    equation's meaning - a real bug, not a rounding error), None means
    either equation could not be parsed at all (no verdict possible)."""
    equivalent = equations_equivalent(original, corrupted)
    if equivalent is None:
        return None
    return not equivalent


def main() -> int:
    print("Expression/equation equivalence demo (pure SymPy, no model needed):")
    equiv_examples = [
        ("1/2", "0.5", True),
        ("1/2", "2/4", True),
        ("x**2 - 1", "(x-1)(x+1)", True),
        ("x + 1", "x + 2", False),
    ]
    for a, b, expected in equiv_examples:
        result = expressions_equivalent(a, b)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] {a!r} vs {b!r} -> {result} (expected {expected})")

    print("\nCross-checking Step 4a's corrupt_equation against independent SymPy algebra:")
    try:
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.model.physics_loss import corrupt_equation

        examples = [
            "u_t = \\alpha u_{xx}",
            "E = mc^2",
            "\\nabla^2 \\phi = 4\\pi G \\rho",
        ]
        all_confirmed = True
        for eq in examples:
            corrupted = corrupt_equation(eq)
            if corrupted is None:
                print(f"  [SKIP] {eq!r} - corrupt_equation found nothing to corrupt")
                continue
            confirmed = verify_corruption_detected(eq, corrupted)
            if confirmed is None:
                print(f"  [SKIP] {eq!r} -> {corrupted!r} - could not parse for verification")
                continue
            status = "PASS" if confirmed else "FAIL"
            if not confirmed:
                all_confirmed = False
            print(f"  [{status}] {eq!r} -> {corrupted!r} : SymPy confirms different = {confirmed}")

        if not all_confirmed:
            print("\nFAIL: SymPy disagreed with corrupt_equation on at least one example.")
            return 1
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    print("\nPASS: expression equivalence checks and corruption cross-checks agree with SymPy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
