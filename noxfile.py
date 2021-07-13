"""Nox sessions."""
# TODO: fix install_with_constraints
import tempfile

import nox

locations = "PyCausality", "tests", "noxfile.py", "docs/conf.py"
nox.options.sessions = "lint", "safety"


def install_with_constraints(session, *args, **kwargs):
    """Install packages constrained by Poetry's lock file."""
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python="3.8")
def black(session):
    """Run black code formatter."""
    args = session.posargs or locations
    # install_with_constraints(session, "black")
    session.install("black")
    session.run("black", *args)


@nox.session(python=["3.8", "3.7"])
def lint(session):
    """Lint using flake8."""
    args = session.posargs or locations
    # install_with_constraints(
    #     session,
    #     "flake8",
    #     "flake8-annotations",
    #     "flake8-bandit",
    #     "flake8-black",
    #     "flake8-bugbear",
    #     "flake8-docstrings",
    #     "flake8-import-order",
    # )
    session.install(
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-import-order",
    )
    session.run("flake8", *args)


@nox.session(python="3.8")
def safety(session):
    """Scan dependencies for insecure packages."""
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        # install_with_constraints(session, "safety")
        session.install("safety")
        session.run("safety", "check", f"--file={requirements.name}", "--full-report")


# @nox.session(python=["3.8", "3.7"])
# def tests(session):
#     """Run the test suite."""
#     session.run("poetry", "install", "--no-dev", external=True)
#     # install_with_constraints(
#     #     session, "nose"
#     # )
#     session.install("nosetests")
#     session.run("nose")


# @nox.session(python=["3.8", "3.7"])
# def mypy(session):
#     """Type-check using mypy."""
#     args = session.posargs or locations
#     install_with_constraints(session, "mypy")
#     session.run("mypy", *args)


# @nox.session(python="3.8")
# def docs(session):
#     """Build the documentation."""
#     install_with_constraints(session, "sphinx")
#     session.run("sphinx-build", "docs", "docs/_build")
