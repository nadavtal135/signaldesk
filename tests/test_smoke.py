# Smoke test: confirms the package imports, so pytest has something to run from day one.
def test_package_imports():
    import signaldesk  # noqa: F401
