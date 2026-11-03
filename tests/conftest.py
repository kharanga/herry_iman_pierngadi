import pytest

from tests.results_pages import is_missing_results_page


def skip_for_a_missing_results_page(error: FileNotFoundError) -> None:
    filename = getattr(error, "filename", None)
    if filename and is_missing_results_page(filename):
        pytest.skip(f"results page not on disk: {filename} (docs/results/ is local only)")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except FileNotFoundError as error:
        skip_for_a_missing_results_page(error)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except FileNotFoundError as error:
        skip_for_a_missing_results_page(error)
        raise
