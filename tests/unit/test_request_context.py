from handwriting.core.request_context import (
    get_request_id,
    request_context,
)


def test_request_id_is_none_by_default():
    assert get_request_id() is None


def test_request_context_sets_and_resets_request_id():
    assert get_request_id() is None

    with request_context("test-request-id"):
        assert get_request_id() == "test-request-id"

    assert get_request_id() is None