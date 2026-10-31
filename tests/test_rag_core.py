import pytest

from rag_core import private_collection_name, safe_filename, uploads_fingerprint


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("notes.pdf", "notes.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("..\\..\\windows\\system.pdf", "system.pdf"),
        ("", "upload.pdf"),
    ],
)
def test_uploaded_names_cannot_leave_the_temp_directory(
    name: str, expected: str
) -> None:
    assert safe_filename(name) == expected


def test_the_same_uploads_match_in_any_order() -> None:
    first = uploads_fingerprint([("a.pdf", b"one"), ("b.pdf", b"two")])
    second = uploads_fingerprint([("b.pdf", b"two"), ("a.pdf", b"one")])

    assert first == second


def test_changed_content_is_detected() -> None:
    assert uploads_fingerprint([("a.pdf", b"one")]) != uploads_fingerprint(
        [("a.pdf", b"two")]
    )


def test_each_index_gets_its_own_collection() -> None:
    assert private_collection_name() != private_collection_name()


def test_collection_names_are_valid_for_chroma() -> None:
    name = private_collection_name()

    assert 3 <= len(name) <= 63
    assert name.replace("_", "").isalnum()
