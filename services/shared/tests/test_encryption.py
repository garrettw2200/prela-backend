"""Tests for encryption utilities.

Tests the Fernet encryption/decryption logic used by the data source
module to encrypt Langfuse API secrets at rest.
"""

import pytest
from cryptography.fernet import Fernet, InvalidToken


def _make_encrypt_decrypt(key: str):
    """Create encrypt/decrypt functions using a specific key.

    Mirrors the logic in shared/encryption.py without needing
    to import it (avoids relative import issues in test context).
    """
    cipher = Fernet(key.encode())

    def encrypt(plaintext: str) -> str:
        return cipher.encrypt(plaintext.encode()).decode()

    def decrypt(ciphertext: str) -> str:
        return cipher.decrypt(ciphertext.encode()).decode()

    return encrypt, decrypt


@pytest.fixture
def fernet_key():
    """Generate a fresh Fernet key for each test."""
    return Fernet.generate_key().decode()


def test_encrypt_decrypt_roundtrip(fernet_key):
    """Encrypting then decrypting returns the original value."""
    encrypt, decrypt = _make_encrypt_decrypt(fernet_key)

    original = "sk-lf-abc123-secret-key"
    encrypted = encrypt(original)

    assert encrypted != original
    assert decrypt(encrypted) == original


def test_encrypt_produces_different_ciphertexts(fernet_key):
    """Fernet produces different ciphertexts for the same plaintext (due to timestamp/IV)."""
    encrypt, _ = _make_encrypt_decrypt(fernet_key)

    plaintext = "test-secret"
    a = encrypt(plaintext)
    b = encrypt(plaintext)
    assert a != b


def test_decrypt_with_wrong_key_fails(fernet_key):
    """Decrypting with a different key raises InvalidToken."""
    encrypt, _ = _make_encrypt_decrypt(fernet_key)
    encrypted = encrypt("my-secret")

    # Decrypt with a different key
    other_key = Fernet.generate_key().decode()
    _, wrong_decrypt = _make_encrypt_decrypt(other_key)

    with pytest.raises(InvalidToken):
        wrong_decrypt(encrypted)


def test_encrypt_empty_string(fernet_key):
    """Encrypting an empty string works."""
    encrypt, decrypt = _make_encrypt_decrypt(fernet_key)

    encrypted = encrypt("")
    assert decrypt(encrypted) == ""


def test_encrypt_unicode(fernet_key):
    """Encrypting unicode strings works."""
    encrypt, decrypt = _make_encrypt_decrypt(fernet_key)

    original = "secret-with-unicode-\u00e9\u00e8\u00ea"
    encrypted = encrypt(original)
    assert decrypt(encrypted) == original


def test_encrypt_long_key(fernet_key):
    """Encrypting long secret keys works."""
    encrypt, decrypt = _make_encrypt_decrypt(fernet_key)

    original = "sk-lf-" + "a" * 500
    encrypted = encrypt(original)
    assert decrypt(encrypted) == original


def test_invalid_ciphertext(fernet_key):
    """Decrypting garbage data raises InvalidToken."""
    _, decrypt = _make_encrypt_decrypt(fernet_key)

    with pytest.raises(Exception):
        decrypt("not-valid-ciphertext")
