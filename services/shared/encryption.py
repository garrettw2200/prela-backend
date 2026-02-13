"""Encryption utilities for storing secrets at rest."""

import logging

from cryptography.fernet import Fernet, InvalidToken

from .config import settings

logger = logging.getLogger(__name__)


def _get_cipher() -> Fernet:
    """Get Fernet cipher from configured encryption key."""
    if not settings.data_source_encryption_key:
        raise ValueError(
            "DATA_SOURCE_ENCRYPTION_KEY not configured. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(settings.data_source_encryption_key.encode())


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a secret string for storage.

    Args:
        plaintext: The secret to encrypt.

    Returns:
        Base64-encoded encrypted string.
    """
    cipher = _get_cipher()
    return cipher.encrypt(plaintext.encode()).decode()


def decrypt_secret(encrypted: str) -> str:
    """Decrypt a stored secret.

    Args:
        encrypted: Base64-encoded encrypted string.

    Returns:
        Original plaintext secret.

    Raises:
        InvalidToken: If the encrypted data is invalid or the key is wrong.
    """
    cipher = _get_cipher()
    try:
        return cipher.decrypt(encrypted.encode()).decode()
    except InvalidToken:
        logger.error("Failed to decrypt secret — encryption key may have changed")
        raise
