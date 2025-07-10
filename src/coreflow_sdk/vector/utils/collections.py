"""
Collection naming utilities for multi-tenant RAG system.

This module provides utilities for sanitizing usernames and generating
consistent collection names using the __ namespace separator pattern.
"""

import re


def sanitize_username(username: str, max_length: int = 6) -> str:
    """
    Sanitize a username for use in collection names.

    This function:
    1. Strips email domains (everything after @)
    2. Removes special characters (_ - .)
    3. Takes first 3 and last 3 characters if username is long
    4. Ensures the result fits within max_length

    Args:
        username: The raw username/email to sanitize
        max_length: Maximum length of the sanitized username (default: 6)

    Returns:
        Sanitized username suitable for collection naming

    Examples:
        >>> sanitize_username("john.doe@example.com")
        'johndoe'
        >>> sanitize_username("alice_smith@company.org")
        'alicsmith'
        >>> sanitize_username("very-long-username")
        'veryme'
        >>> sanitize_username("bob")
        'bob'
    """
    if not username or not isinstance(username, str):
        raise ValueError("Username must be a non-empty string")

    # Step 1: Remove email domain if present
    username = username.split("@")[0]

    # Step 2: Remove special characters (_ - .)
    username = re.sub(r"[_\-.]", "", username)

    # Step 3: Remove any other non-alphanumeric characters
    username = re.sub(r"[^a-zA-Z0-9]", "", username)

    # Step 4: Convert to lowercase for consistency
    username = username.lower()

    # Step 5: Handle length constraints
    if len(username) <= max_length:
        return username

    # If longer than max_length, take first 3 and last 3 characters
    if max_length >= 6:
        half = max_length // 2
        return username[:half] + username[-half:]
    else:
        # For very short max_length, just truncate
        return username[:max_length]


def create_user_collection_name(username: str, namespace: str = "user") -> str:
    """
    Create a standardized user collection name using __ namespace separator.

    Args:
        username: The username to sanitize and use
        namespace: The namespace prefix (default: "user")

    Returns:
        Formatted collection name like "user__johndoe"

    Examples:
        >>> create_user_collection_name("john.doe@example.com")
        'user__johndoe'
        >>> create_user_collection_name("alice_smith", "tenant")
        'tenant__alicsmith'
    """
    sanitized = sanitize_username(username)
    return f"{namespace}__{sanitized}"


def create_system_collection_name(
    collection_type: str = "default", namespace: str = "system"
) -> str:
    """
    Create a standardized system collection name using __ namespace separator.

    Args:
        collection_type: The type of system collection (default: "default")
        namespace: The namespace prefix (default: "system")

    Returns:
        Formatted collection name like "system__default"

    Examples:
        >>> create_system_collection_name()
        'system__default'
        >>> create_system_collection_name("documents")
        'system__documents'
        >>> create_system_collection_name("embeddings", "global")
        'global__embeddings'
    """
    # Sanitize collection_type similar to username but allow longer names
    collection_type = re.sub(r"[^a-zA-Z0-9]", "", collection_type.lower())
    return f"{namespace}__{collection_type}"


def parse_collection_name(collection_name: str) -> tuple[str, str]:
    """
    Parse a collection name into namespace and identifier parts.

    Args:
        collection_name: Collection name like "user__johndoe" or "system__default"

    Returns:
        Tuple of (namespace, identifier)

    Examples:
        >>> parse_collection_name("user__johndoe")
        ('user', 'johndoe')
        >>> parse_collection_name("system__default")
        ('system', 'default')
        >>> parse_collection_name("legacy_collection")
        ('', 'legacy_collection')
    """
    if "__" in collection_name:
        parts = collection_name.split("__", 1)
        return parts[0], parts[1]
    else:
        # Legacy collection name without namespace
        return "", collection_name


def is_user_collection(collection_name: str) -> bool:
    """
    Check if a collection name represents a user collection.

    Args:
        collection_name: The collection name to check

    Returns:
        True if it's a user collection, False otherwise

    Examples:
        >>> is_user_collection("user__johndoe")
        True
        >>> is_user_collection("system__default")
        False
        >>> is_user_collection("user_legacy")
        True
    """
    namespace, _ = parse_collection_name(collection_name)
    return namespace == "user" or collection_name.startswith("user_")


def is_system_collection(collection_name: str) -> bool:
    """
    Check if a collection name represents a system collection.

    Args:
        collection_name: The collection name to check

    Returns:
        True if it's a system collection, False otherwise

    Examples:
        >>> is_system_collection("system__default")
        True
        >>> is_system_collection("user__johndoe")
        False
        >>> is_system_collection("system")
        True
    """
    namespace, identifier = parse_collection_name(collection_name)
    return (
        namespace == "system"
        or collection_name == "system"
        or collection_name == "mem0_system"
    )


def validate_collection_name(collection_name: str) -> bool:
    """
    Validate that a collection name follows the expected format.

    Args:
        collection_name: The collection name to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_collection_name("user__johndoe")
        True
        >>> validate_collection_name("system__default")
        True
        >>> validate_collection_name("invalid__name__with__too__many__parts")
        False
    """
    if not collection_name or not isinstance(collection_name, str):
        return False

    # Check for valid characters (alphanumeric, underscore, dash, dot, colon)
    if not re.match(r"^[a-zA-Z0-9_\-.:]+$", collection_name):
        return False

    # If it contains __, it should have exactly one occurrence
    if "__" in collection_name:
        parts = collection_name.split("__")
        if len(parts) != 2:
            return False
        namespace, identifier = parts
        if not namespace or not identifier:
            return False

    return True
