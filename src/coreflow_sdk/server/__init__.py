"""
FastAPI Server Module for CoreFlow SDK

This module provides ready-to-use FastAPI server components that users can import
and use to create their own servers with credential-aware endpoints.
"""

from .api.main import app, create_app

__all__ = ["app", "create_app"]
