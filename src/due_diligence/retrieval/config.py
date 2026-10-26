"""
Re-export of core configuration for the retrieval subsystem.

Provides convenient access to Settings and get_settings() within
the retrieval namespace, delegating to the centralized configuration
in ``due_diligence.core.config``.
"""

from due_diligence.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
