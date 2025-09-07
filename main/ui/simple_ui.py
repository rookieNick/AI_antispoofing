"""
Simple UI - Backward Compatibility Layer
This file provides backward compatibility by importing the new separated UI
The old monolithic UI code has been moved to simple_ui_backup.py
"""

# Import the new separated UI
from .antispoofing_ui import launch_ui

# For backward compatibility, also expose the main class
from .antispoofing_ui import AntiSpoofingUI

# Legacy function names for compatibility
def launch_simple_ui():
    """Legacy function name - calls new launch_ui"""
    return launch_ui()

# Export for external imports
__all__ = ['launch_ui', 'AntiSpoofingUI', 'launch_simple_ui']
