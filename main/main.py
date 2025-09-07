# Entry point for the Anti-Spoofing application
# This will launch the new separated UI

try:
    # Try the new separated UI first
    from ui.antispoofing_ui import launch_ui
except ImportError:
    # Fallback to simple_ui for backward compatibility
    from ui.simple_ui import launch_ui

if __name__ == "__main__":
    launch_ui()
