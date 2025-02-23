import win32gui
import win32con
import win32api
from screeninfo import get_monitors


class MonitorWatcher:
    def __init__(self):
        self.monitors = get_monitors()
        self.current_monitor_count = self._get_monitor_count()
        self._last_state = self.current_monitor_count
        self._hwnd = self._create_listener_window()

    def _get_monitor_count(self) -> int:
        return len(self.monitors)

    def _create_listener_window(self):
        def monitor_change_handler(hwnd, msg, wparam, lparam):
            if msg == win32con.WM_DISPLAYCHANGE:
                self._last_state = None  # Invalidate to trigger refresh
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = monitor_change_handler
        wc.lpszClassName = "MonitorChangeListener"
        class_atom = win32gui.RegisterClass(wc)
        return win32gui.CreateWindow(
            class_atom, "Monitor Change Listener", 0, 0, 0, 0, 0, 0, 0, 0, None
        )

    def has_monitor_count_changed(self) -> bool:
        """This function needs to be called in order to update the current_monitor_count variable."""
        win32gui.PumpWaitingMessages()
        if self._last_state is None:  # If invalidated
            new_monitor_count = self._get_monitor_count()
            if new_monitor_count != self.current_monitor_count:
                self.monitors = get_monitors()
                self.current_monitor_count = new_monitor_count
                self._last_state = self.current_monitor_count  # Update last state
                return True
        return False

    def cleanup(self):
        win32gui.DestroyWindow(self._hwnd)
