import os


class FileWatcher(object):
    def __init__(self, filepath):
        self._cached_stamp = 0
        self.filepath = filepath

    def has_timestamp_changed(self):
        stamp = os.stat(self.filepath).st_mtime
        if stamp != self._cached_stamp:
            self._cached_stamp = stamp
            return True
        return False
