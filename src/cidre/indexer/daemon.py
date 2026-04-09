from __future__ import annotations
import sys
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from cidre.indexer.scanner import classify_file, should_exclude


class CidreEventHandler(FileSystemEventHandler):
    def __init__(self, callback, exclude_patterns: list[str] | None = None):
        super().__init__()
        self._callback = callback
        self._exclude = exclude_patterns or []

    def _should_process(self, path_str: str) -> bool:
        path = Path(path_str)
        if should_exclude(path, self._exclude):
            return False
        return classify_file(path) is not None

    def on_created(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            self._callback(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            self._callback(event.src_path)


def start_watcher(directories: list[str], callback, exclude_patterns: list[str] | None = None):
    handler = CidreEventHandler(callback=callback, exclude_patterns=exclude_patterns)
    observer = Observer()
    for dir_path in directories:
        resolved = Path(dir_path).expanduser().resolve()
        if resolved.is_dir():
            observer.schedule(handler, str(resolved), recursive=True)
    observer.start()
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def generate_launchd_plist() -> str:
    python_path = sys.executable
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.cidre.watcher</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>cidre.cli</string>
        <string>watch</string>
        <string>start</string>
        <string>--foreground</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{Path.home()}/.cidre/logs/daemon.log</string>
    <key>StandardErrorPath</key>
    <string>{Path.home()}/.cidre/logs/daemon.log</string>
</dict>
</plist>"""
