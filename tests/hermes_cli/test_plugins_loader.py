import sys
import threading
import types

import pytest

from hermes_cli import plugins_loader


def test_nested_plugin_load_runs_inline_on_deadline_worker(monkeypatch):
    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: 0.3)
    abandoned = []

    class Context:
        def __init__(self, name):
            self.name = name

        def _abandon_load(self):
            abandoned.append(self.name)

    observed_threads = []
    release = threading.Event()

    def outer_load():
        observed_threads.append(threading.current_thread())
        plugins_loader.run_with_load_deadline(
            "done", Context("done"), lambda: observed_threads.append(threading.current_thread()),
        )
        plugins_loader.run_with_load_deadline("hung", Context("hung"), release.wait)

    with pytest.raises(plugins_loader.PluginLoadTimeout):
        plugins_loader.run_with_load_deadline("outer", Context("outer"), outer_load)
    release.set()

    assert len(observed_threads) == 2
    assert observed_threads[0] is observed_threads[1]
    # The outer timeout abandons only contexts still loading, not a nested load that already finished.
    assert abandoned == ["outer", "hung"]


def test_evict_modules_survives_concurrent_sys_modules_mutation(monkeypatch):
    """Another thread importing while a plugin is evicted must not abort the load.

    Iterating ``sys.modules`` live raised ``RuntimeError: dictionary changed size during
    iteration``, so the plugin silently failed to load.
    """
    # Ballast widens the window in which the scan can be interleaved with the writer thread.
    for i in range(2000):
        monkeypatch.setitem(sys.modules, f"_evict_race_ballast_{i}", types.ModuleType("x"))
    original_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    stop = threading.Event()

    def churn():
        i = 0
        while not stop.is_set():
            sys.modules[f"_evict_race_churn_{i}"] = types.ModuleType("x")
            sys.modules.pop(f"_evict_race_churn_{i - 50}", None)
            i += 1

    writer = threading.Thread(target=churn, daemon=True)
    writer.start()
    try:
        for _ in range(2000):
            plugins_loader._evict_modules("hermes_plugins.evict_race_target")
    finally:
        stop.set()
        writer.join(timeout=5)
        sys.setswitchinterval(original_interval)
        for name in [n for n in list(sys.modules) if n.startswith("_evict_race_churn_")]:
            sys.modules.pop(name, None)
