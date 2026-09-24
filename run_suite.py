"""Run the regression suite with real pytest if available, otherwise with a minimal collector."""
import importlib, sys, time, traceback, pathlib
root = pathlib.Path(__file__).parent
try:
    import pytest  # noqa
    if not hasattr(pytest, "main"):
        raise ImportError
    sys.exit(pytest.main(["-q", str(root / "tests")]))
except ImportError:
    pass
sys.path[:0] = [str(root / "shim"), str(root), str(root / "tests")]
passed, failed = 0, []
for f in sorted((root / "tests").glob("test_*.py")):
    mod = importlib.import_module(f.stem)
    for name in [n for n in dir(mod) if n.startswith("test_")]:
        fn = getattr(mod, name)
        cases = [()] if not hasattr(fn, "_params") else [tuple(v) if isinstance(v, tuple) else (v,) for v in fn._params[1]]
        for args in cases:
            label = f"{f.stem}::{name}{'' if not args else list(args)}"
            t = time.time()
            try:
                fn(*args); passed += 1; print(f"PASS  {label}  ({time.time()-t:.2f}s)")
            except Exception:
                failed.append(label); print(f"FAIL  {label}\n{traceback.format_exc()}")
print(f"\n{passed} passed, {len(failed)} failed")
sys.exit(1 if failed else 0)
