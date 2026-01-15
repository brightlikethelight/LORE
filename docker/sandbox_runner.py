#!/usr/bin/env python3
"""Sandbox runner for isolated code execution.

This script runs in the sandbox container and executes code snippets
with strict resource and capability limitations.
"""

import ast
import io
import json
import resource
import signal
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Any


# Timeout handler
class TimeoutError(Exception):
    pass


def timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Execution timed out")


# Safe builtins (restricted set)
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    # Math functions via import
    "__import__": None,  # Disabled
}

# Allowed modules (whitelist)
ALLOWED_MODULES = {
    "math",
    "cmath",
    "random",
    "itertools",
    "functools",
    "operator",
    "collections",
    "fractions",
    "decimal",
    "statistics",
}


def restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Restricted import that only allows whitelisted modules."""
    if name not in ALLOWED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed")
    return __builtins__.__import__(name, *args, **kwargs)


def set_resource_limits() -> None:
    """Set resource limits for the execution."""
    # CPU time limit (seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))

    # Memory limit (bytes) - 256MB
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))

    # Limit file descriptors
    resource.setrlimit(resource.RLIMIT_NOFILE, (10, 10))

    # Disable core dumps
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def validate_ast(code: str) -> bool:
    """Validate code AST for dangerous operations."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        # Block dangerous operations
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES:
                    return False

        if isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_MODULES:
                return False

        # Block attribute access to dangerous things
        if isinstance(node, ast.Attribute):
            if node.attr in ["__class__", "__bases__", "__subclasses__",
                           "__globals__", "__code__", "__builtins__"]:
                return False

    return True


def execute_code(code: str, timeout: int = 5) -> dict[str, Any]:
    """Execute code in a restricted environment.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with stdout, stderr, result, and error fields
    """
    result = {
        "stdout": "",
        "stderr": "",
        "result": None,
        "error": None,
        "timed_out": False,
    }

    # Validate AST
    if not validate_ast(code):
        result["error"] = "Code contains disallowed operations"
        return result

    # Set up restricted globals
    restricted_globals = {
        "__builtins__": {**SAFE_BUILTINS, "__import__": restricted_import},
        "__name__": "__main__",
    }

    # Capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compile(code, "<sandbox>", "exec"), restricted_globals)

        result["stdout"] = stdout_capture.getvalue()
        result["stderr"] = stderr_capture.getvalue()

    except TimeoutError:
        result["timed_out"] = True
        result["error"] = "Execution timed out"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"

    finally:
        signal.alarm(0)

    return result


def main() -> None:
    """Main entry point for sandbox runner."""
    # Set resource limits
    set_resource_limits()

    # Read code from stdin
    if len(sys.argv) > 1:
        # Code passed as argument
        code = sys.argv[1]
    else:
        # Read from stdin
        code = sys.stdin.read()

    # Execute and output result
    result = execute_code(code)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
