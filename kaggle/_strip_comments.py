"""Helper: strip Python comments + docstrings from .py files (and embedded code strings in notebook builders)."""
import io
import re
import sys
import ast
import tokenize
from pathlib import Path


def strip_python_comments(source: str) -> str:
    out = io.StringIO()
    last_lineno = -1
    last_col = 0
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for tok in tokens:
        token_type, token_string = tok.type, tok.string
        sl, sc = tok.start
        el, ec = tok.end
        if sl > last_lineno:
            last_col = 0
        if sc > last_col:
            out.write(" " * (sc - last_col))
        if token_type == tokenize.COMMENT:
            pass
        else:
            out.write(token_string)
        last_col = ec
        last_lineno = el
    return out.getvalue()


def remove_docstrings(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines_to_delete = set()

    def collect(node):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                ds = body[0]
                for ln in range(ds.lineno, ds.end_lineno + 1):
                    lines_to_delete.add(ln)

    for node in ast.walk(tree):
        collect(node)

    lines = source.split("\n")
    kept = [ln for i, ln in enumerate(lines, 1) if i not in lines_to_delete]
    return "\n".join(kept)


def strip_inside_triple_strings(source: str) -> str:
    pattern = re.compile(r"('''|\"\"\")(.*?)\1", re.DOTALL)

    def repl(match):
        delim = match.group(1)
        content = match.group(2)
        lines = content.split("\n")
        kept = []
        for line in lines:
            if line.lstrip().startswith("#"):
                continue
            kept.append(re.sub(r"\s+#\s.*$", "", line))
        return delim + "\n".join(kept) + delim

    return pattern.sub(repl, source)


def collapse_blank_lines(source: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", source).strip() + "\n"


def strip_file(path: Path):
    try:
        src = path.read_text(encoding="utf-8")
        cleaned = remove_docstrings(src)
        cleaned = strip_python_comments(cleaned)
        cleaned = strip_inside_triple_strings(cleaned)
        cleaned = collapse_blank_lines(cleaned)
        ast.parse(cleaned)
        path.write_text(cleaned, encoding="utf-8")
        return True, None
    except Exception as e:
        return False, str(e)


def strip_shell(path: Path):
    try:
        src = path.read_text(encoding="utf-8")
        lines = src.split("\n")
        kept = []
        for i, line in enumerate(lines):
            if i == 0 and line.startswith("#!"):
                kept.append(line)
                continue
            if line.lstrip().startswith("#"):
                continue
            kept.append(re.sub(r"\s+#\s.*$", "", line))
        cleaned = collapse_blank_lines("\n".join(kept))
        path.write_text(cleaned, encoding="utf-8")
        return True, None
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    targets = [Path(t) for t in sys.argv[1:]]
    if not targets:
        targets = [
            Path("kaggle/lib"),
            Path("kaggle/vast"),
            Path("kaggle"),
        ]

    files_py = []
    files_sh = []
    for t in targets:
        if t.is_file():
            if t.suffix == ".py":
                files_py.append(t)
            elif t.suffix == ".sh":
                files_sh.append(t)
        elif t.is_dir():
            for p in t.rglob("*.py"):
                if "__pycache__" in str(p) or "/sam3/" in str(p).replace("\\", "/"):
                    continue
                if p.name == "_strip_comments.py":
                    continue
                files_py.append(p)
            for p in t.rglob("*.sh"):
                files_sh.append(p)

    files_py = sorted(set(files_py))
    files_sh = sorted(set(files_sh))

    for p in files_py:
        ok, err = strip_file(p)
        status = "OK" if ok else f"FAIL: {err[:60]}"
        print(f"  [py] {p}: {status}")

    for p in files_sh:
        ok, err = strip_shell(p)
        status = "OK" if ok else f"FAIL: {err[:60]}"
        print(f"  [sh] {p}: {status}")

    print(f"\nDone: {len(files_py)} .py + {len(files_sh)} .sh files")
