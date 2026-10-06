"""Member A — ingestion: walker, Python chunker, parse_repo."""

import subprocess

from tokencut.contracts import MAX_FILE_BYTES, ChunkKind
from tokencut.ingestion.parser import parse_repo
from tokencut.ingestion.python_parser import PythonParser
from tokencut.ingestion.walker import list_python_files
from tokencut.tokens import count_tokens


def by_id(chunks):
    return {c.id: c for c in chunks}


# ---------- sample repo ----------


def test_sample_repo_matches_expected(sample_repo_path, expected_chunks):
    chunks = parse_repo(sample_repo_path)
    assert [c.id for c in chunks] == [c.id for c in expected_chunks]
    for got, want in zip(chunks, expected_chunks):
        assert got.kind == want.kind
        assert (got.start_line, got.end_line) == (want.start_line, want.end_line)
        assert got.signature == want.signature
        assert got.parent_id == want.parent_id
        assert got.code == want.code
        assert got.token_count == want.token_count


def test_sample_repo_has_30_chunks_and_skips_non_python(sample_repo_path):
    chunks = parse_repo(sample_repo_path)
    assert len(chunks) == 30
    files = {c.file_path for c in chunks}
    assert "README.md" not in files and "config.yaml" not in files and "shop/logo.bin" not in files


def test_every_file_has_exactly_one_module_chunk(sample_repo_path):
    chunks = parse_repo(sample_repo_path)
    files = {c.file_path for c in chunks}
    for f in files:
        modules = [c for c in chunks if c.file_path == f and c.kind == ChunkKind.MODULE]
        assert len(modules) == 1
        assert modules[0].id == f"{f}::<module>"


def test_broken_file_does_not_crash(sample_repo_path):
    chunks = by_id(parse_repo(sample_repo_path))
    assert "shop/utils/broken.py::fine_function" in chunks
    assert "shop/utils/broken.py::another_fine_function" in chunks
    module = chunks["shop/utils/broken.py::<module>"]
    assert "def broken_function(" in module.code  # the ERROR region lands in the module chunk
    assert 'return "missing closing parenthesis"' in module.code


def test_empty_file_gives_empty_module_chunk(sample_repo_path):
    chunk = by_id(parse_repo(sample_repo_path))["shop/__init__.py::<module>"]
    assert chunk.code == "" and chunk.token_count == 0
    assert chunk.start_line == chunk.end_line == 1


def test_methods_have_parent_and_qualified_name(sample_repo_path):
    chunks = by_id(parse_repo(sample_repo_path))
    total = chunks["shop/cart.py::Cart.total"]
    assert total.kind == ChunkKind.METHOD
    assert total.parent_id == "shop/cart.py::Cart"
    assert total.qualified_name == "Cart.total"
    assert total.signature == "def total(self) -> float:"
    assert total.code.startswith("    def total")  # full lines, original indentation


def test_paths_use_forward_slashes(sample_repo_path):
    for chunk in parse_repo(sample_repo_path):
        assert "\\" not in chunk.file_path
        assert chunk.id.startswith(chunk.file_path + "::")


def test_parse_repo_is_deterministic(sample_repo_path):
    first = [c.model_dump() for c in parse_repo(sample_repo_path)]
    second = [c.model_dump() for c in parse_repo(sample_repo_path)]
    assert first == second


def test_output_is_sorted(sample_repo_path):
    chunks = parse_repo(sample_repo_path)
    keys = [(c.file_path, c.start_line, c.id) for c in chunks]
    assert keys == sorted(keys)


def test_token_counts_match_code(sample_repo_path):
    for chunk in parse_repo(sample_repo_path):
        assert chunk.token_count == count_tokens(chunk.code)


# ---------- chunker edge cases ----------

CLASS_SOURCE = b'''import os


@dataclass
class Shape(Base):
    """A shape."""

    sides = 0

    @property
    def area(self):
        """Area in m2."""
        return 0

    @area.setter
    def area(self, value):
        pass

    async def fetch(self):
        def helper():
            return 1
        return helper()

    colour = "red"


def outer():
    class Inner:
        pass
    return Inner
'''


def parse(source: bytes, path: str = "pkg/mod.py"):
    return by_id(PythonParser().parse_file(path, source))


def test_class_chunk_excludes_methods_no_double_counting():
    chunks = parse(CLASS_SOURCE)
    shape = chunks["pkg/mod.py::Shape"]
    assert shape.kind == ChunkKind.CLASS
    assert "sides = 0" in shape.code and 'colour = "red"' in shape.code
    assert "def area" not in shape.code and "def fetch" not in shape.code
    assert shape.start_line == 4  # starts at the decorator
    # method bodies are counted once, in the method chunk only
    assert [cid for cid, c in chunks.items() if "return 0" in c.code] == ["pkg/mod.py::Shape.area"]
    total_lines = sum(len(c.code.split("\n")) for c in chunks.values() if c.code)
    assert total_lines <= len(CLASS_SOURCE.decode().split("\n"))


def test_duplicate_names_get_suffix():
    chunks = parse(CLASS_SOURCE)
    getter = chunks["pkg/mod.py::Shape.area"]
    setter = chunks["pkg/mod.py::Shape.area#2"]
    assert getter.qualified_name == setter.qualified_name == "Shape.area"
    assert getter.start_line < setter.start_line
    assert getter.signature == "@property\n    def area(self):"
    assert setter.parent_id == "pkg/mod.py::Shape"


def test_docstrings_and_async_and_nested_definitions():
    chunks = parse(CLASS_SOURCE)
    assert chunks["pkg/mod.py::Shape"].docstring == "A shape."
    assert chunks["pkg/mod.py::Shape.area"].docstring == "Area in m2."
    fetch = chunks["pkg/mod.py::Shape.fetch"]
    assert fetch.signature == "async def fetch(self):"
    assert "def helper" in fetch.code  # nested function stays inside its parent
    assert "pkg/mod.py::helper" not in chunks
    assert "class Inner" in chunks["pkg/mod.py::outer"].code
    assert "pkg/mod.py::Inner" not in chunks


def test_repeated_class_name_suffixes_methods_too():
    source = b"class A:\n    def f(self): pass\n\n\nclass A:\n    def f(self): pass\n"
    chunks = parse(source)
    assert "pkg/mod.py::A#2" in chunks
    assert chunks["pkg/mod.py::A.f#2"].parent_id == "pkg/mod.py::A#2"


def test_module_chunk_collects_top_level_statements():
    source = b'"""Doc."""\nimport os\n\n\ndef f():\n    pass\n\n\nX = 1\nif __name__ == "__main__":\n    f()\n'
    module = parse(source)["pkg/mod.py::<module>"]
    assert module.code == '"""Doc."""\nimport os\nX = 1\nif __name__ == "__main__":\n    f()'
    assert (module.start_line, module.end_line) == (1, 11)


def test_crlf_source_gives_same_code_as_lf():
    lf = parse(CLASS_SOURCE)
    crlf = parse(CLASS_SOURCE.replace(b"\n", b"\r\n"))
    assert [c.code for c in lf.values()] == [c.code for c in crlf.values()]
    assert [c.signature for c in lf.values()] == [c.signature for c in crlf.values()]


def test_non_utf8_file_is_parsed():
    source = "def caf\u00e9():\n    return '\u00e9'\n".encode("latin-1")
    chunks = parse(source)
    assert any(c.kind == ChunkKind.FUNCTION for c in chunks.values())


def test_garbage_file_does_not_raise():
    chunks = parse(b"))))(((( def def class :::: \x01\x02\n")
    assert "pkg/mod.py::<module>" in chunks


# ---------- walker ----------


def write(path, text="x = 1\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")


def test_walker_skips_large_binary_and_ignored_dirs(tmp_path):
    write(tmp_path / "app/main.py")
    write(tmp_path / "app/tests/test_main.py")  # tests/ folders are kept (DECISIONS #12)
    write(tmp_path / "app/migrations/0001.py")
    write(tmp_path / "vendor/lib.py")
    write(tmp_path / "node_modules/x.py")
    write(tmp_path / "app/big.py", "x = 1\n" * (MAX_FILE_BYTES // 6 + 10))
    write(tmp_path / "app/nul.py", b"x = 1\0\n")
    write(tmp_path / "notes.txt")
    assert list_python_files(tmp_path) == ["app/main.py", "app/tests/test_main.py"]


def test_walker_respects_gitignore(tmp_path):
    write(tmp_path / "keep.py")
    write(tmp_path / "build/generated.py")
    write(tmp_path / ".gitignore", "build/\n")
    write(tmp_path / "untracked.py")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "keep.py", ".gitignore"], cwd=tmp_path, check=True)
    assert list_python_files(tmp_path) == ["keep.py", "untracked.py"]


def test_parse_repo_on_folder_without_python(tmp_path):
    write(tmp_path / "README.md", "hi")
    assert parse_repo(tmp_path) == []
