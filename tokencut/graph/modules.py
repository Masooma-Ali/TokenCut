"""File path <-> dotted module name, and reading import statements (IMPLEMENTATION_PLAN B2, B4)."""

import textwrap
from dataclasses import dataclass, field

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from tokencut.contracts import Chunk, ChunkKind

_PARSER = Parser(Language(tspython.language()))

# Source-layout folders that are not part of the import name (pytest uses src/_pytest/...).
LAYOUT_PREFIXES = ("src", "lib")


def parse_code(code: str) -> Node:
    """Parse a chunk's code with tree-sitter. Methods are indented, so dedent first."""
    return _PARSER.parse(textwrap.dedent(code).encode("utf-8")).root_node


def node_text(node: Node) -> str:
    return node.text.decode("utf-8", errors="replace")


def module_name_for(file_path: str) -> str:
    """'shop/cart.py' -> 'shop.cart'; 'shop/payments/__init__.py' -> 'shop.payments'."""
    parts = (
        file_path[: -len(".py")].split("/") if file_path.endswith(".py") else file_path.split("/")
    )
    if parts and parts[-1] == "__init__" and len(parts) > 1:
        parts = parts[:-1]
    return ".".join(parts)


def is_package_file(file_path: str) -> bool:
    return file_path.endswith("/__init__.py") or file_path == "__init__.py"


@dataclass
class ModuleMap:
    """Lookup tables over all module chunks of the repo."""

    module_chunk: dict[str, str] = field(default_factory=dict)  # module name -> module chunk id
    file_module: dict[str, str] = field(default_factory=dict)  # file path -> primary module name
    top_names: dict[str, dict[str, str]] = field(
        default_factory=dict
    )  # module -> {name -> chunk id}

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "ModuleMap":
        mm = cls()
        files = sorted({c.file_path for c in chunks if c.kind == ChunkKind.MODULE})
        names_by_file: dict[str, dict[str, str]] = {f: {} for f in files}
        for chunk in chunks:  # chunks are in line order, so a later definition wins (as in Python)
            if (
                chunk.kind in (ChunkKind.FUNCTION, ChunkKind.CLASS)
                and chunk.file_path in names_by_file
            ):
                names_by_file[chunk.file_path][chunk.qualified_name] = chunk.id

        # Primary names first, so a 'src/'-less alias never hides a real module of that name.
        for file_path in files:
            name = module_name_for(file_path)
            mm.file_module[file_path] = name
            mm.module_chunk[name] = f"{file_path}::<module>"
            mm.top_names[name] = names_by_file[file_path]
        for file_path in files:
            name = mm.file_module[file_path]
            first, _, rest = name.partition(".")
            if first in LAYOUT_PREFIXES and rest and rest not in mm.module_chunk:
                mm.module_chunk[rest] = mm.module_chunk[name]
                mm.top_names[rest] = mm.top_names[name]
        return mm

    def is_module(self, name: str) -> bool:
        return name in self.module_chunk

    def package_of(self, file_path: str) -> str:
        """The package that relative imports in this file are resolved against."""
        name = self.file_module[file_path]
        if is_package_file(file_path):
            return name
        return name.rpartition(".")[0]


@dataclass
class ImportedName:
    """One name bound by an import statement."""

    module: str  # absolute dotted module the statement refers to ('' if unresolvable)
    name: str | None  # imported name for 'from m import name'; None for 'import m'
    alias: str | None  # local binding if 'as' was used
    wildcard: bool = False


def read_imports(module_root: Node, package: str) -> list[ImportedName]:
    """Collect import statements anywhere in a module chunk (e.g. also inside try/if blocks)."""
    found: list[ImportedName] = []
    stack = [module_root]
    while stack:
        node = stack.pop()
        if node.type == "import_statement":
            for child in node.children_by_field_name("name"):
                dotted, alias = _name_and_alias(child)
                found.append(ImportedName(module=dotted, name=None, alias=alias))
        elif node.type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            base = _resolve_from_module(module_node, package) if module_node is not None else ""
            names = node.children_by_field_name("name")
            if any(c.type == "wildcard_import" for c in node.children):
                found.append(ImportedName(module=base, name=None, alias=None, wildcard=True))
            for child in names:
                dotted, alias = _name_and_alias(child)
                found.append(ImportedName(module=base, name=dotted, alias=alias))
        else:
            stack.extend(reversed(node.children))
    return found


def _name_and_alias(node: Node) -> tuple[str, str | None]:
    if node.type == "aliased_import":
        name_node = node.child_by_field_name("name")
        alias_node = node.child_by_field_name("alias")
        return (
            node_text(name_node) if name_node else "",
            node_text(alias_node) if alias_node else None,
        )
    return node_text(node), None


def _resolve_from_module(module_node: Node, package: str) -> str:
    """'from ..a.b import x' inside package 'p.q.r' -> 'p.q.a.b'. '' if it escapes the repo."""
    if module_node.type != "relative_import":
        return node_text(module_node)
    level = 0
    rest = ""
    for child in module_node.children:
        if child.type == "import_prefix":
            level = node_text(child).count(".")
        elif child.type == "dotted_name":
            rest = node_text(child)
    parts = package.split(".") if package else []
    if level - 1 > len(parts):
        return ""
    base_parts = parts[: len(parts) - (level - 1)]
    if rest:
        base_parts = base_parts + rest.split(".")
    return ".".join(base_parts)
