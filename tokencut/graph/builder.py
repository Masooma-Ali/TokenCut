"""Member B — Code-graph construction."""

import builtins

from tree_sitter import Node

from tokencut.contracts import Chunk, ChunkKind, CodeGraph, Edge, EdgeKind
from tokencut.graph.modules import ModuleMap, node_text, parse_code, read_imports

BUILTIN_NAMES = frozenset(dir(builtins))

# One entry of a file's name table: ("module", dotted name) or ("chunk", chunk id).
Binding = tuple[str, str]


def build_graph(chunks: list[Chunk]) -> CodeGraph:
    """Build contains / imports / calls / inherits edges between chunks.

    Must:
      - never point an edge at a chunk id that does not exist
      - mark heuristic call/import guesses with resolved=False
      - handle circular imports without infinite loops
    """
    return _GraphBuilder(chunks).build()


def dependency_closure(graph: CodeGraph, chunk_ids: set[str], max_hops: int = 1) -> set[str]:
    """Return chunk_ids plus everything they depend on within max_hops."""
    from tokencut.graph.queries import GraphIndex  # local import: queries imports nothing from here

    return GraphIndex(graph).closure(chunk_ids, max_hops)


class _GraphBuilder:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = sorted(chunks, key=lambda c: (c.file_path, c.start_line, c.id))
        self.by_id = {c.id: c for c in self.chunks}
        self.modules = ModuleMap.build(self.chunks)
        self.edges: dict[tuple[str, str, EdgeKind], bool] = {}  # (source, target, kind) -> resolved
        self.name_tables: dict[str, dict[str, Binding]] = {}  # file path -> name table

        # (class id, method name) -> method id; first definition wins (property getter before setter)
        self.methods: dict[tuple[str, str], str] = {}
        # last name -> chunk ids, for DECISIONS #7 rule 4
        self.by_last_name: dict[str, list[str]] = {}
        self.classes_by_name: dict[str, list[str]] = {}
        for c in self.chunks:
            if c.kind == ChunkKind.METHOD and c.parent_id:
                self.methods.setdefault((c.parent_id, c.qualified_name.split(".")[-1]), c.id)
            if c.kind != ChunkKind.MODULE:
                last = c.qualified_name.split(".")[-1]
                self.by_last_name.setdefault(last, []).append(c.id)
                if c.kind == ChunkKind.CLASS:
                    self.classes_by_name.setdefault(last, []).append(c.id)
        self.bases: dict[str, list[str]] = {}  # class id -> resolved base class ids

    # ---------- edge helpers ----------

    def _add(self, source: str, target: str | None, kind: EdgeKind, resolved: bool = True) -> None:
        if (
            target is None
            or source == target
            or target not in self.by_id
            or source not in self.by_id
        ):
            return
        key = (source, target, kind)
        self.edges[key] = self.edges.get(key, False) or resolved

    def build(self) -> CodeGraph:
        for c in self.chunks:
            if c.kind == ChunkKind.MODULE:
                self._imports(c)
        for c in self.chunks:
            if c.kind == ChunkKind.METHOD and c.parent_id:
                self._add(c.parent_id, c.id, EdgeKind.CONTAINS)
            elif c.kind == ChunkKind.CLASS:
                self._inherits(c)
        for c in self.chunks:  # after INHERITS, because self.foo() may resolve into a base class
            if c.kind in (ChunkKind.FUNCTION, ChunkKind.METHOD):
                self._calls(c)

        edges = [
            Edge(source=s, target=t, kind=k, resolved=r)
            for (s, t, k), r in sorted(
                self.edges.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2].value)
            )
        ]
        return CodeGraph(chunk_ids=[c.id for c in self.chunks], edges=edges)

    # ---------- IMPORTS (plan B4) ----------

    def _imports(self, module: Chunk) -> None:
        table: dict[str, Binding] = {}
        self.name_tables[module.file_path] = table
        if not module.code.strip():
            return
        mm = self.modules
        package = mm.package_of(module.file_path)
        for imp in read_imports(parse_code(module.code), package):
            if not imp.module:
                continue
            if imp.name is None and not imp.wildcard:
                # import a.b  /  import a.b as z
                if mm.is_module(imp.module):
                    self._add(module.id, mm.module_chunk[imp.module], EdgeKind.IMPORTS)
                if imp.alias:
                    if mm.is_module(imp.module):
                        table[imp.alias] = ("module", imp.module)
                else:
                    first = imp.module.split(".")[0]
                    if mm.is_module(first):
                        table[first] = ("module", first)
            elif imp.wildcard:
                if mm.is_module(imp.module):
                    self._add(module.id, mm.module_chunk[imp.module], EdgeKind.IMPORTS)
            else:
                # from a.b import y [as z]
                local = imp.alias or imp.name
                submodule = f"{imp.module}.{imp.name}"
                if mm.is_module(submodule):
                    self._add(module.id, mm.module_chunk[submodule], EdgeKind.IMPORTS)
                    table[local] = ("module", submodule)
                elif mm.is_module(imp.module):
                    self._add(module.id, mm.module_chunk[imp.module], EdgeKind.IMPORTS)
                    target = mm.top_names[imp.module].get(imp.name)
                    if target is not None:
                        table[local] = ("chunk", target)
                # external module: ignored

    # ---------- name resolution (DECISIONS #7) ----------

    def _same_file_name(self, file_path: str, name: str) -> str | None:
        module = self.modules.file_module.get(file_path)
        if module is None:
            return None
        return self.modules.top_names[module].get(name)

    def _unique(self, name: str, candidates: dict[str, list[str]]) -> str | None:
        ids = candidates.get(name, [])
        return ids[0] if len(ids) == 1 else None

    def _resolve_name(
        self, file_path: str, name: str, classes_only: bool = False
    ) -> tuple[str | None, bool]:
        """Resolve a bare identifier. Returns (chunk id or None, resolved flag)."""
        binding = self.name_tables.get(file_path, {}).get(name)
        if binding is not None:  # rule 1: imported name
            if binding[0] == "chunk" and self._kind_ok(binding[1], classes_only):
                return binding[1], True
            return None, True  # a module, or the wrong kind: no edge, and no guessing
        local = self._same_file_name(file_path, name)  # rule 3: same file
        if local is not None and self._kind_ok(local, classes_only):
            return local, True
        if name in BUILTIN_NAMES:
            return None, True
        return self._rule4(name, classes_only)

    def _rule4(self, name: str, classes_only: bool = False) -> tuple[str | None, bool]:
        # rule 4: exactly one definition with this name in the whole repo -> guess, resolved=False
        target = self._unique(name, self.classes_by_name if classes_only else self.by_last_name)
        return target, False

    def _kind_ok(self, chunk_id: str, classes_only: bool) -> bool:
        return not classes_only or self.by_id[chunk_id].kind == ChunkKind.CLASS

    def _module_of_expr(self, file_path: str, node: Node) -> str | None:
        """If node is `alias` or `alias.sub` naming an imported repo module, return its dotted name."""
        if node.type == "identifier":
            binding = self.name_tables.get(file_path, {}).get(node_text(node))
            if binding is not None and binding[0] == "module":
                return binding[1]
            return None
        if node.type == "attribute":
            obj = node.child_by_field_name("object")
            attr = node.child_by_field_name("attribute")
            if obj is None or attr is None:
                return None
            parent = self._module_of_expr(file_path, obj)
            if parent is not None and self.modules.is_module(f"{parent}.{node_text(attr)}"):
                return f"{parent}.{node_text(attr)}"
        return None

    def _method_in_class_or_bases(self, class_id: str, name: str) -> str | None:
        target = self.methods.get((class_id, name))
        if target is not None:
            return target
        for base in self.bases.get(class_id, []):  # one level of base classes
            target = self.methods.get((base, name))
            if target is not None:
                return target
        return None

    def _class_binding(self, file_path: str, name: str) -> str | None:
        """Chunk id if `name` is a class visible in this file (imported or defined here)."""
        binding = self.name_tables.get(file_path, {}).get(name)
        if binding is not None:
            target = binding[1] if binding[0] == "chunk" else None
        else:
            target = self._same_file_name(file_path, name)
        if target is not None and self.by_id[target].kind == ChunkKind.CLASS:
            return target
        return None

    # ---------- INHERITS (plan B3) ----------

    def _inherits(self, cls: Chunk) -> None:
        # The class signature holds the bases; give it a body so tree-sitter sees a full class.
        root = parse_code(cls.signature + "\n    pass\n")
        definition = next((n for n in _walk(root) if n.type == "class_definition"), None)
        if definition is None:
            return
        superclasses = definition.child_by_field_name("superclasses")
        if superclasses is None:
            return
        resolved_bases = []
        for base in superclasses.named_children:
            if base.type == "keyword_argument":  # metaclass=...
                continue
            if base.type == "subscript":  # Base[T]
                base = base.child_by_field_name("value") or base
            target, resolved = self._resolve_base(cls.file_path, base)
            if target is not None:
                self._add(cls.id, target, EdgeKind.INHERITS, resolved)
                resolved_bases.append(target)
        self.bases[cls.id] = resolved_bases

    def _resolve_base(self, file_path: str, base: Node) -> tuple[str | None, bool]:
        if base.type == "identifier":
            return self._resolve_name(file_path, node_text(base), classes_only=True)
        if base.type == "attribute":
            obj = base.child_by_field_name("object")
            name = node_text(base.child_by_field_name("attribute"))
            module = self._module_of_expr(file_path, obj) if obj is not None else None
            if module is not None:
                target = self.modules.top_names[module].get(name)
                if target is not None and self._kind_ok(target, True):
                    return target, True
            return self._rule4(name, classes_only=True)
        return None, True

    # ---------- CALLS (plan B5) ----------

    def _calls(self, func: Chunk) -> None:
        root = parse_code(func.code)
        for node in _walk(root):
            if node.type != "call":
                continue
            callee = node.child_by_field_name("function")
            if callee is None:
                continue
            target, resolved = self._resolve_callee(func, callee)
            self._add(func.id, target, EdgeKind.CALLS, resolved)

    def _resolve_callee(self, func: Chunk, callee: Node) -> tuple[str | None, bool]:
        file_path = func.file_path
        if callee.type == "identifier":
            return self._resolve_name(file_path, node_text(callee))
        if callee.type != "attribute":
            return None, True  # e.g. f()() or x[0]()
        obj = callee.child_by_field_name("object")
        attr = callee.child_by_field_name("attribute")
        if obj is None or attr is None:
            return None, True
        name = node_text(attr)
        in_method = func.kind == ChunkKind.METHOD and func.parent_id is not None

        # rule 2: self.foo() / cls.foo() inside a method -> same class, then its base classes
        if in_method and obj.type == "identifier" and node_text(obj) in ("self", "cls"):
            target = self._method_in_class_or_bases(func.parent_id, name)
            if target is not None:
                return target, True
        # super().foo() -> base classes only
        elif in_method and obj.type == "call" and _is_super_call(obj):
            for base in self.bases.get(func.parent_id, []):
                target = self.methods.get((base, name))
                if target is not None:
                    return target, True
        else:
            # rule 1: module_alias.foo()
            module = self._module_of_expr(file_path, obj)
            if module is not None:
                target = self.modules.top_names[module].get(name)
                if target is not None:
                    return target, True
            # rule 1 / 3: SomeClass.method() where the class is imported or defined in this file
            elif obj.type == "identifier":
                class_id = self._class_binding(file_path, node_text(obj))
                if class_id is not None:
                    target = self._method_in_class_or_bases(class_id, name)
                    if target is not None:
                        return target, True
        return self._rule4(name)


def _is_super_call(node: Node) -> bool:
    fn = node.child_by_field_name("function")
    return fn is not None and fn.type == "identifier" and node_text(fn) == "super"


def _walk(root: Node):
    """All nodes under root in source order (explicit stack: no recursion limit on deep code)."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))
