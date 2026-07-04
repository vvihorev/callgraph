#!/usr/bin/env python3
"""
callgraph.py - Interactive call-graph viewer for Python modules.

Usage:
    python callgraph.py path/to/module.py       # open the interactive viewer
    python callgraph.py --dump path/to/module.py  # print parsed structure (no GUI)

The viewer parses a Python module (and, best-effort, the local modules it
imports), extracts every function and every call each function makes, and
renders an interactive, layered graph.

Model
-----
Module   : a .py file. Defines Functions. Drawn as a titled frame.
Function : a def (or method). Drawn as a node listing the Calls it makes.
Call     : a call site inside a function body (or module-level code).
           Links two Function nodes when the callee can be resolved.

Two views
---------
Module overview (default / press Home):
    Column 0  = the module frame containing the module-level code node
                plus "entrypoint" functions (defined here, never called
                internally). Functions called from module level are omitted
                from column 0.
    Column 1  = functions defined in the module that ARE called from the
                module itself (second order).
    Column 2+ = third-order functions: from other modules, or deeper in the
                call stack of this module.

Function focus (click any call in any node):
    The clicked node's function moves to the top-left of column 0, the calls
    made from its body become the nodes of column 1, the calls made from
    those become column 2, and so on.

Node display styles (toggle with C, works in either view above):
    Call list (default) : each node lists the calls its function makes.
    Source code         : each node shows the function's full source; the
                          calls it makes are highlighted inline and are
                          themselves clickable (same re-root behaviour).

Controls
--------
    Left click a node/call : focus that function (re-root the graph)
    Left drag              : pan
    Mouse wheel            : zoom to cursor
    C                      : toggle call-list / source-code nodes
    Right click / Backspace: back
    Home                   : return to module overview
    F                      : fit everything on screen
    Esc                    : quit
"""

import ast
import os
import sys
import math
from collections import deque, defaultdict

# raylib is only needed for the GUI. Guard the import so the parser/layout can
# be exercised head-less (e.g. with --dump, or in tests).
try:
    import pyray as pr
except Exception:  # pragma: no cover - depends on local install
    pr = None


# --------------------------------------------------------------------------- #
#  Data model: Module / Function / Call
# --------------------------------------------------------------------------- #
class Call:
    """A single call site found inside a function body or module-level code."""

    def __init__(self, raw_name, span, callee=None):
        # span = (lineno, col_offset, end_lineno, end_col_offset) of the callee
        # expression (the "func" part of the ast.Call, e.g. "helpers.render").
        # Used both for the call list and to highlight/click the call in the
        # source-code view.
        self.raw_name = raw_name      # textual name at the call site, e.g. "helpers.render"
        self.lineno = span[0]
        self.col_offset = span[1]
        self.end_lineno = span[2]
        self.end_col_offset = span[3]
        self.callee = callee          # resolved Function, or None if unresolved
        self.resolved = callee is not None


class Function:
    """A function/method definition (or the synthetic module-level code node)."""

    def __init__(self, qualname, module, node=None, external=False, kind="function"):
        self.qualname = qualname      # display name, e.g. "process" or "Cls.method"
        self.module = module
        self.node = node              # ast node (None for stubs / module node)
        self.external = external      # True when we have no source for it
        self.kind = kind              # "function" | "module" | "class"
        self.calls = []               # list[Call], filled during resolution

        # transient layout fields (reset every time we re-layout)
        self.col = None
        self.x = 0.0
        self.y = 0.0
        self.w = NODE_W
        self.h = 0.0
        self.visible = False
        # code-view geometry (set by layout_node when code view is active)
        self.code_base = None      # absolute lineno of first shown source row
        self.code_rows = None      # list[(abs_lineno, raw_text)]
        self.code_overflow = 0     # hidden trailing lines
        self.code_nlines = 0       # number of shown source rows

    def __repr__(self):
        return "<Function %s.%s>" % (self.module.name, self.qualname)


class Module:
    """A python module (file), or an external/unresolved module placeholder."""

    def __init__(self, name, path=None, external=False):
        self.name = name
        self.path = path
        self.external = external
        self.functions = {}   # qualname -> Function
        self.module_node = None   # synthetic Function for top-level code
        self.source_lines = None  # list[str], the file's source (for the code view)
        self._scan = None         # ModuleScan (imports + raw calls), for resolution
        self._resolved = False

    def get_or_create(self, qualname, node=None, external=False, kind="function"):
        f = self.functions.get(qualname)
        if f is None:
            f = Function(qualname, self, node=node, external=external, kind=kind)
            self.functions[qualname] = f
        return f


# --------------------------------------------------------------------------- #
#  AST scanning: collect defs, imports and raw call sites per scope
# --------------------------------------------------------------------------- #
class ModuleScan(ast.NodeVisitor):
    """Walks a module AST, recording:
        - defined functions/methods (qualname + node)
        - imports (aliases and from-imports)
        - the raw call names made in each scope (per function, and module level)
    Resolution of those raw names to Functions happens later in Project.
    """

    MODULE_SCOPE = "<module>"

    def __init__(self):
        self.class_stack = []
        self.func_stack = []
        self.defined = []                 # list[(qualname, node)]
        self.classes = set()              # class qualnames
        self.calls = {self.MODULE_SCOPE: []}   # scope -> list[(raw_name, lineno)]
        self.imports_alias = {}           # bound name -> module fullname
        self.imports_from = {}            # local name -> (level, module, orig)

    # --- scope helpers ---------------------------------------------------- #
    def _scope(self):
        return self.func_stack[-1] if self.func_stack else self.MODULE_SCOPE

    def _qual(self, name):
        if self.class_stack:
            return ".".join(self.class_stack + [name])
        return name

    # --- imports ---------------------------------------------------------- #
    def visit_Import(self, node):
        for a in node.names:
            if a.asname:
                self.imports_alias[a.asname] = a.name
            else:
                # "import a.b.c" binds "a", but a.b.c.x also works -> map both
                self.imports_alias[a.name.split(".")[0]] = a.name.split(".")[0]
                self.imports_alias[a.name] = a.name

    def visit_ImportFrom(self, node):
        mod = node.module or ""
        level = node.level or 0
        for a in node.names:
            if a.name == "*":
                continue
            local = a.asname or a.name
            self.imports_from[local] = (level, mod, a.name)

    # --- definitions ------------------------------------------------------ #
    def visit_ClassDef(self, node):
        q = self._qual(node.name)
        self.classes.add(q)
        self.defined.append((q, node))
        self.calls.setdefault(q, [])
        self.class_stack.append(node.name)
        for c in node.body:
            self.visit(c)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        q = self._qual(node.name)
        self.defined.append((q, node))
        self.calls.setdefault(q, [])
        self.func_stack.append(q)
        saved = self.class_stack
        self.class_stack = []            # nested defs are not class-qualified
        for c in node.body:
            self.visit(c)
        self.class_stack = saved
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    # --- calls ------------------------------------------------------------ #
    def visit_Call(self, node):
        raw = self._call_name(node.func)
        if raw is not None:
            fn = node.func
            ln = getattr(fn, "lineno", getattr(node, "lineno", 0))
            span = (
                ln,
                getattr(fn, "col_offset", 0),
                getattr(fn, "end_lineno", ln),
                getattr(fn, "end_col_offset", 0),
            )
            self.calls[self._scope()].append((raw, span))
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    def _call_name(self, func):
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            base = self._call_name(func.value)
            return (base + "." + func.attr) if base else func.attr
        return None


# --------------------------------------------------------------------------- #
#  Project: owns all modules, scans + resolves them
# --------------------------------------------------------------------------- #
class Project:
    def __init__(self, entry_path):
        self.entry_path = os.path.abspath(entry_path)
        self.modules = {}          # key -> Module  (key = abspath for local, name for external)
        self.by_display = {}       # display name -> Module (first wins, informational)
        self.main = None
        self._pending = deque()    # modules scanned but not yet resolved
        self.max_modules = 40      # safety cap on how many files we follow

    # ---- public entry ---------------------------------------------------- #
    def build(self):
        self.main = self._scan_file(self.entry_path)
        # Resolve every scanned module; resolution may pull in more modules.
        while self._pending:
            m = self._pending.popleft()
            self._resolve_module(m)
        return self

    # ---- scanning -------------------------------------------------------- #
    def _scan_file(self, path):
        path = os.path.abspath(path)
        if path in self.modules:
            return self.modules[path]
        name = os.path.splitext(os.path.basename(path))[0]
        module = Module(name, path=path)
        self.modules[path] = module
        self.by_display.setdefault(name, module)

        try:
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
            tree = ast.parse(src, filename=path)
        except (OSError, SyntaxError) as exc:
            sys.stderr.write("warning: could not parse %s (%s)\n" % (path, exc))
            module.external = True
            return module

        module.source_lines = src.splitlines()
        scan = ModuleScan()
        scan.visit(tree)
        module._scan = scan

        # Create Function objects for every defined name up front so that
        # cross-references can find them before resolution runs.
        for qual, node in scan.defined:
            kind = "class" if qual in scan.classes else "function"
            module.get_or_create(qual, node=node, kind=kind)
        module.module_node = Function("<module>", module, kind="module")
        module.functions[module.module_node.qualname] = module.module_node

        self._pending.append(module)
        return module

    def _external_module(self, name):
        key = "ext:" + (name or "<external>")
        m = self.modules.get(key)
        if m is None:
            m = Module(name or "<external>", external=True)
            self.modules[key] = m
        return m

    # ---- resolution ------------------------------------------------------ #
    def _resolve_module(self, module):
        if module._resolved or module._scan is None:
            return
        module._resolved = True
        scan = module._scan
        for scope, raw_calls in scan.calls.items():
            if scope == ModuleScan.MODULE_SCOPE:
                caller = module.module_node
            else:
                caller = module.functions.get(scope)
            if caller is None:
                continue
            for raw, span in raw_calls:
                callee = self._resolve_name(module, scan, scope, raw)
                caller.calls.append(Call(raw, span, callee))

    def _resolve_name(self, module, scan, scope, raw):
        """Best-effort resolution of a raw call name to a Function."""
        if "." not in raw:
            name = raw
            f = module.functions.get(name)
            if f is not None and f.kind != "module":
                return f
            if name in scan.imports_from:
                level, mod, orig = scan.imports_from[name]
                return self._resolve_cross(module, level, mod, orig)
            if name in scan.classes:
                init = name + ".__init__"
                if init in module.functions:
                    return module.functions[init]
                return module.functions.get(name)  # the class node itself
            return None

        parts = raw.split(".")
        base, last = parts[0], parts[-1]

        # self / cls method call inside a method
        if base in ("self", "cls") and "." in scope:
            cls = scope.rsplit(".", 1)[0]
            q = cls + "." + last
            if q in module.functions:
                return module.functions[q]
            return None

        # longest matching import alias prefix -> module.func
        for k in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:k])
            if prefix in scan.imports_alias:
                modfull = scan.imports_alias[prefix]
                return self._resolve_cross(module, 0, modfull, last)

        # from-import of an object, then attribute call on it
        if base in scan.imports_from:
            level, mod, orig = scan.imports_from[base]
            return self._resolve_cross(module, level, mod, last)

        return None

    def _resolve_cross(self, importer, level, modfull, funcname):
        """Resolve funcname inside module `modfull` (relative level respected).
        Follows local files when found; otherwise creates an external stub."""
        path = self._module_path(importer, level, modfull)
        if path and len(self.modules) < self.max_modules:
            target = self._scan_file(path)
            f = target.functions.get(funcname)
            if f is not None:
                return f
            # defined-but-unknown: fall through to external stub named by module
        ext = self._external_module(modfull or (importer.name if level else ""))
        return ext.get_or_create(funcname, external=True)

    def _module_path(self, importer, level, modfull):
        if importer.path is None:
            return None
        base = os.path.dirname(importer.path)
        if level and level > 0:
            for _ in range(level - 1):
                base = os.path.dirname(base)
        parts = modfull.split(".") if modfull else []
        cand = os.path.join(base, *parts) if parts else base
        if os.path.isfile(cand + ".py"):
            return cand + ".py"
        init = os.path.join(cand, "__init__.py")
        if os.path.isdir(cand) and os.path.isfile(init):
            return init
        return None

    # ---- analysis helpers ------------------------------------------------ #
    def incoming_counts(self):
        """Number of (non-self) incoming resolved edges per Function."""
        counts = defaultdict(int)
        for m in self.modules.values():
            for f in m.functions.values():
                for call in f.calls:
                    if call.callee is not None and call.callee is not f:
                        counts[call.callee] += 1
        return counts

    def entrypoints(self):
        """Main-module functions never called from anywhere in the project."""
        counts = self.incoming_counts()
        eps = []
        for f in self.main.functions.values():
            if f.kind == "module":
                continue
            if counts.get(f, 0) == 0:
                eps.append(f)
        eps.sort(key=lambda f: f.qualname)
        return eps

    def callers_of(self, func):
        """Every parsed function that makes a resolved call to `func`."""
        res = []
        for m in self.modules.values():
            if m.external:
                continue
            for f in m.functions.values():
                if f is func:
                    continue
                if any(c.callee is func for c in f.calls):
                    res.append(f)
        return res


# --------------------------------------------------------------------------- #
#  Layout: assign each visible function a column (BFS depth) and coordinates.
# --------------------------------------------------------------------------- #
NODE_W = 250.0
HEADER_H = 30.0
ROW_H = 20.0
PAD_TOP = 6.0
PAD_BOT = 8.0
V_GAP = 26.0
COL_STRIDE = NODE_W + 150.0
H_GAP = 150.0          # horizontal gap between columns (list mode == old COL_STRIDE)
NODE_X0 = 40.0
FRAME_PAD = 18.0
FRAME_HEADER = 28.0
LANE_GAP = 44.0
MAX_ROWS = 16

TITLE_FS = 16
ROW_FS = 13
FRAME_FS = 18

# --- source-code view metrics --------------------------------------------- #
CODE_FS = 13
CODE_LINE_H = 16.0
CODE_PAD_X = 10.0
CODE_PAD_TOP = 8.0
CODE_PAD_BOT = 10.0
CODE_GUTTER_GAP = 10.0     # gap between line-number gutter and the code text
CODE_TEXT_PAD = 16.0       # slack added to the right of the widest line
CODE_NODE_MIN_W = 280.0
CODE_NODE_MAX_W = 640.0
MAX_CODE_LINES = 46        # cap a node's shown source lines (rest -> "+N more")

MODULE_PALETTE = [
    (74, 144, 226),    # blue  - main module
    (80, 170, 120),    # green
    (200, 130, 60),    # orange
    (150, 110, 200),   # purple
    (200, 90, 120),    # pink
    (90, 170, 190),    # teal
    (170, 160, 70),    # olive
]


def displayed_rows(f):
    """How many call rows are shown (capped) and whether there is overflow."""
    n = min(len(f.calls), MAX_ROWS)
    overflow = len(f.calls) - n
    return n, overflow


def node_height(f):
    n, overflow = displayed_rows(f)
    extra = ROW_H if overflow > 0 else 0.0
    body = n * ROW_H + extra
    if len(f.calls) == 0:
        body = ROW_H  # keep a minimum body so leaf nodes aren't paper-thin
    return HEADER_H + PAD_TOP + body + PAD_BOT


# --- code-view helpers ---------------------------------------------------- #
def _clampf(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# Active font for on-screen text. None => raylib's built-in default font.
# To use a custom font, load it in Viewer.run() (see the marked hook there) and
# assign it here. Measuring (_text_w) and drawing (_draw_text) both go through
# this single font, so code-view call-token highlights stay aligned with the
# rendered text no matter which font is active.
_FONT = None
_SPACING = 1.0


def _text_w(s, size):
    """Pixel width of a string in the active font (estimate when head-less)."""
    if pr is None:
        return float(len(s)) * size * 0.55
    if _FONT is None:
        return float(pr.measure_text(s, size))
    return float(pr.measure_text_ex(_FONT, s, float(size), _SPACING).x)


def _draw_text(s, x, y, size, color):
    """Draw text in the active font, consistently with _text_w so measured
    widths match rendered widths (critical for code-view highlight alignment)."""
    if _FONT is None:
        pr.draw_text(s, int(x), int(y), int(size), color)
    else:
        pr.draw_text_ex(_FONT, s, pr.Vector2(float(x), float(y)),
                        float(size), _SPACING, color)


def function_span(f):
    """1-based inclusive (start, end) source line range for a Function, or None.

    Decorators are included so the shown snippet matches the file.
    """
    node = f.node
    if node is None or f.module.source_lines is None:
        return None
    start = node.lineno
    for d in getattr(node, "decorator_list", []):
        start = min(start, getattr(d, "lineno", start))
    end = getattr(node, "end_lineno", node.lineno)
    return (start, end)


def code_lines_for(f, max_lines=MAX_CODE_LINES):
    """Return (base_lineno, [(abs_lineno, raw_text), ...], overflow_count).

    For a function, that's its def..end span; for the synthetic module node it
    is the whole file (capped). base_lineno is the absolute line of the first
    shown row, so a call's screen row is (call.lineno - base_lineno).
    """
    src = f.module.source_lines
    if src is None:
        return (1, [], 0)
    if f.kind == "module":
        start, end = 1, len(src)
    else:
        span = function_span(f)
        if span is None:
            return (1, [], 0)
        start, end = span
    rows = []
    for ln in range(start, end + 1):
        if 1 <= ln <= len(src):
            rows.append((ln, src[ln - 1]))
    overflow = 0
    if len(rows) > max_lines:
        overflow = len(rows) - max_lines
        rows = rows[:max_lines]
    return (start, rows, overflow)


def layout_node(f, node_mode):
    """Compute (width, height) for a node in the active display mode, one of
    "calls" (call list), "code" (source lines), or "compact" (title only).

    In code view this also stashes the resolved source rows on the function so
    drawing/edge-anchoring/hit-testing all agree on the geometry.
    """
    if node_mode != "code":
        f.code_base = None
        f.code_rows = None
        f.code_overflow = 0
        f.code_nlines = 0
        if node_mode == "compact":
            label = (f.qualname if f.kind != "module"
                     else "<module> " + f.module.name)
            w = _clampf(_text_w(label, TITLE_FS) + 24.0, 140.0, CODE_NODE_MAX_W)
            return w, HEADER_H
        return NODE_W, node_height(f)

    base, rows, overflow = code_lines_for(f)
    f.code_base = base
    f.code_rows = rows
    f.code_overflow = overflow
    f.code_nlines = len(rows)

    max_line_w = 0.0
    for _, text in rows:
        w = _text_w(text.expandtabs(4), CODE_FS)
        if w > max_line_w:
            max_line_w = w
    last_no = rows[-1][0] if rows else base
    gutter_w = _text_w(str(last_no), CODE_FS)

    inner = CODE_PAD_X * 2 + gutter_w + CODE_GUTTER_GAP + max_line_w + CODE_TEXT_PAD
    w = _clampf(inner, CODE_NODE_MIN_W, CODE_NODE_MAX_W)

    nshown = len(rows) + (1 if overflow else 0)
    if nshown == 0:
        nshown = 1
    h = HEADER_H + CODE_PAD_TOP + nshown * CODE_LINE_H + CODE_PAD_BOT
    return w, h


def _reach_and_adj(sources):
    """Forward-reachable set plus deduped callee adjacency, from the sources."""
    reach = set()
    adj = {}
    stack = list(sources)
    while stack:
        f = stack.pop()
        if f in reach:
            continue
        reach.add(f)
        outs, seen = [], set()
        for call in f.calls:
            g = call.callee
            if g is not None and g is not f and g not in seen:
                seen.add(g)
                outs.append(g)
                if g not in reach:
                    stack.append(g)
        adj[f] = outs
    for f in reach:          # leaves
        adj.setdefault(f, [])
    return reach, adj


def _back_edges(sources, reach, adj):
    """Edges that close a cycle (point at a node still open on the DFS stack).

    These are the only edges that cannot be drawn strictly left-to-right, so we
    exclude them from column assignment; every remaining edge then goes forward.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    color = {f: WHITE for f in reach}
    back = set()
    for s in sources:
        if color[s] != WHITE:
            continue
        color[s] = GREY
        st = [(s, iter(adj[s]))]
        while st:
            u, it = st[-1]
            pushed = False
            for v in it:
                if color[v] == GREY:          # v is an ancestor -> cycle
                    back.add((u, v))
                elif color[v] == WHITE:
                    color[v] = GREY
                    st.append((v, iter(adj[v])))
                    pushed = True
                    break
                # BLACK -> forward/cross edge, keep it
            if not pushed:
                color[u] = BLACK
                st.pop()
    return back


def assign_columns(sources):
    """Longest-path layering.

    Each node is placed one column past its *deepest* caller, so for every call
    edge caller->callee we get col(callee) > col(caller): no edge ever points
    into its own column or an earlier one. The sole exception is recursion --
    genuine cycles cannot be drawn strictly forward, so those back edges are
    detected and left out of the layering (they still get drawn, just backward).
    """
    reach, adj = _reach_and_adj(sources)
    back = _back_edges(sources, reach, adj)

    dag = {f: [v for v in adj[f] if (f, v) not in back] for f in reach}
    indeg = {f: 0 for f in reach}
    for f in reach:
        for v in dag[f]:
            indeg[v] += 1

    col = {}
    dq = deque()
    for f in reach:
        if indeg[f] == 0:                 # sources + any cycle-freed roots
            col[f] = 0
            dq.append(f)
    while dq:                             # Kahn topo order == longest path
        u = dq.popleft()
        for v in dag[u]:
            if col[u] + 1 > col.get(v, -1):
                col[v] = col[u] + 1
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
    for f in reach:                       # safety net (should not fire)
        col.setdefault(f, 0)
    return col


class Layout:
    """Computes node positions and module frames for a set of source functions."""

    def __init__(self, project):
        self.project = project
        self.visible = []          # list[Function]
        self.frames = []           # list[(Module, x, y, w, h)]
        self.node_mode = "calls"   # "calls" | "code" | "compact"

    def _reset(self):
        for m in self.project.modules.values():
            for f in m.functions.values():
                f.col = None
                f.visible = False

    def _commit(self, col, node_mode):
        self.visible = []
        for f, c in col.items():
            f.col = c
            f.visible = True
            f.w, f.h = layout_node(f, node_mode)
            self.visible.append(f)
        self._place_modules()

    def build(self, sources, include_all_main=False, node_mode="calls"):
        self.node_mode = node_mode
        self._reset()

        col = assign_columns(sources)

        if include_all_main:
            # Guarantee every main-module function shows in the overview even
            # if it lives inside a cycle unreachable from the chosen sources.
            # Add the missing ones as extra roots and re-layer over the union so
            # the strictly-forward-edge invariant still holds globally.
            extra = [f for f in self.project.main.functions.values()
                     if f.kind != "module" and f not in col]
            if extra:
                col = assign_columns(list(sources) + extra)

        self._commit(col, node_mode)

    def build_focus(self, root, node_mode="calls"):
        """Focus view centered on `root`: the root's direct callers fill the
        first column, the root sits in the second column, and the root's callees
        fan out to the right. (When the root has no known callers it simply
        stays in the first column so there is no empty gap.)"""
        self.node_mode = node_mode
        self._reset()

        col = assign_columns([root])                 # root=0, callees=1,2,...
        callers = [g for g in self.project.callers_of(root) if g not in col]
        if callers:
            col = {f: c + 1 for f, c in col.items()}  # shift: root->1, callees->2+
            for g in callers:
                col[g] = 0                            # callers occupy column 0

        self._commit(col, node_mode)

    # -- module placement (2D: depth = column, siblings stack in rows) ------ #
    def _module_super_columns(self, by_module):
        """Group modules into left-to-right 'super-columns' by their call depth
        from main (longest path over the module-dependency graph). Modules at
        the same depth -- e.g. two modules both called only by main -- share a
        super-column and are stacked vertically instead of marching further
        right. Returns a list of module lists, ordered left to right."""
        main = self.project.main
        dep = defaultdict(set)
        for f in self.visible:
            for call in f.calls:
                g = call.callee
                if g is not None and g.visible and g.module is not f.module:
                    dep[f.module].add(g.module)

        dist = {main: 0}
        for _ in range(len(by_module) + 1):      # longest-path relaxation
            changed = False
            for u in list(dep):
                if u not in dist:
                    continue
                for v in dep[u]:
                    if v is main:
                        continue
                    if dist[u] + 1 > dist.get(v, -1):
                        dist[v] = dist[u] + 1
                        changed = True
            if not changed:
                break

        groups = defaultdict(list)
        placed = set()
        for m in by_module:
            if m in dist:
                groups[dist[m]].append(m)
                placed.add(m)

        result = []
        for d in sorted(groups):
            result.append(sorted(groups[d],
                                  key=lambda m: (1 if m.external else 0, m.name)))
        # any module unreachable from main (e.g. isolated cycle) -> rightmost
        leftovers = [m for m in by_module if m not in placed]
        if leftovers:
            result.append(sorted(leftovers,
                                  key=lambda m: (1 if m.external else 0, m.name)))
        return result

    def _layout_one_module(self, m, funcs, ox, oy):
        """Position a single module's functions with the band's top-left at
        (ox, oy); return the band's (width, height). Functions are laid out by
        their (compressed) call column so intra-module edges flow rightward."""
        cols = sorted({f.col for f in funcs})
        local = {c: i for i, c in enumerate(cols)}
        per_local = defaultdict(list)
        for f in funcs:
            per_local[local[f.col]].append(f)

        colw = {lc: max(f.w for f in per_local[lc]) for lc in per_local}
        colx = {}
        cx = ox + FRAME_PAD
        for lc in sorted(colw):
            colx[lc] = cx
            cx += colw[lc] + H_GAP
        inner_right = cx - H_GAP

        inner_top = oy + FRAME_HEADER + FRAME_PAD
        band_bottom = inner_top
        for lc in sorted(per_local):
            y = inner_top
            for f in sorted(per_local[lc],
                            key=lambda fn: (fn.kind != "module", fn.qualname)):
                f.x = colx[lc]
                f.y = y
                y += f.h + V_GAP
            band_bottom = max(band_bottom, y)

        fw = (inner_right + FRAME_PAD) - ox
        fh = (band_bottom - oy) + FRAME_PAD - V_GAP
        return fw, fh

    def _place_modules(self):
        """Lay modules out as a 2D grid: horizontal position is a module's call
        depth from main (its super-column), and modules sharing a depth stack
        vertically within that super-column. So a module called by main sits in
        the second column regardless of how many other modules main also calls;
        only modules called by *those* modules move further right. Cross-module
        edges still point rightward, and because same-depth modules never call
        each other, stacking them introduces no sideways edges."""
        by_module = defaultdict(list)
        for f in self.visible:
            by_module[f.module].append(f)

        self.frames = []
        cursor_x = NODE_X0
        for group in self._module_super_columns(by_module):
            col_top = 0.0
            col_w = 0.0
            for m in group:
                fw, fh = self._layout_one_module(m, by_module[m], cursor_x, col_top)
                self.frames.append((m, cursor_x, col_top, fw, fh))
                col_top += fh + LANE_GAP
                col_w = max(col_w, fw)
            cursor_x += col_w + LANE_GAP     # next super-column to the right

    # -- bounds ------------------------------------------------------------ #
    def bounds(self):
        if not self.visible:
            return (0.0, 0.0, 100.0, 100.0)
        minx = min(f.x for f in self.visible)
        miny = min(f.y for f in self.visible)
        maxx = max(f.x + f.w for f in self.visible)
        maxy = max(f.y + f.h for f in self.visible)
        return (minx, miny, maxx - minx, maxy - miny)


# --------------------------------------------------------------------------- #
#  Head-less dump (no raylib needed) - handy for inspecting the parse result
# --------------------------------------------------------------------------- #
def dump(project):
    print("Entry: %s" % project.entry_path)
    print("Modules parsed: %d" % len([m for m in project.modules.values() if not m.external]))
    eps = project.entrypoints()
    print("Entrypoints (main, uncalled): %s" % ", ".join(f.qualname for f in eps) or "(none)")
    print()
    for m in project.modules.values():
        real = [f for f in m.functions.values() if f.kind != "module"]
        if not real and not m.external:
            continue
        tag = " (external)" if m.external else ""
        print("=== module %s%s ===" % (m.name, tag))
        if m.module_node is not None:
            mc = m.module_node.calls
            if mc:
                print("  <module-level>:")
                for c in mc:
                    mark = "->" if c.resolved else " ?"
                    tgt = ("%s.%s" % (c.callee.module.name, c.callee.qualname)) if c.resolved else c.raw_name
                    print("      %s %s" % (mark, tgt))
        for f in sorted(m.functions.values(), key=lambda fn: fn.qualname):
            if f.kind == "module":
                continue
            print("  def %s  (%d calls)" % (f.qualname, len(f.calls)))
            for c in f.calls:
                mark = "->" if c.resolved else " ?"
                tgt = ("%s.%s" % (c.callee.module.name, c.callee.qualname)) if c.resolved else c.raw_name
                print("      %s %s" % (mark, tgt))
        print()

    print("Overview layout (module view):")
    lay = Layout(project)
    sources = [project.main.module_node] + project.entrypoints()
    lay.build(sources, include_all_main=True)
    cols = defaultdict(list)
    for f in lay.visible:
        cols[f.col].append("%s.%s" % (f.module.name, f.qualname))
    for c in sorted(cols):
        print("  column %d: %s" % (c, ", ".join(sorted(cols[c]))))


# --------------------------------------------------------------------------- #
#  Interactive viewer (raylib)
# --------------------------------------------------------------------------- #
def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class Viewer:
    WIN_W = 1400
    WIN_H = 850
    OFFSET_X = 60.0        # world 'target' maps to this screen position...
    OFFSET_Y = 90.0        # ...so focused nodes land near the top-left.

    def __init__(self, project):
        self.project = project
        self.layout = Layout(project)
        self.mode = "module"       # "module" or "function"
        self.node_mode = "calls"   # "calls" | "code" | "compact"
        self.root = None           # focused Function in function mode
        self.selected = None       # single-clicked node (edges highlighted)
        self.history = []          # stack of (mode, root) focus states only

        # code-view interaction
        self.mouse_world = (0.0, 0.0)
        self.code_hits = []        # list of (x, y, w, h, caller_func, call) in world space
        self._hover_call = None    # call token under the cursor (code view)

        # camera + animation
        self.cam = None
        self.goal_target = (0.0, 0.0)
        self.goal_zoom = 1.0
        self.animating = False

        # input state
        self.dragging = False
        self.press_pos = (0.0, 0.0)
        self.rdragging = False
        self.rpress_pos = (0.0, 0.0)
        self.prev_mouse = (0.0, 0.0)

        self.hover_func = None
        self.module_colors = {}

    # ---- colour bookkeeping --------------------------------------------- #
    def _module_color(self, module):
        if module not in self.module_colors:
            if module is self.project.main:
                idx = 0
            else:
                idx = 1 + (len(self.module_colors) % (len(MODULE_PALETTE) - 1))
            self.module_colors[module] = MODULE_PALETTE[idx % len(MODULE_PALETTE)]
        return self.module_colors[module]

    # ---- view switching -------------------------------------------------- #
    def show_overview(self, record=True):
        if record and self.mode == "function":
            self.history.append((self.mode, self.root))
        self.mode = "module"
        self.root = None
        self.selected = None
        sources = [self.project.main.module_node] + self.project.entrypoints()
        self.layout.build(sources, include_all_main=True, node_mode=self.node_mode)
        self.fit_all()

    def _rebuild_current(self):
        """Re-run the layout for whatever graph is showing (e.g. after a
        display-mode change), preserving root/selection."""
        if self.mode == "module":
            sources = [self.project.main.module_node] + self.project.entrypoints()
            self.layout.build(sources, include_all_main=True, node_mode=self.node_mode)
        else:
            self.layout.build_focus(self.root, node_mode=self.node_mode)

    def cycle_node_mode(self):
        order = ("calls", "code", "compact")
        self.node_mode = order[(order.index(self.node_mode) + 1) % len(order)]
        self._rebuild_current()
        self.fit_all()

    def select(self, func):
        """First click: mark a node so its call edges stay highlighted."""
        self.selected = func

    def focus(self, func, record=True):
        """Zoom in on a node: it moves to the second column with its callers in
        the first column and callees to the right. Only focus navigation is
        recorded in history, so Backspace steps between focused nodes."""
        if func is None or func.external or func.kind == "module":
            return
        if record:
            self.history.append((self.mode, self.root))
        self.mode = "function"
        self.root = func
        self.selected = func
        self.layout.build_focus(func, node_mode=self.node_mode)
        # Frame the top-left-most node -- the first caller in column 0 -- so the
        # callers are visible on the left rather than off-screen. (With no
        # callers this is the focused node itself.)
        anchor = min(self.layout.visible, key=lambda f: (f.x, f.y), default=func)
        self.goal_target = (anchor.x - 12.0, anchor.y - 12.0)
        self.goal_zoom = 1.0
        self.animating = True

    def go_back(self):
        if not self.history:
            return
        mode, root = self.history.pop()
        if mode == "module" or root is None:
            self.show_overview(record=False)
        else:
            self.focus(root, record=False)

    # ---- camera ---------------------------------------------------------- #
    def fit_all(self):
        x, y, w, h = self.layout.bounds()
        avail_w = self.WIN_W - self.OFFSET_X - 40
        avail_h = self.WIN_H - self.OFFSET_Y - 40
        zoom = min(avail_w / max(w, 1.0), avail_h / max(h, 1.0))
        zoom = _clamp(zoom, 0.15, 1.2)
        self.goal_target = (x - 12.0, y - 12.0)
        self.goal_zoom = zoom
        self.animating = True

    def _ease_camera(self):
        if not self.animating:
            return
        t = 0.20
        tx, ty = self.goal_target
        nx = self.cam.target.x + (tx - self.cam.target.x) * t
        ny = self.cam.target.y + (ty - self.cam.target.y) * t
        nz = self.cam.zoom + (self.goal_zoom - self.cam.zoom) * t
        self.cam.target = pr.Vector2(nx, ny)
        self.cam.zoom = nz
        if (abs(tx - nx) < 0.5 and abs(ty - ny) < 0.5
                and abs(self.goal_zoom - nz) < 0.002):
            self.cam.target = pr.Vector2(tx, ty)
            self.cam.zoom = self.goal_zoom
            self.animating = False

    # ---- hit testing ----------------------------------------------------- #
    def _func_at(self, wx, wy):
        for f in self.layout.visible:
            if f.x <= wx <= f.x + f.w and f.y <= wy <= f.y + f.h:
                return f
        return None

    def _code_line_center_y(self, f, lineno):
        """World y of the center of a source line inside f's code node, or None
        when that line is not currently shown."""
        base = f.code_base
        if base is None:
            return None
        rel = lineno - base
        if 0 <= rel < f.code_nlines:
            return f.y + HEADER_H + CODE_PAD_TOP + rel * CODE_LINE_H + CODE_LINE_H / 2
        return None

    def _row_index_at(self, f, wy):
        top = f.y + HEADER_H + PAD_TOP
        n, _ = displayed_rows(f)
        if wy < top:
            return None
        idx = int((wy - top) // ROW_H)
        if 0 <= idx < n:
            return idx
        return None

    # ---- input ----------------------------------------------------------- #
    def _handle_input(self):
        mx, my = pr.get_mouse_position().x, pr.get_mouse_position().y
        dx = mx - self.prev_mouse[0]
        dy = my - self.prev_mouse[1]
        self.prev_mouse = (mx, my)

        # zoom to cursor
        wheel = pr.get_mouse_wheel_move()
        if wheel != 0:
            mouse = pr.Vector2(mx, my)
            before = pr.get_screen_to_world_2d(mouse, self.cam)
            factor = 1.12 if wheel > 0 else 1 / 1.12
            self.cam.zoom = _clamp(self.cam.zoom * factor, 0.1, 3.0)
            after = pr.get_screen_to_world_2d(mouse, self.cam)
            self.cam.target = pr.Vector2(
                self.cam.target.x + (before.x - after.x),
                self.cam.target.y + (before.y - after.y),
            )
            self.animating = False

        # right button: right-drag pans; a right-click without dragging focuses
        # the callee under the cursor (as if the called node itself was clicked)
        if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_RIGHT):
            self.rpress_pos = (mx, my)
            self.rdragging = False
        if pr.is_mouse_button_down(pr.MOUSE_BUTTON_RIGHT):
            rmoved = abs(mx - self.rpress_pos[0]) + abs(my - self.rpress_pos[1])
            if rmoved > 6:
                self.rdragging = True
            if self.rdragging:
                self.cam.target = pr.Vector2(
                    self.cam.target.x - dx / self.cam.zoom,
                    self.cam.target.y - dy / self.cam.zoom,
                )
                self.animating = False
        if pr.is_mouse_button_released(pr.MOUSE_BUTTON_RIGHT):
            if not self.rdragging:
                self._on_right_click(mx, my)
            self.rdragging = False

        # left button: distinguish click from drag-pan
        if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT):
            self.press_pos = (mx, my)
            self.dragging = False
        if pr.is_mouse_button_down(pr.MOUSE_BUTTON_LEFT):
            moved = abs(mx - self.press_pos[0]) + abs(my - self.press_pos[1])
            if moved > 6:
                self.dragging = True
            if self.dragging:
                self.cam.target = pr.Vector2(
                    self.cam.target.x - dx / self.cam.zoom,
                    self.cam.target.y - dy / self.cam.zoom,
                )
                self.animating = False
        if pr.is_mouse_button_released(pr.MOUSE_BUTTON_LEFT):
            if not self.dragging:
                self._on_click(mx, my)
            self.dragging = False

        # keyboard
        if pr.is_key_pressed(pr.KEY_HOME):
            self.show_overview()
        if pr.is_key_pressed(pr.KEY_BACKSPACE):
            self.go_back()
        if pr.is_key_pressed(pr.KEY_F):
            self.fit_all()
        if pr.is_key_pressed(pr.KEY_C):
            self.cycle_node_mode()

        # hover
        world = pr.get_screen_to_world_2d(pr.Vector2(mx, my), self.cam)
        self.mouse_world = (world.x, world.y)
        self.hover_func = self._func_at(world.x, world.y)

    def _on_click(self, mx, my):
        """Two-stage left click: the first click on a node selects it (its call
        edges stay highlighted); a second click on the same node focuses it
        (zoom in). Clicking empty space clears the selection."""
        world = pr.get_screen_to_world_2d(pr.Vector2(mx, my), self.cam)
        f = self._func_at(world.x, world.y)
        if f is None:
            self.selected = None
            return
        if f.kind == "module":
            self.select(f)                 # highlight only; module node isn't focusable
            return
        if f is self.selected and f is not self.root:
            self.focus(f)                  # second click -> zoom in
        else:
            self.select(f)                 # first click -> highlight its call edges

    def _on_right_click(self, mx, my):
        """Right-clicking a call jumps straight to the CALLEE node -- the same
        as clicking the called node itself. (Left-click selects/focuses the
        node the call is made FROM.) Right-clicking anything else does nothing."""
        world = pr.get_screen_to_world_2d(pr.Vector2(mx, my), self.cam)
        if self.node_mode == "code":
            for (hx, hy, hw, hh, caller, call) in self.code_hits:
                if hx <= world.x <= hx + hw and hy <= world.y <= hy + hh:
                    self.focus(call.callee)      # focus() ignores None/external
                    return
            return
        if self.node_mode == "calls":
            f = self._func_at(world.x, world.y)
            if f is None:
                return
            idx = self._row_index_at(f, world.y)
            if idx is not None and idx < len(f.calls):
                self.focus(f.calls[idx].callee)
        # compact mode shows no calls -> nothing to focus

    # ---- drawing --------------------------------------------------------- #
    @staticmethod
    def _col(rgb, a=255):
        return pr.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(a))

    @staticmethod
    def _fit_text(s, max_w, size):
        if _text_w(s, size) <= max_w:
            return s
        while s and _text_w(s + "...", size) > max_w:
            s = s[:-1]
        return s + "..."

    def _draw_frames(self):
        for module, x, y, w, h in self.layout.frames:
            rgb = self._module_color(module)
            rec = pr.Rectangle(x, y, w, h)
            pr.draw_rectangle_rec(rec, self._col(rgb, 20))
            pr.draw_rectangle_lines_ex(rec, 2.0, self._col(rgb, 150))
            # title tab
            title = module.name + (" (external)" if module.external else ".py")
            tw = _text_w(title, FRAME_FS) + 20
            tab = pr.Rectangle(x, y, float(tw), FRAME_HEADER)
            pr.draw_rectangle_rec(tab, self._col(rgb, 210))
            _draw_text(title, int(x + 10), int(y + 6), FRAME_FS,
                         self._col((255, 255, 255)))

    def _row_center_y(self, f, i):
        n, _ = displayed_rows(f)
        i = min(i, max(n - 1, 0))
        return f.y + HEADER_H + PAD_TOP + i * ROW_H + ROW_H / 2

    def _edge_anchor_y(self, f, call, i):
        if self.node_mode == "code":
            y = self._code_line_center_y(f, call.lineno)
            if y is not None:
                return y
            return f.y + f.h / 2          # call not on a shown line
        if self.node_mode == "compact":
            return f.y + f.h / 2          # title-only node: anchor at its middle
        return self._row_center_y(f, i)

    def _draw_edges(self):
        for f in self.layout.visible:
            for i, call in enumerate(f.calls):
                g = call.callee
                if g is None or not g.visible:
                    continue
                start = pr.Vector2(f.x + f.w, self._edge_anchor_y(f, call, i))
                end = pr.Vector2(g.x, g.y + HEADER_H / 2)
                if self.selected is f or self.selected is g:
                    color = self._col((235, 150, 40), 255)   # selected node's edges
                    thick = 3.0
                elif self.hover_func is f or self.hover_func is g:
                    color = self._col((90, 130, 200), 230)
                    thick = 2.5
                else:
                    color = self._col((150, 160, 175), 120)
                    thick = 1.6
                pr.draw_line_bezier(start, end, thick, color)

    def _draw_nodes(self):
        self.code_hits = []        # rebuilt every frame (code view only)
        for f in self.layout.visible:
            rgb = self._module_color(f.module)
            rec = pr.Rectangle(f.x, f.y, f.w, f.h)
            pr.draw_rectangle_rec(rec, self._col((252, 252, 253)))

            is_root = (self.mode == "function" and f is self.root)
            is_sel = (f is self.selected)
            if is_root:
                border, bt = self._col((40, 90, 200)), 3.0
            elif is_sel:
                border, bt = self._col((235, 150, 40)), 3.0
            elif self.hover_func is f:
                border, bt = self._col(rgb, 255), 2.5
            else:
                border, bt = self._col((205, 210, 220)), 1.5
            pr.draw_rectangle_lines_ex(rec, bt, border)

            # header
            hdr = pr.Rectangle(f.x, f.y, f.w, HEADER_H)
            pr.draw_rectangle_rec(hdr, self._col(rgb, 235 if f.kind != "module" else 255))
            label = f.qualname
            if f.kind == "module":
                label = "<module> " + f.module.name
            label = self._fit_text(label, f.w - 16, TITLE_FS)
            _draw_text(label, int(f.x + 8), int(f.y + 7), TITLE_FS,
                         self._col((255, 255, 255)))

            if self.node_mode == "code":
                self._draw_code_body(f, rgb)
            elif self.node_mode == "calls":
                self._draw_list_body(f, rgb)
            # compact: header/title only, no body

    def _draw_list_body(self, f, rgb):
        n, overflow = displayed_rows(f)
        if not f.calls:
            _draw_text("(no calls)", int(f.x + 10),
                         int(f.y + HEADER_H + PAD_TOP + 2), ROW_FS,
                         self._col((160, 165, 175)))
        for i in range(n):
            call = f.calls[i]
            ry = f.y + HEADER_H + PAD_TOP + i * ROW_H
            if call.resolved:
                txt = "-> " + call.callee.qualname
                if call.callee.module is not f.module:
                    txt = "-> " + call.callee.module.name + "." + call.callee.qualname
                col = self._col((40, 50, 65))
            else:
                txt = "  " + call.raw_name
                col = self._col((165, 170, 180))
            if self.hover_func is f and 0 <= self._hover_row < n and self._hover_row == i:
                pr.draw_rectangle_rec(
                    pr.Rectangle(f.x + 2, ry, f.w - 4, ROW_H),
                    self._col(rgb, 40))
            txt = self._fit_text(txt, f.w - 20, ROW_FS)
            _draw_text(txt, int(f.x + 10), int(ry + 3), ROW_FS, col)
        if overflow > 0:
            ry = f.y + HEADER_H + PAD_TOP + n * ROW_H
            _draw_text("+%d more..." % overflow, int(f.x + 10),
                         int(ry + 3), ROW_FS, self._col((150, 155, 165)))

    def _draw_code_body(self, f, rgb):
        rows = f.code_rows or []
        if not rows:
            _draw_text("(no source)", int(f.x + 10),
                         int(f.y + HEADER_H + CODE_PAD_TOP + 2), CODE_FS,
                         self._col((160, 165, 175)))
            return

        gutter_w = self._gutter_w(f)
        body_top = f.y + HEADER_H + CODE_PAD_TOP
        num_right = f.x + CODE_PAD_X + gutter_w
        code_left = num_right + CODE_GUTTER_GAP
        body_right = f.x + f.w - CODE_PAD_X

        # group call sites by the source line they start on
        calls_by_line = defaultdict(list)
        for call in f.calls:
            calls_by_line[call.lineno].append(call)

        for row, (lineno, raw) in enumerate(rows):
            ly = body_top + row * CODE_LINE_H
            disp = raw.expandtabs(4)

            # line-number gutter (right-aligned)
            num = str(lineno)
            _draw_text(num, int(num_right - self._text_w(num, CODE_FS)),
                         int(ly), CODE_FS, self._col((175, 180, 190)))

            # call highlights (drawn behind the code text)
            for call in calls_by_line.get(lineno, ()):  # only lines starting here
                self._draw_call_token(f, call, raw, ly, code_left, body_right)

            # code text, truncated to the node's inner width
            text = self._fit_text(disp, body_right - code_left, CODE_FS)
            _draw_text(text, int(code_left), int(ly), CODE_FS,
                         self._col((45, 52, 66)))

        if f.code_overflow > 0:
            oy = body_top + len(rows) * CODE_LINE_H
            _draw_text("+%d more lines..." % f.code_overflow,
                         int(code_left), int(oy), CODE_FS,
                         self._col((150, 155, 165)))

    def _draw_call_token(self, f, call, raw, ly, code_left, body_right):
        """Highlight one call's callee expression on its line and register it as
        a clickable region. Tokens spanning multiple lines are clamped to the
        first line; tokens past the node's right edge are skipped."""
        c0 = max(0, min(call.col_offset, len(raw)))
        if call.end_lineno == call.lineno:
            c1 = max(c0, min(call.end_col_offset, len(raw)))
        else:
            c1 = len(raw)
        if c1 <= c0:
            return

        pre_w = self._text_w(raw[:c0].expandtabs(4), CODE_FS)
        tok_w = self._text_w(raw[c0:c1].expandtabs(4), CODE_FS)
        tx = code_left + pre_w
        if tok_w <= 0 or tx >= body_right:
            return
        if tx + tok_w > body_right:          # partly clipped -> not clickable
            return

        hovered = (self._hover_call is call)
        is_focus_hi = (f is self.selected)      # tokens of the selected node
        if call.resolved:
            if is_focus_hi:
                fill = self._col((235, 150, 40), 150 if hovered else 90)
                line = self._col((235, 150, 40), 220)
            else:
                fill = self._col((90, 150, 235), 140 if hovered else 70)
                line = self._col((90, 150, 235), 200 if hovered else 120)
        else:
            fill = self._col((150, 150, 160), 80 if hovered else 36)
            line = self._col((150, 150, 160), 120 if hovered else 60)

        rect = pr.Rectangle(tx - 1, ly - 1, tok_w + 2, CODE_LINE_H)
        pr.draw_rectangle_rec(rect, fill)
        pr.draw_rectangle_lines_ex(rect, 1.0, line)
        self.code_hits.append((tx - 1, ly - 1, tok_w + 2, CODE_LINE_H, f, call))

    def _gutter_w(self, f):
        last = f.code_rows[-1][0] if f.code_rows else (f.code_base or 1)
        return self._text_w(str(last), CODE_FS)

    @staticmethod
    def _text_w(s, size):
        return _text_w(s, size)

    def _draw_hud(self):
        # top bar
        pr.draw_rectangle(0, 0, self.WIN_W, 40, self._col((30, 34, 42)))
        if self.mode == "module":
            crumb = "Overview  -  %s" % self.project.main.name
        else:
            crumb = "Focus  -  %s.%s" % (self.root.module.name, self.root.qualname)
        crumb += "   [%s]" % self.node_mode
        _draw_text(crumb, 12, 11, 18, self._col((255, 255, 255)))
        hint = ("click: select   click again: zoom   R-click call: open callee   "
                "drag: pan   wheel: zoom   C: mode (calls/code/compact)   "
                "Backspace: back   Home: overview   F: fit   Esc: quit")
        hw = _text_w(hint, 12)
        _draw_text(hint, self.WIN_W - hw - 12, 14, 12, self._col((170, 178, 190)))

    # ---- main loop ------------------------------------------------------- #
    def run(self):
        pr.set_config_flags(pr.FLAG_WINDOW_RESIZABLE | pr.FLAG_MSAA_4X_HINT)
        pr.init_window(self.WIN_W, self.WIN_H, "callgraph - %s" % self.project.main.name)
        pr.set_target_fps(60)

        # ---- font hook ---------------------------------------------------- #
        # All on-screen text goes through _text_w/_draw_text, which use the
        # module-level _FONT. Leaving it None uses raylib's built-in font. To
        # use a custom font, load it HERE (needs the window/GL context) and set
        # _FONT + _SPACING; because measuring and drawing share this font,
        # code-view call-token highlights stay aligned automatically. Example:
        #     global _FONT, _SPACING
        #     _FONT = pr.load_font_ex("MyMono.ttf", 32, None, 0)
        #     pr.set_texture_filter(_FONT.texture, pr.TEXTURE_FILTER_BILINEAR)
        #     _SPACING = 0.5
        # (A monospaced font is recommended for the code view.)
        global _FONT, _SPACING
        _FONT = pr.load_font_ex("CascadiaCode.ttf", 32, None, 0)
        pr.set_texture_filter(_FONT.texture, pr.TEXTURE_FILTER_BILINEAR)
        _SPACING = 0.5

        self.cam = pr.Camera2D(pr.Vector2(self.OFFSET_X, self.OFFSET_Y),
                               pr.Vector2(0, 0), 0.0, 1.0)
        self.prev_mouse = (pr.get_mouse_position().x, pr.get_mouse_position().y)
        self.show_overview(record=False)

        while not pr.window_should_close():
            self.WIN_W = pr.get_screen_width()
            self.WIN_H = pr.get_screen_height()
            self._handle_input()
            self._ease_camera()

            # precompute hover state used while drawing
            self._hover_row = -1
            self._hover_call = None
            if self.node_mode == "code":
                wx, wy = self.mouse_world
                for (hx, hy, hw, hh, caller, call) in self.code_hits:
                    if hx <= wx <= hx + hw and hy <= wy <= hy + hh:
                        self._hover_call = call
                        break
            elif self.node_mode == "calls" and self.hover_func is not None:
                world = pr.get_screen_to_world_2d(pr.get_mouse_position(), self.cam)
                ri = self._row_index_at(self.hover_func, world.y)
                self._hover_row = ri if ri is not None else -1

            pr.begin_drawing()
            pr.clear_background(self._col((243, 244, 247)))
            pr.begin_mode_2d(self.cam)
            self._draw_frames()
            self._draw_edges()
            self._draw_nodes()
            pr.end_mode_2d()
            self._draw_hud()
            pr.end_drawing()

        pr.close_window()


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}

    if not args:
        # default to the bundled sample so it runs out of the box
        here = os.path.dirname(os.path.abspath(__file__))
        default = os.path.join(here, "sample_project", "main.py")
        path = default if os.path.exists(default) else os.path.abspath(__file__)
    else:
        path = args[0]

    if not os.path.isfile(path):
        sys.stderr.write("error: no such file: %s\n" % path)
        return 2

    project = Project(path).build()

    if "--dump" in flags:
        dump(project)
        return 0

    if pr is None:
        sys.stderr.write(
            "raylib is not installed. Install it with:\n"
            "    pip install raylib\n"
            "Then run:\n"
            "    python callgraph.py %s\n"
            "(You can inspect the parse result without a GUI via --dump.)\n" % path)
        return 1

    Viewer(project).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
