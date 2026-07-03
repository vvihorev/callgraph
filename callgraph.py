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

Controls
--------
    Left click a node/call : focus that function (re-root the graph)
    Left drag              : pan
    Mouse wheel            : zoom to cursor
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

    def __init__(self, raw_name, lineno, callee=None):
        self.raw_name = raw_name      # textual name at the call site, e.g. "helpers.render"
        self.lineno = lineno
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
        self.h = 0.0
        self.visible = False

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
            self.calls[self._scope()].append((raw, getattr(node, "lineno", 0)))
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

        scan = ModuleScan()
        scan.visit(tree)
        module._scan = scan

        # Create Function objects for every defined name up front so that
        # cross-references can find them before resolution runs.
        for qual, node in scan.defined:
            kind = "class" if qual in scan.classes else "function"
            module.get_or_create(qual, node=node, kind=kind)
        module.module_node = Function("\u27e8module\u27e9", module, kind="module")
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
            for raw, lineno in raw_calls:
                callee = self._resolve_name(module, scan, scope, raw)
                caller.calls.append(Call(raw, lineno, callee))

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
NODE_X0 = 40.0
FRAME_PAD = 18.0
FRAME_HEADER = 28.0
LANE_GAP = 44.0
MAX_ROWS = 16

TITLE_FS = 16
ROW_FS = 13
FRAME_FS = 18

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


def bfs_columns(sources, col=None):
    """Assign a column (shortest call-distance) to each reachable function."""
    if col is None:
        col = {}
    dq = deque()
    for s in sources:
        if s not in col:
            col[s] = 0
            dq.append(s)
    while dq:
        f = dq.popleft()
        c = col[f]
        for call in f.calls:
            g = call.callee
            if g is not None and g not in col:
                col[g] = c + 1
                dq.append(g)
    return col


class Layout:
    """Computes node positions and module frames for a set of source functions."""

    def __init__(self, project):
        self.project = project
        self.visible = []          # list[Function]
        self.frames = []           # list[(Module, x, y, w, h)]

    def build(self, sources, include_all_main=False):
        # reset transient state
        for m in self.project.modules.values():
            for f in m.functions.values():
                f.col = None
                f.visible = False

        col = bfs_columns(sources)

        if include_all_main:
            # Guarantee every main-module function shows in the overview even
            # if it lives inside a cycle unreachable from the chosen sources.
            for f in self.project.main.functions.values():
                if f.kind == "module":
                    continue
                if f not in col:
                    bfs_columns([f], col)

        # commit column assignment + node heights
        self.visible = []
        for f, c in col.items():
            f.col = c
            f.visible = True
            f.h = node_height(f)
            self.visible.append(f)

        self._place_lanes()

    # -- module lanes ------------------------------------------------------ #
    def _place_lanes(self):
        by_module = defaultdict(list)
        for f in self.visible:
            by_module[f.module].append(f)

        def module_key(m):
            if m is self.project.main:
                return (0, 0, "")
            funcs = by_module[m]
            min_col = min(f.col for f in funcs)
            return (1 if not m.external else 2, min_col, m.name)

        ordered = sorted(by_module.keys(), key=module_key)

        self.frames = []
        lane_top = 0.0
        for m in ordered:
            funcs = by_module[m]
            per_col = defaultdict(list)
            for f in funcs:
                per_col[f.col].append(f)

            inner_top = lane_top + FRAME_HEADER + FRAME_PAD
            lane_bottom = inner_top
            min_x = math.inf
            max_x = -math.inf
            for c in sorted(per_col):
                y = inner_top
                for f in sorted(per_col[c], key=lambda fn: (fn.kind != "module", fn.qualname)):
                    f.x = NODE_X0 + c * COL_STRIDE
                    f.y = y
                    y += f.h + V_GAP
                    min_x = min(min_x, f.x)
                    max_x = max(max_x, f.x + NODE_W)
                lane_bottom = max(lane_bottom, y)

            fx = min_x - FRAME_PAD
            fw = (max_x - min_x) + 2 * FRAME_PAD
            fh = (lane_bottom - lane_top) + FRAME_PAD - V_GAP
            self.frames.append((m, fx, lane_top, fw, fh))
            lane_top = lane_top + fh + LANE_GAP

    # -- bounds ------------------------------------------------------------ #
    def bounds(self):
        if not self.visible:
            return (0.0, 0.0, 100.0, 100.0)
        minx = min(f.x for f in self.visible)
        miny = min(f.y for f in self.visible)
        maxx = max(f.x + NODE_W for f in self.visible)
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
        self.root = None           # focused Function in function mode
        self.highlight = None      # callee highlighted after a click
        self.history = []          # stack of (mode, root, highlight)

        # camera + animation
        self.cam = None
        self.goal_target = (0.0, 0.0)
        self.goal_zoom = 1.0
        self.animating = False

        # input state
        self.dragging = False
        self.press_pos = (0.0, 0.0)
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
        if record:
            self.history.append((self.mode, self.root, self.highlight))
        self.mode = "module"
        self.root = None
        self.highlight = None
        sources = [self.project.main.module_node] + self.project.entrypoints()
        self.layout.build(sources, include_all_main=True)
        self.fit_all()

    def focus(self, func, highlight=None, record=True):
        if func is None or func.external:
            return
        if record:
            self.history.append((self.mode, self.root, self.highlight))
        self.mode = "function"
        self.root = func
        self.highlight = highlight
        self.layout.build([func])
        # place the focused node near the top-left of the screen
        self.goal_target = (func.x - 12.0, func.y - 12.0)
        self.goal_zoom = 1.0
        self.animating = True

    def go_back(self):
        if not self.history:
            return
        mode, root, highlight = self.history.pop()
        if mode == "module":
            self.show_overview(record=False)
        else:
            self.focus(root, highlight, record=False)

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
            if f.x <= wx <= f.x + NODE_W and f.y <= wy <= f.y + f.h:
                return f
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

        # right-drag pans
        if pr.is_mouse_button_down(pr.MOUSE_BUTTON_RIGHT):
            self.cam.target = pr.Vector2(
                self.cam.target.x - dx / self.cam.zoom,
                self.cam.target.y - dy / self.cam.zoom,
            )
            self.animating = False

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
        if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_RIGHT):
            pass  # right press begins a pan; back is on release-less Backspace
        if pr.is_key_pressed(pr.KEY_F):
            self.fit_all()

        # hover
        world = pr.get_screen_to_world_2d(pr.Vector2(mx, my), self.cam)
        self.hover_func = self._func_at(world.x, world.y)

    def _on_click(self, mx, my):
        world = pr.get_screen_to_world_2d(pr.Vector2(mx, my), self.cam)
        f = self._func_at(world.x, world.y)
        if f is None or f.kind == "module":
            return
        # if a specific call row was hit, remember its callee to highlight
        highlight = None
        idx = self._row_index_at(f, world.y)
        if idx is not None and idx < len(f.calls):
            highlight = f.calls[idx].callee
        # clicking a call focuses the function the call is made FROM
        self.focus(f, highlight=highlight)

    # ---- drawing --------------------------------------------------------- #
    @staticmethod
    def _col(rgb, a=255):
        return pr.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(a))

    @staticmethod
    def _fit_text(s, max_w, size):
        if pr.measure_text(s, size) <= max_w:
            return s
        while s and pr.measure_text(s + "\u2026", size) > max_w:
            s = s[:-1]
        return s + "\u2026"

    def _draw_frames(self):
        for module, x, y, w, h in self.layout.frames:
            rgb = self._module_color(module)
            rec = pr.Rectangle(x, y, w, h)
            pr.draw_rectangle_rec(rec, self._col(rgb, 20))
            pr.draw_rectangle_lines_ex(rec, 2.0, self._col(rgb, 150))
            # title tab
            title = module.name + (" (external)" if module.external else ".py")
            tw = pr.measure_text(title, FRAME_FS) + 20
            tab = pr.Rectangle(x, y, float(tw), FRAME_HEADER)
            pr.draw_rectangle_rec(tab, self._col(rgb, 210))
            pr.draw_text(title, int(x + 10), int(y + 6), FRAME_FS,
                         self._col((255, 255, 255)))

    def _row_center_y(self, f, i):
        n, _ = displayed_rows(f)
        i = min(i, max(n - 1, 0))
        return f.y + HEADER_H + PAD_TOP + i * ROW_H + ROW_H / 2

    def _draw_edges(self):
        for f in self.layout.visible:
            for i, call in enumerate(f.calls):
                g = call.callee
                if g is None or not g.visible:
                    continue
                start = pr.Vector2(f.x + NODE_W, self._row_center_y(f, i))
                end = pr.Vector2(g.x, g.y + HEADER_H / 2)
                if (self.mode == "function" and f is self.root
                        and g is self.highlight):
                    color = self._col((235, 150, 40), 255)
                    thick = 3.0
                elif self.hover_func is f or self.hover_func is g:
                    color = self._col((90, 130, 200), 230)
                    thick = 2.5
                else:
                    color = self._col((150, 160, 175), 120)
                    thick = 1.6
                pr.draw_line_bezier(start, end, thick, color)

    def _draw_nodes(self):
        for f in self.layout.visible:
            rgb = self._module_color(f.module)
            rec = pr.Rectangle(f.x, f.y, NODE_W, f.h)
            pr.draw_rectangle_rec(rec, self._col((252, 252, 253)))

            is_root = (self.mode == "function" and f is self.root)
            is_hi = (f is self.highlight)
            if is_root:
                border, bt = self._col((40, 90, 200)), 3.0
            elif is_hi:
                border, bt = self._col((235, 150, 40)), 3.0
            elif self.hover_func is f:
                border, bt = self._col(rgb, 255), 2.5
            else:
                border, bt = self._col((205, 210, 220)), 1.5
            pr.draw_rectangle_lines_ex(rec, bt, border)

            # header
            hdr = pr.Rectangle(f.x, f.y, NODE_W, HEADER_H)
            pr.draw_rectangle_rec(hdr, self._col(rgb, 235 if f.kind != "module" else 255))
            label = f.qualname
            if f.kind == "module":
                label = "\u27e8module\u27e9 " + f.module.name
            label = self._fit_text(label, NODE_W - 16, TITLE_FS)
            pr.draw_text(label, int(f.x + 8), int(f.y + 7), TITLE_FS,
                         self._col((255, 255, 255)))

            # call rows
            n, overflow = displayed_rows(f)
            if not f.calls:
                pr.draw_text("(no calls)", int(f.x + 10),
                             int(f.y + HEADER_H + PAD_TOP + 2), ROW_FS,
                             self._col((160, 165, 175)))
            for i in range(n):
                call = f.calls[i]
                ry = f.y + HEADER_H + PAD_TOP + i * ROW_H
                if call.resolved:
                    txt = "\u2192 " + call.callee.qualname
                    if call.callee.module is not f.module:
                        txt = "\u2192 " + call.callee.module.name + "." + call.callee.qualname
                    col = self._col((40, 50, 65))
                else:
                    txt = "  " + call.raw_name
                    col = self._col((165, 170, 180))
                if self.hover_func is f and 0 <= (self._hover_row) < n and self._hover_row == i:
                    pr.draw_rectangle_rec(
                        pr.Rectangle(f.x + 2, ry, NODE_W - 4, ROW_H),
                        self._col(rgb, 40))
                txt = self._fit_text(txt, NODE_W - 20, ROW_FS)
                pr.draw_text(txt, int(f.x + 10), int(ry + 3), ROW_FS, col)
            if overflow > 0:
                ry = f.y + HEADER_H + PAD_TOP + n * ROW_H
                pr.draw_text("+%d more\u2026" % overflow, int(f.x + 10),
                             int(ry + 3), ROW_FS, self._col((150, 155, 165)))

    def _draw_hud(self):
        # top bar
        pr.draw_rectangle(0, 0, self.WIN_W, 40, self._col((30, 34, 42)))
        if self.mode == "module":
            crumb = "Overview  \u2022  %s" % self.project.main.name
        else:
            crumb = "Focus  \u2022  %s.%s" % (self.root.module.name, self.root.qualname)
        pr.draw_text(crumb, 12, 11, 18, self._col((255, 255, 255)))
        hint = ("click node: focus   drag: pan   wheel: zoom   "
                "Backspace: back   Home: overview   F: fit   Esc: quit")
        hw = pr.measure_text(hint, 12)
        pr.draw_text(hint, self.WIN_W - hw - 12, 14, 12, self._col((170, 178, 190)))

    # ---- main loop ------------------------------------------------------- #
    def run(self):
        pr.set_config_flags(pr.FLAG_WINDOW_RESIZABLE | pr.FLAG_MSAA_4X_HINT)
        pr.init_window(self.WIN_W, self.WIN_H, "callgraph \u2013 %s" % self.project.main.name)
        pr.set_target_fps(60)
        self.cam = pr.Camera2D(pr.Vector2(self.OFFSET_X, self.OFFSET_Y),
                               pr.Vector2(0, 0), 0.0, 1.0)
        self.prev_mouse = (pr.get_mouse_position().x, pr.get_mouse_position().y)
        self.show_overview(record=False)

        while not pr.window_should_close():
            self.WIN_W = pr.get_screen_width()
            self.WIN_H = pr.get_screen_height()
            self._handle_input()
            self._ease_camera()

            # precompute which row is hovered (used while drawing rows)
            self._hover_row = -1
            if self.hover_func is not None:
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
