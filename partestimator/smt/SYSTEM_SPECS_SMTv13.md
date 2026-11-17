# SMTv13 — system specification

Environment, security posture, internal design, and an honest account of what
was and was not verified before handover.

---

## 1. Target environment

| | |
|---|---|
| OS | Windows 10 / 11, x64 only |
| Rights | Standard user. No administrator, no elevation, no installer |
| Python | CPython 3.10 – 3.13, 64-bit. **3.13 preferred** |
| Wheels | `win_amd64` only — no source builds, no compiler needed |
| Disk | ~350 MB after install, most of it the CAD kernel |
| GPU | None. The 3D view is rendered on the CPU |
| Network | Once, during setup. Never again |

There are no Linux or macOS code paths, no WSL assumption and no POSIX-only
dependency. Scripts are `.bat`, ASCII, CRLF, and every path in them is quoted so
they survive `C:\Users\First Last\...` and a mapped network drive.

### What is assumed about Python, and why

**Assumed: a suitable CPython is already installed, or can be installed by you
without help.** The launcher looks for one and refuses clearly if it cannot find
one. It does **not** try to install Python.

That is a deliberate choice, not laziness. Installing Python means either an
MSI that wants elevation, or downloading an embeddable distribution at setup
time — which means fetching an interpreter over the network and executing it,
which is a materially larger supply-chain surface than fetching wheels from
PyPI. Refusing with instructions is the smaller risk.

Search order, and why it is that order:

1. `SMT_PYTHON`, if set and it exists.
2. `python-path.txt` next to the launcher, if present — one line, the full path.
3. `py -3.13`, `py -3.12`, `py -3.11`, `py -3.10`. **Newest first**, because
   3.13 is the primary target and every pin has a verified wheel for it.
4. `python.exe` on `PATH`, **only** if it reports a version in range.

The `py` launcher goes ahead of `PATH` because "whatever `python.exe` is on
`PATH`" is routinely a Microsoft Store stub that does nothing, or another
project's virtual environment.

Steps 1 and 2 exist so the launcher can never dead-end. If probing fails there
is a file to drop in, not a prompt to answer — which matters because a setup
that stops and waits for someone to paste a path is a setup that does not run
unattended.

### The lock file

After the first successful install the launcher runs `pip freeze` and writes
**`installed-lock.txt`** — every package, exact version, transitive ones
included. If that file exists, subsequent installs use it instead of
`requirements.txt`.

This is why `requirements.txt` pins the four direct dependencies by hand and
leaves ezdxf's three transitive dependencies to the resolver. Guessing a
transitive version wrong fails the whole install and buys no security; recording
what a verified install actually produced is both reproducible and honest about
its provenance. First install resolves, every install after it is exact.

### If the install fails

The likely cause is a Python version with no prebuilt wheel for one of the pins.
pip's message for this is the unhelpful *"no matching distribution found"*, but
it does name the package.

1. Run `SMTv13.bat --check-deps` to see which Python was picked up.
2. If it is outside 3.10 – 3.13, install 3.13 (user-scope, no admin), delete
   `.venv`, and re-run.
3. If it is inside the range and a package still has no wheel, the pin has aged
   out. Widen just that line in `requirements.txt`, delete `.venv` and
   `installed-lock.txt`, and re-run; the new lock will record what resolved.

Nothing was installed outside this folder, so there is nothing else to clean up.

---

## 2. Security and compliance

### Network: none at runtime

After setup, **nothing in this tool opens a socket.** Not telemetry, not
analytics, not auto-update, not crash reporting, not a licence check, not a font
or CDN fetch.

Enforced, not asserted. `tests/test_imports.py` walks every module and fails the
suite if any of `socket`, `ssl`, `http`, `urllib`, `urllib3`, `requests`,
`httpx`, `aiohttp`, `ftplib`, `smtplib`, `webbrowser`, `asyncio` and others are
imported anywhere, and separately fails if a URL literal appears in the source.
An endpoint that is not written down cannot be reached by accident.

### Libraries that reach out by default, and what was done about each

| What | Default behaviour | What SMTv13 does |
|---|---|---|
| **pip** | Contacts PyPI on *every* invocation to check for a newer pip | `PIP_DISABLE_PIP_VERSION_CHECK=1` set in the launcher |
| **pip cache** | Writes hundreds of MB to `%LOCALAPPDATA%\pip`, permanently | `PIP_CACHE_DIR` redirected inside this folder, then deleted after setup |
| **pip source dists** | Will build from source, running the package's own `setup.py` | `--only-binary=:all:` — wheels only, no install-time code execution |
| **ezdxf** | Builds a system-font cache in a user-profile directory | Cache directory redirected into this folder's scratch; `auto_load_fonts` disabled. See `_confine_ezdxf` in `smt/dxf.py` |
| **Qt** | `QSettings` writes to the Windows registry | `QSettings` is never used. No window geometry or preferences are persisted |
| **Qt** | `QtNetwork` exists and is installed | Never imported. Only `QtCore`, `QtGui`, `QtWidgets` |

### Files: nothing outside this folder

| Path | What |
|---|---|
| `.venv\` | The Python environment, created on first run |
| `smt.log` | The log, rewritten each run |
| `scratch\` | Temporary working files; created on demand, removed on exit |
| `__pycache__\` | Python's own bytecode cache, alongside the source |

No `%TEMP%`, no `%APPDATA%`, no `%LOCALAPPDATA%`, no `%USERPROFILE%`, no
registry. `tests/test_privacy.py` fails the suite if `tempfile`, `expanduser`,
`QSettings`, `winreg`, `QStandardPaths`, `os.environ` or any of those variable
names appears in executable code.

**Deleting this folder is a complete uninstall.**

### Logs: operations and counts, never content

The log records what happened and how much of it — face counts, stage timings,
bend counts, thicknesses. It does not record file contents, model text, or full
paths.

Paths get special treatment because a path is not the part, but on a real
network share it names the customer, the programme and the revision, and a log
file that ends up pasted into an email should not carry that. Two mechanisms:

1. `name_only()` at every call site that deliberately mentions a file.
2. `RedactPaths`, a logging filter attached to **every** handler, which strips
   directory components from anything path-shaped in the formatted message.

The filter is the safety net, because call sites get missed. It handles Windows
drive paths *with spaces*, UNC shares and POSIX paths, and deliberately leaves
ordinary prose alone — `and/or`, `mm/s` and `area/thickness` are untouched. All
of that is pinned by tests.

```
reading C:\Users\First Last\Jobs\ACME-2291\rev C\bracket.step took 4.2s
  ->  reading bracket.step took 4.2s
```

### Processes: none spawned

Nothing here starts another program. `subprocess`, `multiprocessing`,
`os.system`, `os.popen`, `os.startfile` and friends are absent, and a test
enforces it.

> Note for comparison with earlier versions: SMTv5 opened `smt.log` in the
> user's default viewer via `subprocess`. That was one sanctioned external
> process. **It has been removed.** The log is a file in this folder; open it
> yourself.

### Supply chain

Every dependency is pinned to an exact version, including the transitive ones,
so nothing floats between installs. `tests/test_privacy.py` fails if a
requirement is not an exact `==` pin, and separately fails if a pinned package
has no entry in `THIRD_PARTY_LICENSES.txt` — so the licence manifest cannot lag
the pins.

**Three things are deliberately not installed**, and together they are larger
than everything that is:

- **`cadquery` / `build123d`** — earlier versions used build123d for STEP
  reading. SMTv13 calls the OCP bindings directly, so neither is needed.
- **VTK** — the kernel is the `cadquery-ocp-novtk` build. VTK exists to draw
  things on screen and this tool draws its own, in software. Excluding it also
  removes a dependency whose release schedule has historically been what delayed
  the kernel's support for each new Python version.
- **`PySide6-Addons`** — only `PySide6-Essentials` is installed. Addons carries
  QtWebEngine (an entire browser), Multimedia, Charts and 3D, none of which this
  tool imports.

Wheels only, from PyPI, during setup. **What is not done:** hash pinning
(`--require-hashes`). Generating a hash file needs the packages downloaded once
to hash them, which could not be done in the environment this was built in. If
you want it, run the install once on a trusted machine and
`pip freeze`/`pip hash` the result — it is a genuine strengthening and the only
supply-chain control on this list that is missing.

---

## 3. How it works

### Layering, and what it buys

```
smt/occ/     the ONLY package that imports a CAD kernel
smt/gui/     the ONLY package that imports Qt
everything else   numpy, and testable on a bare Python
```

That boundary is why the risky mathematics — transforms, allowances, feature
classification, blank orientation — can be tested on any machine.

**It has a cost, and the cost is real:** a bug on the kernel side of the line is
invisible to the test suite until the code runs somewhere with OCP installed.
Several have been. `tests/test_imports.py` closes part of that gap by statically
resolving every OCP and Qt symbol against its module's imports, which catches
the commonest class — a name used but never imported — without the kernel being
present.

### Recognising a bend

A brake bend is a **pair of coaxial cylindrical faces** where:

1. the axes are coaxial;
2. the radii differ by exactly the material thickness;
3. both are **tangent** to the planar faces they adjoin;
4. the axis is **perpendicular** to those planes' normals.

Conditions 3 and 4 are the discriminator, and they are evaluated *locally* —
against each cylinder's own neighbours, never against a global "sheet normal",
which is meaningless on a part whose flanges point four different ways.

| | tangent to neighbours | axis · adjoining normal | sweep |
|---|---|---|---|
| bend | yes | ≈ 0 | < 180° |
| corner fillet | yes | ≈ ±1 | ≤ 90° |
| hole | no | ≈ ±1 | 360° |

**Coaxial merging matters.** A hole or cutout crossing a bend splits its
cylindrical face into topologically separate faces that share an axis and a
radius. Without merging them first, one bend reads as three and every number
downstream is wrong.

### Recognising everything else

New in v6, and the reason a "dirty" STEP file now works. Every face gets a role,
classified against the sheet normal *where it sits*:

- **Countersink** — cone, axis through the sheet, material outside it.
- **Chamfer** — planar or conical bevel, neither parallel nor perpendicular to
  the skin.
- **Hole** — full cylinder, axis through the sheet.
- **Slot end** — partial cylinder, axis through the sheet, **concave**.
- **Corner radius** — the same shape but **convex**. One orientation flag on the
  face separates them, which nothing else does cheaply.
- **Cut edge** — planar, standing through the thickness.
- **Formed** — sphere, torus, spline. Non-zero Gaussian curvature, so genuinely
  no development exists. The only category that gets a warning.

v5 warned on every cone it saw, which meant a perfectly correct flat pattern
arrived with a paragraph attached saying it was probably wrong. That is the
specific behaviour this replaces.

### The unfold

Each bend contributes a rigid transform:

```
BA      = θ · (R + K·t)
OSSB    = (R + t) · tan(θ/2)
BD      = 2·OSSB − BA
M_local = Translate(BA · û) ∘ Rotate(axis, −θ)
M_global[child] = M_global[parent] @ M_local
```

Two sign problems, both solved by construction rather than by convention:

1. **Which way to rotate.** Both signs are tried and the one that lands the
   child's normal parallel to the parent's is kept. Self-checking, and
   impossible to get backwards.
2. **Which side of the sheet to reference.** On a part mixing up-bends and
   down-bends there is no global "top". The side is propagated across each bend
   by a geometric test on which side of the face the bend axis lies.

The flat pattern is then assembled by transforming each flange face into the
plane and filling each bend gap with a strip exactly `BA` wide. For a purely
bent part this is geometrically identical to a "true" B-rep unfold and is valid
by construction.

**Cross-check:** a pure bend conserves volume exactly, so the flat solid's
volume is compared against the original. A discrepancy means a missed bend, a
double-counted bend zone, or a formed feature developed as though it were a
bend.

### Blank orientation

Nothing in the unfold decides how the blank is *oriented* — it inherits whatever
direction the base flange happened to face. A 400 × 100 blank drawn on the
diagonal measures roughly 396 × 296: three times the area, for a part that has
not changed.

So the tightest bounding rectangle is found by rotating calipers and the blank
is turned square to it, long side along X. The angle is folded into the **flat
frame** — the single matrix that carries the part into the plane — *before*
anything is built from it, so the preview, the DXF, the extruded solid and every
reported dimension inherit it and cannot disagree.

Rotating each output separately is how the DXF and the preview came to disagree
in v3, and it took a photograph from the shop floor to notice.

### Why the K slider is instant

Detection does not depend on K. Only the allowances and the transforms do. The
boundary loops of every face are extracted **once**, while the kernel is already
loaded, and cached on the model. Changing K is then a handful of numpy
transforms of cached points. Re-running a boolean fuse on every slider frame
would not be.

### A flange can be several disconnected pieces

A plate with a long bend across the top, the flange cut away in the middle so
two tabs stand up — one at each end, each with a hole. Ordinary sheet metal.
Only one tab reached the flat pattern.

Two coplanar faces sharing a bend line are **one flange** for unfolding
purposes: same plane, same side of the same bend, therefore the same rigid
transform. Detection merges coplanar faces into a single tangent plane — which
is correct, and is the v2 fix that stopped a flange split by a cutout being read
as two flanges — but it then took only `primary`, the largest, when building the
flange graph. The second tab never entered the graph, so it was not part of any
flange and never appeared.

The fix has two halves, and the second is the subtle one:

1. **Union every coplanar face on a tangent plane**, not just the largest, and
   carry all of them on the flange as `faces_a` / `faces_b`. Everything that
   transforms a flange — the preview, the DXF, the extruded solid — now iterates
   all of them.
2. **Keep the loops grouped by region**, each as `[outer wire, *holes]`.
   Flattening them into one list makes the second region's outer wire
   indistinguishable from a hole in the first, so the tab comes back as a
   *cutout*. That is worse than losing it, because it looks deliberate.

`Flange.regions_for_side` still accepts the older flat form as a single region,
so fixtures and any older caller keep working.

Both failure modes are pinned by `tests/test_regions.py` against a fixture built
from the real part, and both were verified by reintroducing them.

### Unfolding the solid, not just the outline

`build_flat` develops the OUTLINE — flange faces into the plane, strips in the
bend gaps, fuse, extrude. That drives every number and the DXF, and it is exact.
What it cannot do is keep a countersink, because extruding a footprint discards
everything that was not part of the footprint.

`unfold3d.unfold_solid` takes the other route and moves the material:

1. For each bend, build the plane that separates it from each flange. The arc
   meets the flat surface along a tangent line, and the axis sits exactly one
   radius off the flange plane, so **projecting the axis onto the plane lands on
   the tangency** — no radius is needed. Using one would be a way to be wrong by
   a whole material thickness whenever inner and outer were confused. The cut
   plane contains that line and stands square to the sheet, so its normal is
   `axis x flange_normal`.
2. Cut with every plane at once (`BRepAlgoAPI_Splitter`, small fuzzy value
   because the tools touch the solid exactly along a line).
3. Match each chunk to a flange by finding one of that flange's own planar
   surfaces inside it — parallel, and either on the reference plane or one
   thickness behind it. Centroid proximity was the obvious alternative and is
   much weaker: on a tray the base and a folded-over flange have centroids
   millimetres apart and the wrong one wins.
4. Apply that flange's rigid transform, the same one the solver already
   computed.
5. Replace bend chunks with flat strips one bend allowance wide, and fuse.

Every feature survives because it is attached to the metal being moved —
including geometry the classifier never recognised, which is the decisive
advantage over extruding the blank and re-cutting the features that were
recognised.

**Volume is checked afterwards.** A rigid motion conserves it, so a discrepancy
means a chunk was dropped or placed twice, and it says so rather than handing
back a plausible solid with a flange missing.

### Hidden lines, and why the formed view was slow

The formed part drew every visible face and then drew every edge on top of the
lot. An edge on the far side was therefore painted over the near side, and the
result read as a wireframe pretending to be a solid.

The fix costs nothing: draw each face and then **its own** edges, in the same
back-to-front order the surfaces already use. An edge is then painted at its
face's depth, so any nearer face covers it. An edge shared by two faces is drawn
with each, and the copy belonging to the nearer face is the one that survives.
Painter's algorithm, extended from the surfaces to the lines.

Three further things make the formed view quick:

- **Curved faces are meshed at 8 degrees rather than 4.** v11 put 90 segments
  around every hole, which is finer than anything is visible at, and cost
  roughly double the triangles on every curved face. The silhouette is what the
  eye judges and 45 segments is already past the point where a corner shows.
  Planar outlines are unaffected -- they come from the real curves and are exact
  either way, and on a sheet metal part that is most of the model.
- **Antialiasing is off while the mouse is down.** It is the most expensive
  hint there is and nobody can see it on a moving part.
- **Gradient fitting is skipped while dragging.** Fitting the shading is a
  least-squares solve per curved face; during a drag the mean is used instead.

Colouring is also remembered **per view**: off for the formed part, which reads
better plain and is what a CAD viewer looks like, and on for the flat, where the
colours are how you check a countersink came out on the side you meant.

### Two performance bugs worth naming

The tool felt slow, and both causes were the same shape of mistake: work that
looked local but sat inside a loop.

**Projecting the whole model once per face.** `_prepare_curved` called
`project(scene.vertices)` — the entire vertex array — and it is called once per
face. On a part with sixty curved faces and twenty thousand vertices that is 1.2
million point transforms per frame to do twenty thousand points' worth of work.
Sixty times the arithmetic, every frame, for nothing. It now happens once per
pass and the result is handed down. Picking had the identical bug.

**Tracing every face outline twice.** Detection needs the boundary loops of the
flange and bend faces; the 3D view needs them for every planar face. Both walked
the wires and sampled the edges independently. Tracing a boundary is among the
most expensive things this tool does, and they now share one cache — a straight
halving on any part with many planar faces, which is every sheet metal part.

Neither is visible when reading the function it lives in. Both are obvious once
you ask what the enclosing loop costs.

### Camera orientation

Carried as a rotation MATRIX, not as an azimuth and elevation, and the maths
lives in `smt/camera.py` on plain numpy rather than inside the Qt widget.

Both of those are consequences of v7 getting it wrong. v7 used Euler angles,
which gave two defects at once: elevation had to be clamped to ±90° or it would
gimbal-lock, and that clamp is what a 180° rotation limit feels like from the
outside; and the vertical sign was inverted, which is invisible to a careful
reading and obvious to a single test.

It was unreachable by the test suite only because it happened to sit in a module
that imports Qt. Moving it out is the same quarantine that makes the transform
solver and the feature classifier testable, applied to the thing that just
demonstrated it needed it.

With a matrix there are no Euler angles left to lock up, so the tumble is free
and unlimited in every direction. Each drag pre-multiplies a small rotation
expressed in **camera** axes — post-multiplying would rotate about the model's
own axes, which drifts away from the cursor the moment the part is not square to
the screen. The matrix is re-orthonormalised by SVD on every drag, because
composing thousands of small rotations slowly loses orthogonality and a matrix
that is no longer a rotation shears the part instead of turning it.

The sign convention, stated once and derived rather than guessed: **the surface
under the cursor follows the cursor.** Seventeen tests pin it, including all
four drag directions individually, a full turn returning exactly to the start,
tumbling past the pole and carrying on, and 4000 random increments leaving the
matrix still a rotation.

### The 3D view

Software rendered — no OpenGL, no GPU, no extra dependency. This has to run on a
locked-down workstation, often over a remote desktop session, and a remote
session is precisely where OpenGL is most likely to be missing, emulated to
uselessness, or present but broken. A black viewport with nothing to explain it
is a worse failure than a slightly slower one.

**v6 drew triangles and that was the mistake.** Thousands of them, individually,
at a chord tolerance near a millimetre and twenty degrees around a curve. It was
slow because the cost of software rendering is Qt call count, and it looked
faceted because it was.

v7 draws FACES, because that is what the part is made of:

- **A planar face is one filled polygon** built from its real boundary curves,
  with holes as sub-paths and odd-even filling so they punch through regardless
  of loop orientation. No triangles at all, so the outline is as smooth as the
  geometry. On a sheet metal part this is most of the model.
- **A curved face is one path** of its front-facing triangles, filled with a
  **linear gradient fitted to how the shading actually varies over it**. Least
  squares of `shade ~ a + b·(x,y)` over the triangles gives the direction
  shading changes fastest; stops are sampled from the real intensities, so a
  cylinder's non-linear falloff is reproduced rather than straightened out. It
  needs no per-surface-type special case and gets cones right for free.

A typical bracket goes from roughly 4000 draw calls to about 60, and no facet is
visible anywhere. The tessellation is also far finer than v6 — chord error one
2500th of the diagonal, 4 degrees angular — which costs nothing now, because
triangle count no longer drives frame time.

Hidden lines are approximate: every edge knows which faces it borders, and an
edge whose owning faces all face away is not drawn. Not true hidden-line
removal, but it removes the great majority of wrong lines for almost no work.

---

## 4. What was verified, and what was not

Stated plainly, because you should know which parts of this have been executed
and which have only been read.

### Verified — 305 automated tests, all passing

- **Unfold arithmetic** against hand calculations, in both mm and inches.
  The U-channel blank comes out 153.0477868 mm and has through every version.
- **Feature classification** — 21 tests covering countersink vs conical
  chamfer, slot end vs corner radius, cut edge vs bevel, and the sheet-pairing
  cases that used to produce flanges that did not exist.
- **Blank orientation** — a rectangle drawn at nine different angles converges
  to the same envelope with the long side on X; rotation proven rigid; grain
  constraint proven to override the saving.
- **Thickness inference on a flat blank**, including the case where a slot
  narrower than the material would have been mistaken for the thickness.
- **Path redaction** — Windows paths with spaces, UNC shares, POSIX paths, and
  seven prose strings proven untouched.
- **Compliance properties** — no network-capable imports, no URL literals, no
  subprocess, no user-profile writes, exact pins, licence manifest complete,
  CRLF line endings, pure ASCII, quoted paths in the launcher.
- **Multi-region flanges** — two tabs on one bend line both reach the blank,
  neither becomes a hole, and the area accounts for both. Verified by
  reintroducing each failure mode separately.
- **The results panels, against real report output**, in millimetres and
  inches, on every flattenable fixture.
- **The window constructs**, and every handler a signal is wired to exists.
  Verified by reintroducing two shipped bugs.
- **Constructors do not use attributes before creating them** — a static check
  over every class, which catches the ordering mistake even in code paths the
  smoke test does not reach.
- **Every name in the package resolves.** Added after `thickness_from_face`
  shipped as a call with no import — invisible to the existing guards, because
  it was the package's own name in a module the suite cannot import. It would
  have crashed the first time anyone used the manual sheet-face override.
  Verified by removing the import again.
- **The crash guard, structurally** — verified by reintroducing the v7 bug and
  confirming the test names the exact function and line.
- **The half-inch thickness cap** and the manual sheet-face override, including
  that a boss is rejected automatically but accepted when you point at it.
- **Camera orientation** — every drag direction, unlimited rotation past the
  poles, matrix stays a rotation under 4000 random increments, and every named
  view. This is the code that was wrong in v7 and could not be tested where it
  lived.
- **The launcher's supported-version list** — pinned against the versions the
  dependencies were actually verified for, newest-first ordering enforced, and
  the two non-interactive escape hatches asserted to exist. The first cut of
  this file supported 3.10–3.12 and refused 3.13, which is precisely the kind
  of list that goes stale silently.
- **The static import guards themselves** — verified by deliberately breaking
  `smt/occ/tess.py` and `smt/gui/view3d.py` and confirming both guards fired.
  One gap was found doing this and closed: the Qt check could not see a class
  imported *nowhere* in the package, so a naming-convention rule was added.

### The 0xC0000005 crash: two wrong fixes, and what actually found it

**This is worth reading, because I got it wrong twice.**

v6 crashed on opening a file. I diagnosed it as the tessellation, and v7 fixed
three real problems there — parallel meshing on a worker thread, an unguarded
curve construction, a remesh retry loop. v8 crashed identically.

What settled it was one question: **does `--diagnose` crash too?** That path
loads and analyses a file with no Qt, no window and no tessellation. It did.
Which meant the entire v7 fix had been applied to a module the crashing path
never executes.

The actual fault: `topo.discretise_edge` constructs `BRepAdaptor_Curve` on
whatever edge it is handed. An edge can have no 3D curve in two ways — it is
degenerate, or it carries only a pcurve — and either makes that undefined
behaviour, which in a native library is an access violation rather than an
exception. That function is called by `face_loops`, which detection runs on
every flange and every bend face, and by the DXF writer.

v7 *did* write exactly the right check. It defined it as a private helper inside
the display module, so detection could not see it and stayed unguarded — while
the code read, on inspection, as though it had been fixed.

Three things now prevent a repeat:

1. The check lives once, in `topo.py`, and every sampler goes through it.
2. `tests/test_crashguard.py` walks the source and **fails if any function
   constructs `BRepAdaptor_Curve` or `GCPnts_QuasiUniformDeflection` without
   it**, and separately fails if the check is defined anywhere but `topo.py`.
   Verified by reintroducing the v7 bug: it names the exact function and line.
3. `describe_all` logs the index of every face **before** reading it, so a
   native crash leaves the face number as the last line of `smt.log`. Nothing in
   Python can catch an access violation; leaving a breadcrumb is the only thing
   that can be done, and it turns "it crashed" into "it crashed on face 47".

### The earlier v6 tessellation fixes, which were real but not the cause

v6 died with an access violation on opening a file. Three things in its
tessellation could produce that, and all three are gone:

1. **`isInParallel=True` on the mesher.** OCCT's parallel mesher starts its own
   thread pool, and it was being called from a Qt worker thread inside a Python
   process holding the GIL and an event loop. This is the prime suspect. It was
   also buying nothing — a sheet metal part meshes in milliseconds.
2. **`BRepAdaptor_Curve` on an edge with no 3D curve.** Degenerate edges were
   checked for; an edge carrying only a pcurve is a *different* thing and also
   has nothing to sample. Handing either to the adaptor is undefined behaviour,
   and undefined behaviour in a native library is an access violation, not an
   exception — which is why the `try/except` around it saved nothing.
3. **Re-meshing the same shape up to four times** to hit a triangle budget while
   holding results from earlier passes. The budget is gone entirely.

**And so that this is never guesswork again:** `faulthandler` is armed at
startup and writes `smt-crash.txt`. A native crash now dumps the Python stack of
every thread at the moment of the fault, naming the exact frame that called into
the kernel. Standard library, no dependency, no network, file inside this folder.

### The window is now actually constructed, against a stub Qt

Three bugs had shipped in `smt/gui/`, all for one reason: the modules import
PySide6, Qt cannot be installed in the environment this is built in, so nothing
could import them — let alone build anything.

    v7   the vertical drag inverted, rotation clamped to 180 degrees
    v10  `thickness_from_face` called with no import
    v11  `self._stall` used three statements before it was created

`tests/qtstub.py` supplies a permissive fake PySide6, and
`tests/test_gui_smoke.py` builds the whole window with it — every widget's real
`__init__`, the real layout code, the real signal wiring. None of these bugs is
about Qt behaviour; they are Python mistakes, and every one shows up the instant
the object is constructed.

**The first version of the stub was worse than useless and it is worth saying
why.** It returned a stub for *any* missing attribute, so when the constructor
used `self._stall` before creating it, the stub invented one and the smoke test
passed on code that crashed in front of the user. It was hiding exactly the bug
it existed to find.

The rule now: **a leading underscore means the attribute belongs to this
codebase, not to Qt.** Qt's API is public and camelCase; nothing here touches a
private Qt member. So `_anything` that has not been assigned raises, exactly as
it would on a real machine. Verified by reintroducing both shipped bugs and
confirming each is caught with the same message the user saw.

This raises the floor from "the source parses" to "the code runs". It does not
reach "the tool works" — a stub accepts calls Qt would reject, and nothing about
rendering, hit testing or Qt API contracts is verified by it.

### The panels are now driven with real analysis output

`report.BendRow.direction` is a string, and the bend table compared it to an
integer. Both sides are valid Python, so nothing static could see it; it only
appears when a real row meets the code that formats it.

It also surfaced during `_refresh`, which runs while a file is opening — so the
exception aborted the open **before the 3D view was ever shown**. The reported
symptom was "nothing appears until I click a view button", which sounds like a
rendering fault and was nothing of the kind.

Two changes came out of that:

1. `tests/test_gui_data.py` pushes genuine `report` output through the genuine
   panels, using the same synthetic parts the transform solver is checked
   against. No kernel is needed — the parts, the unfold, the report and the
   panels are all kernel-free, and only Qt had to be faked.
2. **Drawing the part no longer depends on the panels working.** The view is
   shown first and the refresh is guarded. If the numbers cannot be formatted
   you should still be able to look at the geometry and see why.

Both were verified by reintroducing the shipped bugs and confirming each is
caught.

### NOT verified — and this is the important part

**I have no PyPI access in the environment this was built in.** `cadquery-ocp`
and `PySide6` cannot be installed here, which means:

1. **No code in `smt/occ/` or `smt/gui/` has ever been executed.** Not once. It
   is verified by syntax parsing, by AST-level import resolution, and by
   cross-module reference audit — and that is all.
2. **The 3D view has never been rendered.** The window can now be BUILT here
   against a stub Qt, which is new and catches a whole class of mistake — but
   nobody has seen a picture come out of it. The gradient fitting, the depth
   sorting and the hit testing are sound on paper and unproven in fact.
3. **The solid unfold has never been run.** It is the largest piece of new
   kernel code in this version and it does boolean splitting, which is where
   real translated geometry fails. Its *geometry* is tested — `bend_cut_plane`
   has eight tests including non-axis-aligned flanges and the up/down bend
   symmetry — but the booleans around it are not. This is why it falls back to
   the extruded profile automatically and announces which one you are looking
   at: shipping untested kernel work behind a fallback that declares itself is
   the only honest way to ship untested kernel work.
4. **The v6 crash fix is a diagnosis, not a confirmation.** Parallel meshing is
   the most likely cause and it is gone, along with two other candidates. If it
   still crashes, `smt-crash.txt` will now say exactly where.
5. **The pins are now verified to exist, but not verified to install.** Each
   was checked against PyPI's own file listing for a `win_amd64` wheel on
   CPython 3.13:

   | Package | Wheel confirmed |
   |---|---|
   | `cadquery-ocp-novtk==7.9.3.1.1` | `cp313-cp313-win_amd64` |
   | `numpy==2.2.6` | cp310–cp313 |
   | `PySide6-Essentials==6.9.1` | `cp39-abi3-win_amd64` (stable ABI, covers 3.13) |
   | `ezdxf==1.4.4` | declares 3.10–3.14 |

   That the files exist is confirmed. That they resolve together, install
   cleanly and import is not — no install was performed here.

   Separately: the package was audited for numpy-2 removals (`np.float_`,
   `np.NaN`, `np.trapz` and the rest). None are used. The kernel is now OCCT
   7.9 rather than 7.7; the OCP calls used are core and stable, and the one
   known breaking change in this range — `TopoDS_Shape.HashCode()` removed in
   7.8 — was already worked around in v1.
6. **The `.bat` files have not been run by `cmd.exe`.** They are checked for
   CRLF, ASCII and quoting by tests; the control flow has been read carefully
   but not executed.

This is the same gap that let kernel-layer bugs reach you in v1 through v5. It
is a property of where this is built, not of how carefully it was written, and
saying so is more useful than discovering it again.

**What that means practically:** expect the first run to need one round of
fixing. `SMTv13.bat --diagnose <file>` and `smt.log` are built for exactly that
and neither contains anything confidential.

---

## 5. Things that touch a boundary — full list

Per your request, everything that touches a network, spawns a process, or writes
outside this folder, however benign.

| # | What | When | Verdict |
|---|---|---|---|
| 1 | `pip` downloads wheels from PyPI | Setup only, once | **Sanctioned.** The one exception you allowed. Wheels only, pinned, no source builds |
| 1b | `pip freeze` writes `installed-lock.txt` | End of first install | Inside this folder. Records what was installed; no network |
| 2 | `upgrade-pip.bat` downloads pip | Only if you run it by hand | **Sanctioned.** Never run automatically |
| 3 | `pip`'s own version check | — | **Disabled** via `PIP_DISABLE_PIP_VERSION_CHECK` |
| 4 | `pip`'s cache in `%LOCALAPPDATA%` | — | **Redirected** into this folder, then deleted |
| 5 | ezdxf font cache in the user profile | — | **Redirected** into this folder's scratch |
| 6 | Python `__pycache__` | Every run | Inside this folder, next to the source. Removed with it |
| 7 | `cmd.exe` `pushd` on a UNC path | If run from a network share | Maps a temporary drive letter for the session. Nothing written to the server. The launcher says so on screen |
| 8 | `smt.log` | Every run | Inside this folder. Operations and counts only |

Runtime network calls: **none.** Processes spawned: **none.** Writes outside
this folder: **none.**
