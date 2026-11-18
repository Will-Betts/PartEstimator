# Part Estimator

One window for both kinds of part. Open a STEP file and it is classified
automatically:

- **Sheet metal** opens in the sheet metal tool (formerly **SMTv13**), which
  gives you the flat pattern, bend table and DXF/STEP/report export.
- **Machined** opens in the machining estimator (formerly **STEP Estimator
  v2.5**), which gives you geometry features, stock size, cycle-time estimate
  and CSV/JSON rows.

The two tools appear as **Sheet Metal** and **Machining** tabs in one window,
with the same light theme, the same 3D viewer and the same layout (settings and
results on the left, model on the right). Every feature of both tools is still
there. Clicking the other tab while a part is open overrides the detection and
opens the same file there.

## What's new in 1.1

- **Material library: 59 materials**, one list shared by both tabs: aerospace
  aluminum (2024, 2219, 3003, 5052, 6061, 6063, 7050, 7075, 7475, A356), carbon,
  alloy and tool steels, 300/400-series and PH stainless, A-286, Inconel
  625/718, Hastelloy X, titanium, copper alloys and eight engineering plastics.
  See *Material library* below.
- **Turning is now priced as turning.** Before, "Lathe" only swapped in the
  lathe's horsepower and kept the milling model. See *Mill vs turn* below.
- **Round parts open as Lathe automatically**, and you can still change it.
- **One look for both tabs:** light theme, the machining tab's VTK 3D viewer
  in the Sheet Metal tab too, panels on the left, and the same view buttons
  (Iso / Top / Front / Right / Fit) and mouse controls on both.

The tool runs entirely on your computer. After the one-time install it makes
no network connection, and it writes nothing outside this folder.

---

## Getting started

Everything runs from PowerShell through one script, `pe.ps1`. It works in
Windows PowerShell 5.1 and in PowerShell 7.

1. Clone or unzip to a folder with a **short path**, such as
   `C:\Tools\PartEstimator`. The CAD kernel has deeply nested files, and
   Windows rejects paths longer than 260 characters.
2. Make sure **Python 3.12 (64-bit)** is installed. 3.11 and 3.13 also work.
3. Allow local scripts once, if you have not already:

   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```

   (Or run it without changing policy:
   `powershell -ExecutionPolicy Bypass -File .\pe.ps1 ...`)
4. Set up and check:

   ```powershell
   .\pe.ps1 setup -Dev     # .venv in this folder, pinned packages, pytest
   .\pe.ps1 test -Full     # every check (about 460); expect PASS / PASS
   .\pe.ps1 gui            # open the window
   ```

   The first `setup` needs internet **once** and takes several minutes.
   After that nothing makes a network connection.

## Commands

```text
.\pe.ps1 gui [file.step] [-Wait]   open the window (detached; -Wait streams the log here; --gl)
.\pe.ps1 classify <step...>        sheet metal or machined?   --json  --cutoff IN
.\pe.ps1 sheet <step> [options]    flat pattern: DXF, STEP, report   (sheet --help)
.\pe.ps1 calibrate <csv>           fit calibration factors to measured times
.\pe.ps1 test [pytest args]        fast tests: timing model and CLI (seconds)
.\pe.ps1 test -Full                machining self-test plus the full suite
.\pe.ps1 setup [-Dev] [-Force]     create/update .venv; -Force rebuilds from scratch
.\pe.ps1 reset                     delete .venv
.\pe.ps1 py [args]                 the venv's Python, e.g. .\pe.ps1 py -m pip list
.\pe.ps1 help
```

- Any command except `setup`, `reset` and `help` sets `.venv` up first if it is
  missing, or updates it if `requirements.txt` has changed.
- Arguments after the command go to Python untouched, and **exit codes are
  passed through**, so the script works in other scripts and CI.
- Relative paths are taken from your current directory.
- Output can be piped:

  ```powershell
  (.\pe.ps1 classify .\parts\*.step --json | ConvertFrom-Json) |
      Where-Object kind -eq machined | Select-Object file, reason
  .\pe.ps1 sheet bracket.step --material AL5052 -o out -q
  ```

  `classify` expands wildcards itself, because PowerShell passes `*.step`
  through to a native program unexpanded. Keep the parentheses around
  `... | ConvertFrom-Json`: Windows PowerShell 5.1 otherwise passes the whole
  array down the pipe as one object.

**Working inside the venv.** `setup` installs this package in editable mode,
so code changes take effect without reinstalling. It also puts a
`part-estimator` command in the venv:

```powershell
.\.venv\Scripts\Activate.ps1
part-estimator --help
part-estimator classify bracket.step --json
python -m partestimator gui bracket.step    # the same CLI
```

**Virtual machines and remote desktops.** The 3D viewer needs OpenGL 3.2.
A VM without a GPU (Hyper-V, most VDI) offers only Windows' built-in
OpenGL 1.1, which used to crash the window silently at startup. `gui` now
checks the driver first (about 0.2 s). If it is too old, the window uses the
software renderer that PySide6 already ships (`opengl32sw.dll`, Mesa
llvmpipe, OpenGL 3.3), so nothing extra is downloaded. It draws on the CPU,
so very large parts rotate less smoothly. The log says which one is in use
(`3D viewer: ...`). Override it with:

```powershell
.\pe.ps1 gui --gl software    # force the software renderer
.\pe.ps1 gui --gl hardware    # always use the graphics driver
```

**Choosing the Python.** `setup` tries `py -3.12`, then `-3.13`, then
`-3.11`, then `python` on `PATH`, and accepts only 64-bit 3.11 to 3.13. To
override it, set `$env:PE_PYTHON` or put the full path to `python.exe` in
`python-path.txt` beside `pe.ps1`.

| File | What it is |
|---|---|
| `pe.ps1` | The entry point (see above). The `PIP_*` settings it applies last only while it runs; your shell is left as it was. |
| `installed-lock.txt` | Written by the first `setup`: the exact package set that resolved. Later setups install from it. It is discarded when `requirements.txt` changes. |
| `settings.json` | Created when you change Settings. Holds the sheet metal thickness cutoff. |
| `part_estimator.log` | Log of the last run (operations and counts only, no file paths). |
| `last_error.log` | Written only if the app fails to start. |

To uninstall, delete this folder.

---

## How a part is classified

| The part is… | It opens as… |
|---|---|
| **Bent**, with uniform thickness | **Sheet metal**, at any thickness |
| **Flat**, with uniform thickness **at or under 0.190"** | **Sheet metal** (a laser or punch blank) |
| Flat and uniform but **thicker than 0.190"** | **Machined** (plate) |
| **Not uniform thickness** (pockets, steps, spot faces, blocks) | **Machined** |
| Several bodies in one file | **Machined** |

0.190" matches the machining tool's own rule ("sheet bought to gauge, never
faced"). To change the cutoff, use **Settings…**.

**How uniform thickness is measured:** rays are fired straight through the
material from points spread across every face, in proportion to each face's
area. Each ray records how far it travels before leaving the part, and only
rays that leave through a face directly opposite are counted. For a part to
count as uniform:

- at least 80% of the measured area must be within ±3% of one thickness, and
- no more than 1% of it may be *thinner*.

A thinner region is a pocket, step or spot face, which marks a machined part.
Cut edges and hole walls (faces only one thickness tall) are set aside.
"Bent" means the sheet metal tool's own bend detector found a brake bend, so
detection and unfolding can't disagree.

Classification runs in a background process and usually takes well under a
second. The banner states the reason in plain words, for example *"Thickness is
not uniform: 28% of the surface is thinner than the main 0.500" wall"*.

---

## Material library

Both tabs read one file, `partestimator/data/materials.csv`, so a material
has the same name and density everywhere. Machining speeds and feeds are in
`machining/data/machining_rates.csv` under the same names: ten values per
material, five for milling and five for turning.

**Picking a material.** Both tabs use the same single dropdown. Click its
arrow for a menu of **families** (Aluminum, Steel, Stainless, ...); each
opens a submenu of its **alloys**, so no list is longer than about fifteen
entries. The current alloy is ticked and its family shown in bold. Or just
type into the box (or start typing while the menu is open): matches drop
down as you type. Every word must match the name, family, ISO group or old
short key, so `7075`, `stainless 304`, `inconel 718` and `CRS` all work.
Press Enter to take the first match, or Esc to put the old value back.

| Family | Materials |
|---|---|
| Aluminum | 2024-T3, 2024-O\*, 2219-T87, 3003-H14, 5052-H32, 6061-T6, 6061-O\*, 6063-T5, 7050-T7451, 7075-T6, 7075-O\*, 7475-T7351, A356-T6 cast |
| Steel | 1008/1010 CRS, A36 HRS, galvanized\*, 1018, 1045, 12L14, 4130, 4140 annealed, 4140 pre-hard, 4340, 8620, A2, D2, H13 |
| Stainless | 301 ¼ hard\*, 303, 304, 316, 321, 410, 440C, 15-5PH H1025, 17-4PH H1150 and H900, 17-7PH Cond A\* |
| Superalloy | A-286, Inconel 625, Inconel 718 (annealed and aged), Hastelloy X |
| Titanium | CP Grade 2, Ti-6Al-4V |
| Copper | C110, C260 brass\*, C360 brass, C932 bearing bronze, C954 aluminum bronze, C172 beryllium copper |
| Plastic | Delrin, Nylon 6/6, PEEK, Ultem 1000, UHMW-PE, PTFE, polycarbonate, acrylic |

\* sheet metal only. 34 materials are available as sheet: each carries a
minimum bend radius (across and with the grain), tensile strength for press
tonnage, the natural air-bend radius coefficient and a K-factor curve.

**Where the numbers come from.** Every machining row names its source and
URL:

- Kyocera and Haas turning tables
- milling and turning speed bands and cutting-force values (*Kc*) per material
  group from a published reference table, with a separate table for nickel
  alloys
- drill-maker charts
- Boedeker for plastics
- WIDIA cutting-force data

Where I had to **derive** a value instead of reading it directly (for
example, annealed tool steel between two published bands), the row's note
starts with **DERIVED** and says how. Bend radii come from the Aircraft Spruce
aluminum tables and the ESI material guides where available; the rest are
marked *typical, verify*. Properties not checked against a specific source are
marked *typical handbook value*.

The ten SMTv13 sheet materials and the twelve v2.5 machining materials keep
their exact previous values. Old names still work everywhere: `CRS`,
`AL5052` and `Aluminum 6061` resolve to the new names in old CSV exports, the
deductions file and the command line.

---

## Mill vs turn

**What changed.** In v2.5, choosing "Lathe" only swapped in the lathe's
horsepower; the part was still priced like an end mill clearing volume. Now
the Lathe setting prices the part as a **turn-mill with live tooling** would
make it:

| Term | How it's calculated |
|---|---|
| **OD roughing** | Stock bar minus the **turned profile**, station by station along the axis. Each station gets one pass per depth of cut, and each pass runs at its own diameter at constant surface speed. As the diameter shrinks the rpm rises until it hits `lathe_max_rpm` (default 4000); below that, surface speed falls. That is why small turned parts take longer per cubic inch. Horsepower also caps the rpm. |
| **Facing** | One pass across each end, integrated from the outside in, with the same rpm limit near the center. |
| **ID boring** | Axial holes over 0.75" are drilled to 0.75" and then bored out pass by pass. |
| **Finish turning** | All turned area (OD, bores, faces) at the feed the Ra target allows: *Ra = f² / (32 × nose radius)*, with a practical factor of 1.5, capped at the catalogue finishing feed. Tighter finish means slower feed. |
| **Live-tool milling** | Whatever sits inside the turned profile but isn't part of the part (flats, slots, cross features), milled by the 8 hp live tool at up to 6000 rpm, plus finishing of the non-turned faces. |
| **Holes** | Holes on the turning axis are drilled from the turret with the part spinning. Off-axis holes use the live tool. |
| Also charged | Internal grooves, part-off, C-axis indexing, turret indexing (1.5 s per tool, against 12 s for a mill tool change), lathe rapids and lathe load/unload. |

**How the turned profile is measured.** The part's largest radius is measured
at every station along the axis from the mesh, and the axis is snapped to the
real cylinder faces. On a test shaft the profile volume is within 0.05% of the
exact figure. A face counts as turned if it is a surface of revolution about
that axis, or a flat face square to it.

**Automatic choice.** A part opens as **Lathe** when its stock is round and at
least 50% of its surface is turned (`turn_auto_fraction`). The reason shows
under the Removal type box, and you can change the setting at any time.

**Example.** On a stepped test shaft (2.000" and 1.250" diameters, 3.5" long,
with a bore, a flat and a cross hole):

| Material | Turned | Milled |
|---|---|---|
| 6061 | 4.0 min | 13.7 min |
| 304 | 8.8 min | 58.8 min |

**Settings.** All lathe settings are new rows in `shop_profile.csv`: lathe and
live-tool horsepower and rpm limits, duty factors, nose radius, Ra practical
factor, bore threshold, part-off, turret and C-axis times, and lathe
load/unload. The training CSV gains turning columns
(`turned_area_fraction`, `turn_envelope_volume_in3`, …). Older rows stay
aligned, because columns are matched by name.

---

## Choosing the stock

The stock the part is cut from drives both roughing time and stock weight,
and the NC programmer may not use what the rules recommend. **Stock** in the
machining tab's **Part Inputs** sets it. The cycle time, stock weight, 3D
overlay, feature row and export all follow it at once.

| Stock type | What you set |
|---|---|
| **Plate / block** | Thickness, width, length |
| **Round bar** | Diameter, length (needs the part's cylinder fit) |
| **Extrusion profile** | A profile from the list, and its length |

- A new part always opens on the **recommended** stock: the existing rules,
  shown under *Stock Size*.
- **Snap to minimum** gives the part's own bounding box, with no cleanup or
  work-holding allowance. **Snap to recommended** goes back to the rules.
- Typing any size makes it **custom**. Typing the recommended or minimum
  numbers back in is recognised as such.
- A stock too small to enclose the part is flagged in red.
- The export records which was used, in `stock_basis` (`recommended`,
  `minimum`, `custom` or `extrusion`) and `stock_profile`.
- Roughing only ever removes what the stock holds (stock − part). For
  stock that encloses the part's convex hull, which includes every
  recommended size, the estimate is exactly as before.
- With **Lathe** selected, turning is priced from a round bar: choose
  **Round bar** to set its size. Other stock types show a warning.

**Extrusions.** The profile is one dropdown, like the material. Open it for
a menu of shapes (angle, channel, I-beam, tee, rectangular or round tube,
flat/square bar, hex bar) and aerospace standards (AND10133 to AND10141),
each with its sizes or dash numbers in a submenu. Or type into it, e.g.
`angle 2 x 2` or `AND10134-1205`. When you switch to Extrusion, the tool
suggests the profile that holds the part with the least material, from both
lists.

**Part orientation in stock.** For plate and extrusion stock, the part is
placed in its stock by the smallest box that fits it. When that is not how
it will be held, set it under *Part orientation in stock*:

- **Pick part face**, then click a flat face of the part in the 3D view (it
  turns green; curved faces are refused). **Pick stock face**, then click a
  face of the stock box. Either order. The two faces are made parallel, and
  the stock is refitted as tight as that allows.
- **Rotate part** X, Y and Z turn the part, in degrees, about the stock's
  length, width and thickness axes from there (right-hand rule). The part
  stays put on screen and the stock turns round it.
- The stock's faces are named by size, so if the part is longest in the
  direction you picked, the note under the controls says the face ended up
  on the stock's end or side rather than its top.
- **Automatic orientation** goes back to the smallest box. A new part always
  opens on it.
- Stock size, bounding box, weight, features and cycle time all follow the
  orientation. A custom size you typed is kept.

**AND / AMS standard extrusions.** The library holds 456 dies from the
nine Army-Navy aeronautical standards for extruded aluminum shapes. SAE's
AMS101xx standards replaced them with the same dash numbers, so the search
finds a die by either number: `AMS10134-1205` is `AND10134-1205`.

| Standard | Shape | Dies |
|---|---|---|
| AND10133 / AMS10133 | Angle, equal leg | 34 |
| AND10134 / AMS10134 | Angle, unequal leg | 57 |
| AND10135 / AMS10135 | Angle, bulb section | 17 |
| AND10136 / AMS10136 | Tee | 62 |
| AND10137 / AMS10137 | Channel | 107 |
| AND10138 / AMS10138 | Zee, equal leg | 70 |
| AND10139 / AMS10139 | Zee, unequal leg | 72 |
| AND10140 / AMS10140 | I-section | 30 |
| AND10141 / AMS10141 | Tee, bulb section | 7 |

Stock volume and weight use the area the standard lists, which includes
the radii. The drawn section has sharp corners. Each row was checked
against the AND drawings posted on everyspec.com. AND10141's table is not
posted, so its seven dies come from the Aircraft Extrusion Co. catalog
(aircraftextrusion.com), with areas computed from the section. The AND
drawings are cancelled: check a die against the current AMS revision before
ordering.

**Your own profiles (custom library).** Dies your shop or supplier makes
that are in neither list go in `custom_extrusions.csv`, beside the launcher.
With Extrusion selected, **Edit custom library** creates the file (with
instructions and two example rows) and opens it in your editor for .csv
files. Save it, then press **Reload**. The profiles appear first in the
picker, under *Custom library*, and the search and the suggested profile use
them too.

- One row per profile: a name, a shape and its dimensions in inches. The
  shapes and columns are those of the built-in lists, explained at the top
  of the file.
- `area_in2` is optional. Give the drawing's area and stock weight uses it;
  leave it blank and the sharp-cornered section is used.
- A row that cannot be used is skipped, and the panel says which line and
  why. A typo never stops the tool. A name that is already a built-in
  profile is refused, so a part number always means one die.
- The file is yours. It is not overwritten, is kept out of git, and like
  everything else the tool keeps, it lives in the tool folder. Copy it
  between installs to share a library.

- The profile is fitted to the part using its mesh. It is turned, flipped
  and slid within the part's section to the pose that holds the most of the
  part. It counts as holding the part when 98% of the part's vertices lie
  inside its section, and the 3D view draws it in that pose.
- It runs along whichever direction gives the shortest cut.
- **Minimum length** is the part's length along the profile.
  **Recommended length** is that plus 1.00", rounded up to the next 1/4"
  (the same rule as L for plate).
- The length is editable, and the two snap buttons set it to the minimum or
  the recommended length.

The profiles are in `partestimator/machining/data/extrusions.csv`: one row
per size, with the shape and its nominal dimensions in inches (`a_in`,
`b_in`, `t_in`, `t2_in`; the file's module docstring says which is which
for each shape). Add the sizes your suppliers stock. A row with impossible
dimensions stops the tool loading the file, so a typo cannot go unnoticed.
Section areas are computed with **sharp corners**, so check the shipped
sizes against your supplier's catalog.

---

## Sheet metal DXF: what changed

The DXF now contains **only the flat pattern and the bend lines**, and it is
**always in inches**.

| Layer | Contents | Line type |
|---|---|---|
| `OUTLINE` | Outer profile | Continuous |
| `HOLES` | Every internal cutout | Continuous |
| `BEND` | Every bend centerline | **Up = `DASHED`, Down = `PHANTOM`** |

There is no text, no dimensions and no blocks. Lines and arcs are written as
true `LINE` / `ARC` / `CIRCLE` entities.

**About the oversized DXFs.** The geometry numbers in the old files were
correct: on test parts the outline read back at exactly the blank size. What
was wrong was the file *around* the geometry:

- It was created by `ezdxf` as a **metres** drawing, with metric text styles,
  dimension styles, line types and paper layout. Only afterwards was the unit
  header switched to inches.
- Its header extents were left at the ±1e20 placeholders.
- It carried text notes.

That mismatch is the most likely reason SolidWorks and eDrawings scaled it
wrongly. The file is now built as an inch drawing from the start, with real
extents and nothing else in it.

I could not open a file in SolidWorks from here. **Please check the first DXF
you export.** In eDrawings, use Measure on a known edge.

`tests/test_dxf_scale.py` reopens every exported DXF and checks it against
the developed blank:

- units are inches
- the outline extents match the blank to 0.0001"
- hole diameters and positions are correct
- bend line lengths are correct
- up and down bends use different line types
- the file contains no text

---

## Bugs found and fixed while merging

Building real test parts and running both tools against a live CAD kernel
turned up bugs in the originals. These fixes are included.

**Machining estimator. Some of these change your numbers.**

- **Edge convexity was wrong on every part.** Half of a plain box's edges were
  counted as inside corners, because the face orientation was flipped twice.
  **Deburr time is driven by this**, so deburr estimates will change. Hole
  edges are now correctly counted as outside corners (where burrs form).
- **The thin-wall scan rejected every real wall.** It tested for the opposite
  normal direction, so it found nothing. **The thin-wall finishing derate now
  actually applies** to thin-walled parts.
- **Moment of inertia was shifted twice**, which made it negative for any part
  not centered on the origin.
- **Normals ignored part placement.** Faces carrying a placement transform
  (common in STEP exports and instanced bodies) used unrotated normals.
- Two stale self-test expectations were corrected. One claimed hole edges are
  inside corners; the other checked for a weight column removed in Rev 2.0.

➜ Because deburr and thin-wall terms now work, **re-run `.\pe.ps1 calibrate`** if you calibrated against v2.5.

**Sheet metal tool**

- **The window updated itself from a background thread.** On a real display
  this crashed when opening a file. Results are now passed back to the
  window's own thread.
- **The bounding-box drawing function was missing**, so the flat view's
  painting stopped part-way through. The box and its L/W/T labels now draw.
- **The "move the real material" solid unfold never ran.** An unsupported call
  crashed it every time, so it silently fell back to the extruded blank. It
  now runs, but on test parts it produced several loose pieces. A strict check
  now accepts it only when it yields **one solid, one sheet thick, within 1%
  volume**. Otherwise the extruded blank is used as before, so exports are
  what you got before. Dimensions were never affected.
- An edge with no 3D curve could hang the DXF writer in an endless loop.

---

## Things you should know

- **The mouse wheel never changes an input.** Over a dropdown, number box or
  slider it scrolls the panel instead, so scrolling down the settings cannot
  change a value by accident. Change inputs by clicking, typing or with the
  arrow keys.
- **Each tab keeps everything you enter in one place**, under *Part Inputs*
  at the top of the left panel, with the results below it. In the machining
  tab, the 3D display toggles moved to the bar under the viewer. When a new
  part opens, its stock goes back to the recommended size and *Tapped holes*
  goes back to *auto*. Material, finish, tolerance and operations carry
  over.

- **The machining tool still uses its separate analysis process** for loading
  (so a big part can't freeze the window, and Cancel works). The sheet metal
  tool works as before.
- **Both tabs now share one look and one viewer.** The Sheet Metal tab's own
  software renderer was replaced by the machining tab's VTK viewer, so mouse
  controls changed there to left-drag rotate, middle-drag pan, wheel zoom.
  "Pick the sheet face…" is now a plain click (no drag) on the face.
- pyvista brings `requests` / `pooch` into the environment for its optional
  example datasets. Nothing in this tool imports them, and
  `tests/test_offline_all.py` fails if anything ever does.
- **Not yet verified on Windows.** Everything was built and tested on Linux
  with Python 3.12 and the exact pinned packages. Every pin was confirmed to
  have a Windows wheel for Python 3.11–3.13. `.\pe.ps1 test -Full` is the
  check on your machine.
- **Not verified against your real parts.** No production STEP files were
  available, so all testing used parts built in the kernel with known
  dimensions: L-bracket, Z-section, U-channel, countersunk bracket, flat
  blanks from 0.075" to 0.250", pocketed plates, a spot-faced plate, a block
  and a turned part. If a real part is classified wrongly, click the other tab,
  and send me the banner text.

## Reference

- `partestimator/machining/step_estimator_system_spec.pdf`: the machining
  estimator spec (Rev 2.5).
- `partestimator/smt/SYSTEM_SPECS_SMTv13.md`: the sheet metal tool spec.
- `partestimator/data/materials.csv`: the shared material library.
- `partestimator/machining/data/machining_rates.csv`: every speed and feed,
  each row with its source URL.
- `partestimator/machining/data/extrusions.csv`: the extrusion profiles
  offered as stock (see *Choosing the stock*).
- `partestimator/machining/data/standard_extrusions.csv`: the AND / AMS
  standard extrusions, one row per dash number, with each row's source.
- `custom_extrusions.csv` (beside the launcher, created on first use): your
  own extrusion profiles.
- `partestimator/machining/data/shop_profile.csv`: machine limits, duty
  factors and calibration, including the new lathe settings.
- `.\pe.ps1 classify a.step b.step` prints the classification from the
  command line (`--json` for every measurement).
- `.\pe.ps1 sheet part.step -o out` runs the full SMTv13 command line, with
  every option (`.\pe.ps1 sheet --help`).
- The older `python -m partestimator --classify ...` / `--sheet-cli ...`
  forms still work.

## Development: unit tests

The timing model (`machining/geometry/cycle_time.py`, `turn_time.py`, the rate
and shop-profile loaders, and `calibrate.py`) has its own unit tests in
`tests/timing/`. They need only Python and numpy, with no CAD kernel and no Qt,
so they run in seconds. GitHub Actions runs them on every push and pull
request (`.github/workflows/tests.yml`, Windows and Linux, Python 3.11 to 3.13).

Run them, from this folder:

```powershell
.\pe.ps1 test                      # timing model + CLI tests (installs pytest on first use)
.\pe.ps1 test -k turn -x           # any pytest arguments pass through
.\pe.ps1 test tests/test_cli.py    # a single file
```

CI uses a separate, minimal environment with no CAD kernel and no Qt. To
reproduce it locally:

```powershell
python -m venv .venv-test
.venv-test\Scripts\python -m pip install -r requirements-dev.txt
.venv-test\Scripts\python -m pytest
```

The arithmetic tests use a small synthetic rate table and shop profile
(`tests/timing/support.py`), so retuning `machining_rates.csv` or
`shop_profile.csv` does not break them. The shipped data files are checked
separately for completeness and for sensible behaviour, such as harder
materials taking longer, but no exact minutes are pinned against them.

The full suite, including the CAD-kernel and GUI tests, runs through
`.\pe.ps1 test -Full`, which also picks up `tests/timing/`. It makes the same
OpenGL choice as `gui`, so the GUI tests also run on a VM without a GPU.
Running pytest over `tests/` directly skips that choice, so there the GUI
tests need a real OpenGL 3.2 display. To leave them out:

```powershell
.\pe.ps1 test tests --ignore=tests/test_gui_smoke.py --ignore=tests/test_unified_window.py
```
