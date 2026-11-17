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

1. Unzip to a folder with a **short path**, such as `C:\Tools\PartEstimator`.
   The CAD kernel has deeply nested files, and Windows rejects paths longer
   than 260 characters.
2. Make sure **Python 3.12 (64-bit)** is installed. 3.11 and 3.13 also work.
3. Double-click **`Run Part Estimator.bat`**. The first run creates a `.venv`
   inside this folder and installs the pinned packages. It needs internet
   **once** and takes several minutes. After that the app opens immediately,
   with no console window.
4. Run **`Run Self-Test.bat`** once. It builds known parts in the CAD kernel
   and checks every answer. Expect *Machining self-test: PASS* and *Unit tests:
   PASS*.

You can also drop a `.step` file onto the launcher, or onto the open window.

| File | What it does |
|---|---|
| `Run Part Estimator.bat` | Starts the app. Sets up the environment on first run, and again if `requirements.txt` changes. |
| `Run Self-Test.bat` | Runs every automated check (about 300). |
| `Calibrate To My Times.bat` | Fits the machining calibration factors to your measured times. Drag your exported CSV onto it. |
| `Reset Environment.bat` | Deletes `.venv` so the next launch rebuilds it. |
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

➜ Because deburr and thin-wall terms now work, **re-run `Calibrate To My
Times.bat`** if you calibrated against v2.5.

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
  have a Windows wheel for Python 3.11–3.13. `Run Self-Test.bat` is the
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
- `partestimator/machining/data/shop_profile.csv`: machine limits, duty
  factors and calibration, including the new lathe settings.
- `python -m partestimator --classify a.step b.step` prints the
  classification from the command line.
- `python -m partestimator --sheet-cli part.step -o out` runs the full
  SMTv13 command line, with every option.
