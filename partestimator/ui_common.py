"""Small Qt helpers shared by both tools (imported lazily; Qt required)."""
from __future__ import annotations


def fill_material_combo(combo, names: list[str], extra: list[str] | None = None) -> None:
    """Material names grouped by family, with a separator between families."""
    from . import materials_db as db

    fam = {m.name: m.family for m in db.load()}
    combo.clear()
    last = None
    for n in names:
        f = fam.get(n, "Other")
        if last is not None and f != last:
            combo.insertSeparator(combo.count())
        combo.addItem(n)
        last = f
    for n in extra or []:
        combo.insertSeparator(combo.count())
        combo.addItem(n)
    combo.setMaxVisibleItems(30)
