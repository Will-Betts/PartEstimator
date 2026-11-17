"""Offline-by-construction, for the WHOLE package (both tools and the shell).

The sheet metal tests check `partestimator/smt`; this extends the two cheapest
and most decisive checks -- no network-capable imports, no URL literals in
code -- to the machining estimator and the unified window as well.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "partestimator"
NETWORK = {"socket", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib",
           "smtplib", "poplib", "imaplib", "telnetlib", "xmlrpc", "ssl", "pooch",
           "webbrowser"}


def modules():
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


class TestWholePackageOffline(unittest.TestCase):
    def test_no_network_capable_imports(self):
        bad = []
        for p in modules():
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module.split(".")[0]]
                bad += [f"{p.relative_to(PKG)}: {n}" for n in names if n in NETWORK]
        self.assertEqual(bad, [])

    def test_no_url_literals_in_code(self):
        rx = re.compile(r"https?://")
        bad = [f"{p.relative_to(PKG)}:{i}" for p in modules()
               for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
               if rx.search(line)]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
