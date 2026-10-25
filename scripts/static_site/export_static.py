"""Export the dashboard as a static, read-only site (the Vercel demo).

Every answer the dashboard needs for the Reference Dataset is computed once
here and written as a JSON file. In the browser, static-api.js answers the
page's /api/* calls from those files, so the site needs no Python server.

Usage (from anywhere; needs ../dkdata.json, the same as run_ui.py):
    python scripts/static_site/export_static.py [out_dir]

out_dir defaults to <repo>/dist/static-site. It is deleted and rebuilt each run.

Output:
  index.html                composed page, with static-api.js loaded before app.js
  src/ui/...                the page's JS/CSS/vendor files, at the same URLs as the server
  static-api.js             stands in for /api/* in the browser
  data/theme.json, help.json, overview.json
  data/entity-index.json    entity id -> file number
  data/entity/<n>.json      the /api/entity/<id> response
  data/links/<n>.json       [[counterparty, in_tx, out_tx], ...]; the browser rebuilds
                            /api/network/<id> from this for any filter combination
  _checks/network-samples.json
                            real server answers for verify_static.mjs; not deployed
                            (listed in .vercelignore)
"""

import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

out = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1
       else REPO / "dist" / "static-site")

# The app resolves ../dkdata.json and src/ imports from the repo root.
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from src.ui import app as ui_app  # noqa: E402
from src.ui.help_content import HELP_DICT  # noqa: E402
from src.ui.theme import THEME  # noqa: E402

if out.exists():
    shutil.rmtree(out)
for sub in ("data/entity", "data/links", "_checks"):
    (out / sub).mkdir(parents=True)


def dump(path, obj):
    # Same encoding as APIHandler.send_json.
    path.write_text(json.dumps(obj), encoding="utf-8")


ctx = ui_app.resolver.get_analysis_context(None)
adapter = ui_app.adapter
graph = ctx.graph

dump(out / "data" / "theme.json", THEME)
dump(out / "data" / "help.json", HELP_DICT)
overview = adapter.get_overview(ctx)
dump(out / "data" / "overview.json", overview)

entity_ids = sorted(str(n) for n in graph.nodes)
index = {}
for n, eid in enumerate(entity_ids):
    index[eid] = n
    details = adapter.get_entity_details(eid, ctx)
    if details is not None:
        dump(out / "data" / "entity" / f"{n}.json", details)

    # Transactions to/from each counterparty, as adapter.get_network counts them.
    # Its nx.ego_graph follows outgoing edges only on a directed graph, so only
    # accounts this entity paid appear; accounts that only paid it do not.
    out_counts, in_counts = {}, {}
    for _, v in graph.out_edges(eid):
        out_counts[v] = out_counts.get(v, 0) + 1
    for u, _ in graph.in_edges(eid):
        in_counts[u] = in_counts.get(u, 0) + 1
    links = [
        [str(c), in_counts.get(c, 0), out_counts.get(c, 0)]
        for c in sorted(set(out_counts) - {eid}, key=str)  # sorted: same files every build
    ]
    dump(out / "data" / "links" / f"{n}.json", links)
dump(out / "data" / "entity-index.json", index)

# Server answers for a spread of entities and every filter option in the UI,
# so verify_static.mjs can check the browser rebuild against them.
top_ids = [e["entity_id"] for e in overview["top_entities"][:5]]
spread = entity_ids[:: max(1, len(entity_ids) // 15)]
samples = []
for eid in top_ids + spread:
    for max_n in (5, 15, 25, 0):
        for min_tx in (1, 2, 5, 10):
            for recip in (False, True):
                samples.append({
                    "entity_id": eid, "max_neighbors": max_n, "min_tx": min_tx,
                    "reciprocal": recip,
                    "expected": adapter.get_network(eid, ctx, max_n, min_tx, recip),
                })
dump(out / "_checks" / "network-samples.json", samples)

# The page and its assets, at the same URLs the server used.
page = ui_app.compose_index()
marker = '<script src="/src/ui/components/app.js"></script>'
assert marker in page, "app.js script tag not found in the composed page"
page = page.replace(marker, '<script src="/static-api.js"></script>\n    ' + marker)
(out / "index.html").write_text(page, encoding="utf-8")
shutil.copytree("src/ui/vendor", out / "src/ui/vendor")
shutil.copytree("src/ui/components", out / "src/ui/components",
                ignore=shutil.ignore_patterns("*.html"))
shutil.copy("src/ui/styles.css", out / "src/ui/styles.css")
shutil.copy(HERE / "static-api.js", out / "static-api.js")
# Keep the test samples and anything `vercel link` writes (.env.local holds a
# Vercel token) out of the public deployment.
(out / ".vercelignore").write_text("_checks\n.env*\n", encoding="utf-8")

print(f"Exported {len(entity_ids)} entities and {len(samples)} network checks to {out}")
