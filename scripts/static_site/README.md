# Static demo site (Vercel)

Live: https://mule-account-detection-sable.vercel.app

A read-only copy of the dashboard for the Reference Dataset. It runs without
the Python server: every answer the page needs is computed ahead of time and
saved as JSON, and `static-api.js` answers the page's `/api/*` calls from those
files in the browser.

**What works:** overview, risk distribution, the ranked entity list, every
entity's dossier, and the network explorer with all of its filters.

**What doesn't:** uploading a dataset, schema mapping and Run Analysis. These
need the Python server, so the demo answers them with a "read-only demo"
message. Run `python run_ui.py` for the full app.

## Files

| File | Purpose |
|---|---|
| `export_static.py` | Runs the real pipeline once and writes the site |
| `static-api.js` | Serves `/api/*` from the exported JSON; rebuilds network views with the same rules as `DetectionAdapter.get_network` |
| `verify_static.mjs` | Checks the export: all entity files present, and the network rebuild matches real server answers |

## Rebuild and deploy

Needs the same setup as the dashboard (`pip install -r requirements.txt` and
`../dkdata.json`), plus Node.js for the check and the Vercel CLI.

```bash
python scripts/static_site/export_static.py          # writes dist/static-site
node scripts/static_site/verify_static.mjs           # must report 0 failures
npx vercel link --cwd dist/static-site --project mule-account-detection --yes
npx vercel deploy --cwd dist/static-site --prod
```

The export takes under a minute and produces about 6,900 files (~16 MB).
`dist/` is git-ignored: the site is rebuilt from the code, never committed.
The generated `.vercelignore` keeps `_checks/` and the `.env.local` that
`vercel link` writes (it holds a Vercel token) out of the deployment.

Redeploy whenever the model, the features or the dashboard change; the live
site does not update itself from GitHub.
