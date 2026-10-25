// Check an exported static site before deploying it.
//
//   node scripts/static_site/verify_static.mjs [site_dir]   (default dist/static-site)
//
// 1. Every entity in the index has its entity and links files.
// 2. static-api.js rebuilds /api/network exactly as the Python server answered
//    for every sample in _checks/network-samples.json (node order aside, since
//    the server builds that list from a Python set).
// Exits non-zero on any failure.
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import vm from 'node:vm';

const site = process.argv[2] || 'dist/static-site';
const read = p => JSON.parse(readFileSync(join(site, p), 'utf8'));
let failures = 0;

const index = read('data/entity-index.json');
for (const [id, n] of Object.entries(index)) {
  for (const kind of ['entity', 'links']) {
    if (!existsSync(join(site, 'data', kind, `${n}.json`))) {
      console.error(`missing data/${kind}/${n}.json for ${id}`);
      failures++;
    }
  }
}

// Load static-api.js as the browser would, with just enough of `window`.
const window = { fetch: () => { throw new Error('no network in checks'); }, location: { href: 'http://localhost/' } };
vm.runInNewContext(readFileSync(join(site, 'static-api.js'), 'utf8'), { window, URL, Response: class {}, console });
const { buildNetwork } = window.__staticApi;

const sortedNodes = nodes => nodes.map(n => JSON.stringify(n)).sort().join('|');
const samples = read('_checks/network-samples.json');
for (const s of samples) {
  const links = read(`data/links/${index[s.entity_id]}.json`);
  const got = buildNetwork(s.entity_id, links, s.max_neighbors, s.min_tx, s.reciprocal);
  const e = s.expected;
  const same = got.hidden_count === e.hidden_count
    && sortedNodes(got.nodes) === sortedNodes(e.nodes)
    && JSON.stringify(got.edges) === JSON.stringify(e.edges)
    && JSON.stringify(got.relationships) === JSON.stringify(e.relationships);
  if (!same) {
    console.error(`network mismatch: ${s.entity_id} max=${s.max_neighbors} min_tx=${s.min_tx} reciprocal=${s.reciprocal}`);
    failures++;
  }
}

console.log(`${Object.keys(index).length} entities, ${samples.length} network samples, ${failures} failure(s)`);
process.exit(failures ? 1 : 0);
