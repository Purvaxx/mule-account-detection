// Read-only stand-in for the dashboard's Python API.
//
// The page calls /api/*; on this static site those answers were computed ahead
// of time (data/*.json). The network view is rebuilt here from per-entity
// transaction counts, using the same rules as DetectionAdapter.get_network,
// so every filter combination still works. Anything that would change data
// (uploads, mapping, running analysis) is refused with a clear message.
(function () {
  const realFetch = window.fetch.bind(window);
  const READ_ONLY_MSG =
    'This is a read-only demo of the reference dataset. Uploading and analysing ' +
    'your own data needs the full app: see the GitHub repository to run it locally.';

  const cache = new Map();
  function getJson(path) {
    if (!cache.has(path)) {
      cache.set(path, realFetch(path).then(r => {
        if (!r.ok) throw new Error(path + ' ' + r.status);
        return r.json();
      }));
    }
    return cache.get(path);
  }

  function json(body, status) {
    return new Response(JSON.stringify(body), {
      status: status || 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }
  function text(body, status) {
    return new Response(body, { status, headers: { 'Content-Type': 'text/plain' } });
  }

  // Python compares str by code point; localeCompare would not match it.
  function cmp(a, b) { return a < b ? -1 : a > b ? 1 : 0; }

  function buildNetwork(entityId, links, maxNeighbors, minTx, reciprocalOnly) {
    let scored = [];
    for (const [neighbor, inTx, outTx] of links) {
      const total = inTx + outTx;
      const recip = inTx > 0 && outTx > 0;
      if (total < minTx) continue;
      if (reciprocalOnly && !recip) continue;
      scored.push([(recip ? 1000000 : 0) + total, neighbor, inTx, outTx]);
    }
    // Python: sort(key=(score, neighbor), reverse=True)
    scored.sort((a, b) => (b[0] - a[0]) || cmp(b[1], a[1]));

    let hidden = 0;
    if (maxNeighbors > 0 && scored.length > maxNeighbors) {
      hidden = scored.length - maxNeighbors;
      scored = scored.slice(0, maxNeighbors);
    }

    const visible = new Set(scored.map(s => s[1]));
    visible.add(entityId);

    const relationships = links.map(([neighbor, inTx, outTx]) => {
      const recip = inTx > 0 && outTx > 0;
      return {
        counterparty: neighbor,
        direction: recip ? 'RECIPROCAL' : (inTx > 0 ? 'IN' : 'OUT'),
        transaction_count: inTx + outTx,
        reciprocal: recip,
        visible_in_network: visible.has(neighbor),
      };
    });
    relationships.sort((a, b) =>
      (a.visible_in_network === b.visible_in_network ? 0 : (a.visible_in_network ? -1 : 1)) ||
      (b.transaction_count - a.transaction_count) ||
      cmp(a.counterparty, b.counterparty));

    const nodes = [...visible].map(n => ({ id: n, label: String(n), is_target: n === entityId }));
    const edges = [];
    for (const [, neighbor, inTx, outTx] of scored) {
      if (outTx > 0) edges.push({ from: entityId, to: neighbor, value: outTx, title: `${outTx} transactions` });
      if (inTx > 0) edges.push({ from: neighbor, to: entityId, value: inTx, title: `${inTx} transactions` });
    }
    return { nodes, edges, hidden_count: hidden, relationships };
  }

  async function entityFile(kind, rawId) {
    const id = decodeURIComponent(rawId);
    const index = await getJson('/data/entity-index.json');
    if (!(id in index)) return null;
    return { id, data: await getJson(`/data/${kind}/${index[id]}.json`) };
  }

  async function handle(url, method) {
    const parts = url.pathname.split('/').filter(Boolean); // ['api', ...]
    if (method !== 'GET') return text(READ_ONLY_MSG, 403);

    const route = parts.slice(1);
    if (route.length === 1 && route[0] === 'theme') return json(await getJson('/data/theme.json'));
    if (route.length === 1 && route[0] === 'help') return json(await getJson('/data/help.json'));
    if (route.length === 1 && route[0] === 'overview') return json(await getJson('/data/overview.json'));
    if (route.length === 1 && route[0] === 'workspaces') return json([]);

    if (route.length === 2 && route[0] === 'entity') {
      const hit = await entityFile('entity', route[1]);
      return hit ? json(hit.data) : text('Entity not found', 404);
    }
    if (route.length === 2 && route[0] === 'network') {
      const hit = await entityFile('links', route[1]);
      if (!hit) return json({ nodes: [], edges: [], hidden_count: 0 });
      const q = url.searchParams;
      const maxN = parseInt(q.get('max_neighbors') || '15', 10);
      const minTx = parseInt(q.get('min_tx') || '1', 10);
      const recip = (q.get('reciprocal') || 'false').toLowerCase() === 'true';
      return json(buildNetwork(hit.id, hit.data, maxN, minTx, recip));
    }
    if (route[0] === 'workspaces') return text(READ_ONLY_MSG, 403);
    return text('Not available in the static demo', 404);
  }

  window.fetch = function (input, init) {
    const raw = typeof input === 'string' ? input : input.url;
    const url = new URL(raw, window.location.href);
    if (url.origin === window.location.origin && url.pathname.startsWith('/api/')) {
      const method = ((init && init.method) || (input.method) || 'GET').toUpperCase();
      return handle(url, method).catch(err => text(String(err), 500));
    }
    return realFetch(input, init);
  };

  // Exposed only so the build can be checked against real server answers.
  window.__staticApi = { buildNetwork };
})();
