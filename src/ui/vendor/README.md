# Vendored front-end libraries

The console is a local tool that is demonstrated on machines which may have no
route to the public internet. Loading these three libraries from a CDN meant an
offline machine served an unstyled page with no interactivity and no graph, so
they are committed here and `src/ui/pages/index.html` loads them from this
directory.

These are third-party build outputs, vendored unmodified. Do not edit them; to
change a version, re-fetch from the URL below and update this file.

| File | Version | Fetched from | SHA-256 |
|---|---|---|---|
| `tailwind-3.4.17.min.js` | 3.4.17 | `https://cdn.tailwindcss.com/3.4.17` | `176e894661aa9cdc9a5cba6c720044cbbf7b8bd80d1c9a142a7c24b1b6c50d15` |
| `alpine-3.17.1.min.js` | 3.17.1 | `https://cdn.jsdelivr.net/npm/alpinejs@3.17.1/dist/cdn.min.js` | `b30997fc126d808b1a9b20ab3f504ded88df957818c02d6249bba3ec114eb0ec` |
| `vis-network-9.1.9.min.js` | 9.1.9 | `https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js` | `f53f833ddb9bf97efe856bb0637d4fe88f39e39999c7e94a4b8afc8de8a1a2e5` |

The previous CDN references were unpinned (`alpinejs@3.x.x`, `vis-network`
latest), so the console's behaviour could change without any commit here. The
versions above are the ones it is verified against.

**Alpine is pinned to 3.17.1 deliberately**: that is what the unpinned
`@3.x.x` reference was resolving to at the time these components were written
and verified, so it is the version the console's behaviour is actually known
against. Re-verify a workspace switch end to end before changing it.

Verify the files are intact with:

```bash
sha256sum src/ui/vendor/*.js
```
