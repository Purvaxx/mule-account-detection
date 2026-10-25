import json
import re
import http.server
import socketserver
import threading
import urllib.parse
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ui.adapter import DetectionAdapter
from src.ui.resolver import WorkspaceAnalysisResolver
from src.ui.context import WorkspaceNotFoundError, WorkspaceNotReadyError, InvalidWorkspaceError
from src.ui.config import UIConfig
from src.ui.theme import THEME, css_variables
from src.datasets.workspace_manager import WorkspaceManager
from src.datasets.parsers import ALLOWED_EXTENSIONS

adapter = DetectionAdapter()
resolver = WorkspaceAnalysisResolver()

#: Every API handler runs under this lock, so domain work is exactly as
#: serialized as it was under the single-threaded server: the resolver's cache,
#: the workspace store on disk and the prediction artifacts are each touched by
#: one request at a time.  What the lock deliberately does *not* cover is
#: static file serving, which is read-only and needs no coordination.
#:
#: That distinction is the point.  Resolving a large workspace rebuilds its
#: graph and 167-feature matrix and can take a minute or more; on a
#: single-threaded server that stalled *every* socket, so the page itself
#: stopped loading and the console looked crashed rather than busy.  Now the
#: page, its scripts and its stylesheet still load and the console can show
#: that it is working.
_API_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Page composition
# ---------------------------------------------------------------------------
#: ``<!-- INCLUDE: path -->`` in a template is replaced by that file's contents.
#: Fragments may include fragments -- a view is assembled from the same table
#: and panel fragments other views use -- so the substitution repeats until none
#: are left; a single pass would leave a nested include unresolved.
_INCLUDE = re.compile(r'<!--\s*INCLUDE:\s*(.*?)\s*-->')

#: Depth ceiling.  A fragment that included itself would otherwise expand until
#: the process ran out of memory.  The marker is left in place instead, which
#: shows up in the page rather than taking the server down.
_MAX_INCLUDE_DEPTH = 8

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _resolve_includes(content: str) -> str:
    def include_file(match):
        inc_path = match.group(1).strip()
        full_path = os.path.abspath(os.path.join(_PROJECT_ROOT, inc_path))
        try:
            with open(full_path, 'r', encoding='utf-8') as inc_f:
                return inc_f.read()
        except Exception as e:
            return f"<!-- Error including {inc_path}: {e} -->"

    for _ in range(_MAX_INCLUDE_DEPTH):
        expanded = _INCLUDE.sub(include_file, content)
        if expanded == content:
            break
        content = expanded
    return content


def compose_index() -> str:
    """The served console page: fragments resolved, design tokens injected.

    The document adapter for the design tokens lives here: ``src/ui/theme.py``
    is rendered into a ``:root`` custom-property block, which the page's
    Tailwind config and its stylesheet both resolve against.  Injecting at
    compose time rather than fetching ``/api/theme`` means the palette is
    correct on first paint.

    Kept out of the request handler so the *composed* page -- not one fragment
    of it -- is what the template tests can assert against.
    """
    with open(os.path.join(os.path.dirname(__file__), "pages/index.html"),
              'r', encoding='utf-8') as f:
        content = f.read()

    content = _resolve_includes(content)
    tokens = f"<style>\n{css_variables()}\n</style>"
    return content.replace('<!-- THEME_VARIABLES -->', tokens)


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------
# The workspace error taxonomy maps to HTTP the same way everywhere, so it is
# stated once here.  It used to be written out per route, which left the routes
# that did not repeat it silently inconsistent -- a genuinely missing workspace
# on /mapping returned 500.
#
# Anything not named here is an unexpected fault and stays 500.  A route adds
# to this only where it genuinely means something different, and says why.
#
# These are ordered tuples, not dicts, because the first isinstance match wins
# and the ordering is load-bearing: JSONDecodeError subclasses ValueError.
_WORKSPACE_ERROR_STATUS = (
    (WorkspaceNotFoundError, 404),
    (WorkspaceNotReadyError, 409),
    (InvalidWorkspaceError, 400),
)

#: An unknown workspace is a missing resource, and a malformed request body or
#: an unusable declared value is a client error -- not a server fault.
_WORKSPACE_RESOURCE_ERRORS = (
    (FileNotFoundError, 404),
    (json.JSONDecodeError, 400),
    (ValueError, 400),
)

#: predict() raises ValueError when the workspace is not READY: the request
#: conflicts with current state rather than being malformed.
_PREDICT_ERRORS = (
    (FileNotFoundError, 404),
    (ValueError, 409),
)

#: Only the workspace lookup is reclassified; a canonicalization failure
#: remains an unexpected fault, as it always has been.
_UNKNOWN_WORKSPACE_IS_404 = (
    (FileNotFoundError, 404),
)


class HttpError(Exception):
    """An outcome the handler wants reported with a specific status."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _status_for(exc: Exception, overrides) -> int:
    """Resolve an exception to a status, honouring a route's own overrides."""
    for exc_type, status in tuple(overrides) + _WORKSPACE_ERROR_STATUS:
        if isinstance(exc, exc_type):
            return status
    return 500


def workspace_configuration_view(workspace_id: str, manager: WorkspaceManager) -> dict:
    """The configuration surface for one workspace: state, options, verdict.

    GET and POST return the same shape, so the UI renders one view model and
    never has to reconcile two.  The verdict comes from
    ``src.datasets.policy``, the same function the canonicalizer uses to decide
    READY, so the configuration screen and the compatibility report cannot
    disagree about whether a declaration is honourable.
    """
    from src.datasets.parsers import load_sample
    from src.datasets.policy import (
        assess_configuration,
        collect_dataset_evidence,
        configuration_options,
        mapped_concepts,
        unresolved_concept_claims,
    )

    ws = manager.get_workspace(workspace_id)
    mapping = manager.get_mapping(workspace_id) or {}
    concept_sources = mapped_concepts(mapping)

    # Evidence comes from the real upload, bounded exactly as profiling is.
    # If it cannot be read the assessment still runs and reports that it had no
    # evidence, rather than assuming the declaration holds.
    evidence = None
    if ws.manifest.source_file:
        try:
            sample = load_sample(ws.directory / ws.manifest.source_file)
            evidence = collect_dataset_evidence(sample, concept_sources, is_exact=False)
        except Exception:
            evidence = None

    config = ws.manifest.dataset_config
    assessment = assess_configuration(
        config, concept_sources, evidence, unresolved_concept_claims(mapping)
    )

    return {
        "workspace_id": workspace_id,
        "display_name": ws.manifest.display_name,
        "config": config.to_dict(),
        "assessment": assessment.to_dict(),
        "options": configuration_options(),
        "mapped_concepts": sorted(concept_sources),
        "evidence": evidence.to_dict() if evidence else None,
    }


class APIHandler(http.server.SimpleHTTPRequestHandler):
    # -----------------------------------------------------------------------
    # Route table
    # -----------------------------------------------------------------------
    # (method, path segments, handler, error overrides).  A ``{name}`` segment
    # captures that segment and passes it to the handler as a keyword.
    #
    # Matching is on the full, exactly-sized segment list.  The previous
    # startswith/endswith ladder matched any path merely *ending* in the right
    # word, so `/api/workspaces/<a>/<b>/config` reached the config branch and
    # read `<a>` as the workspace id.
    ROUTES = (
        ("GET", ("api", "theme"), "get_theme", ()),
        ("GET", ("api", "overview"), "get_overview", ()),
        ("GET", ("api", "entity", "{entity_id}"), "get_entity", ()),
        ("GET", ("api", "network", "{entity_id}"), "get_network", ()),
        # Cycle evidence is a collection over the whole context, with the
        # entity as an optional *filter* rather than the resource — so both
        # addressing parameters ride in the query string, the same
        # ``workspace_id`` every other data-bearing route already accepts.
        ("GET", ("api", "cycles"), "get_cycles", ()),
        # One cycle's complete evidence, resolved inside the addressed context.
        # Different arity from the index route, so neither shadows the other.
        ("GET", ("api", "cycles", "{cycle_id}"), "get_cycle_detail", ()),
        ("GET", ("api", "help"), "get_help", ()),
        ("GET", ("api", "workspaces"), "get_workspaces", ()),
        ("GET", ("api", "workspaces", "{workspace_id}", "mapping"),
         "get_workspace_mapping", _WORKSPACE_RESOURCE_ERRORS),
        ("GET", ("api", "workspaces", "{workspace_id}", "config"),
         "get_workspace_config", _WORKSPACE_RESOURCE_ERRORS),
        ("GET", ("api", "workspaces", "{workspace_id}", "trust"),
         "get_workspace_trust", _WORKSPACE_RESOURCE_ERRORS),
        # Literal 'upload' has a different arity from the {workspace_id}
        # routes, so it cannot be shadowed by them.  An ingestion refusal
        # (multi-sheet workbook, empty file) keeps its existing 500.
        ("POST", ("api", "workspaces", "upload"), "post_upload", ()),
        ("POST", ("api", "workspaces", "{workspace_id}", "mapping"),
         "post_workspace_mapping", _WORKSPACE_RESOURCE_ERRORS),
        ("POST", ("api", "workspaces", "{workspace_id}", "config"),
         "post_workspace_config", _WORKSPACE_RESOURCE_ERRORS),
        ("POST", ("api", "workspaces", "{workspace_id}", "predict"),
         "post_workspace_predict", _PREDICT_ERRORS),
        ("POST", ("api", "workspaces", "{workspace_id}", "validate"),
         "post_workspace_validate", _UNKNOWN_WORKSPACE_IS_404),
    )

    @classmethod
    def match_route(cls, method: str, path: str):
        """Resolve ``(method, path)`` to a handler and its captured segments.

        Literal segments are compared as written; only captured segments are
        percent-decoded, so an encoded path cannot spell its way into a
        different route. First match wins, so a literal route must be declared
        before any same-arity route that would capture it.
        """
        segments = [s for s in path.split('/') if s]
        for route_method, pattern, handler, overrides in cls.ROUTES:
            if route_method != method or len(pattern) != len(segments):
                continue
            params = {}
            for expected, actual in zip(pattern, segments):
                if expected.startswith('{') and expected.endswith('}'):
                    params[expected[1:-1]] = urllib.parse.unquote(actual)
                elif expected != actual:
                    break
            else:
                return handler, params, overrides
        return None, None, None

    # -----------------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------------
    def end_headers(self):
        """Never let the browser cache this console.

        The page is composed from the working tree on every request, and its
        components (``app.js``, the fragments) are served by the static
        fallback with only a ``Last-Modified`` header.  A browser is free to
        treat that as heuristically fresh, which pins a running console to a
        stale component while the served HTML is current — the two then
        disagree about which state fields exist, and bindings fail at runtime.
        """
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        handler, params, overrides = self.match_route('GET', parsed.path)
        if handler:
            self.dispatch(handler, params, overrides, parsed)
        elif parsed.path in ('/', '/index.html'):
            self.serve_index()
        elif parsed.path == '/favicon.ico':
            self.serve_favicon()
        else:
            # Fallback for static assets
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        handler, params, overrides = self.match_route('POST', parsed.path)
        if handler:
            self.dispatch(handler, params, overrides, parsed)
        else:
            self.send_error(404, "Not Found")

    def dispatch(self, handler_name, params, overrides, parsed):
        """Run a handler and apply the one error contract to whatever it raises.

        A handler returns the JSON body, or raises: ``HttpError`` for an
        outcome it wants reported with a specific status, anything else for a
        fault the error contract classifies.
        """
        query = urllib.parse.parse_qs(parsed.query)
        try:
            with _API_LOCK:
                result = getattr(self, handler_name)(params, query)
        except HttpError as e:
            self.send_error(e.status, e.message)
        except Exception as e:
            self.send_error(_status_for(e, overrides), str(e))
        else:
            self.send_json(result)

    # -----------------------------------------------------------------------
    # Handlers — transport only; domain work stays in src/datasets and src/ui
    # -----------------------------------------------------------------------
    def get_theme(self, params, query):
        return THEME

    def get_help(self, params, query):
        from src.ui.help_content import HELP_DICT
        return HELP_DICT

    def get_workspaces(self, params, query):
        from src.ui.workspaces import get_workspaces_summary
        return get_workspaces_summary()

    def _context(self, query):
        return resolver.get_analysis_context(query.get('workspace_id', [None])[0])

    def get_overview(self, params, query):
        return adapter.get_overview(self._context(query))

    def get_entity(self, params, query):
        res = adapter.get_entity_details(params['entity_id'], self._context(query))
        if res is None:
            raise HttpError(404, "Entity not found")
        return res

    def get_network(self, params, query):
        max_n = int(query.get('max_neighbors', [UIConfig.MAX_VISIBLE_NEIGHBORS])[0])
        min_tx = int(query.get('min_tx', [UIConfig.DEFAULT_MIN_TX])[0])
        recip = query.get('reciprocal', ['false'])[0].lower() == 'true'
        return adapter.get_network(
            params['entity_id'], self._context(query), max_n, min_tx, recip
        )

    def get_cycles(self, params, query):
        """Layer 10 cycle evidence for the addressed context.

        ``?workspace_id=`` selects the dataset exactly as it does on
        ``/api/overview``; ``?entity_id=`` optionally filters to the cycles one
        entity participates in.

        Two outcomes are kept apart deliberately.  An ``entity_id`` the dataset
        does not contain is a **client error** — 404, the same status and
        message ``/api/entity/{id}`` uses for the same mistake.  Cycle analysis
        having failed is **not** an error of the request: the response is 200
        with ``state = "UNAVAILABLE"``, because the caller asked a well-formed
        question and the honest answer is "this evidence does not exist",
        which is different from "there are no cycles".

        Entity membership is checked against the context's graph rather than
        its prediction frame, because a cycle is a fact about the graph.
        """
        context = self._context(query)
        entity_id = query.get('entity_id', [None])[0]

        if entity_id is not None and entity_id not in context.graph:
            raise HttpError(404, "Entity not found")

        limit = self._bounded_int(
            query, 'limit', UIConfig.CYCLE_PAGE_SIZE,
            minimum=1, maximum=UIConfig.CYCLE_PAGE_MAX_SIZE,
        )
        offset = self._bounded_int(query, 'offset', 0, minimum=0, maximum=None)

        return adapter.get_cycles(context, entity_id, limit=limit, offset=offset)

    @staticmethod
    def _bounded_int(query, name, default, minimum, maximum):
        """Read one integer paging parameter, or refuse it as a client error.

        Stated explicitly rather than left to ``int()``: an unparseable value
        would otherwise surface as a 500, which reads as a server fault for
        what is a malformed request.
        """
        raw = query.get(name, [None])[0]
        if raw is None:
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise HttpError(400, f"{name} must be an integer, got {raw!r}")
        if value < minimum:
            raise HttpError(400, f"{name} must be >= {minimum}, got {value}")
        if maximum is not None and value > maximum:
            raise HttpError(400, f"{name} must be <= {maximum}, got {value}")
        return value

    def get_cycle_detail(self, params, query):
        """The complete evidence for one cycle of the addressed context.

        The id is resolved **within** the requested context and nowhere else —
        there is no global cycle registry — so an id minted by one workspace is
        simply absent in another and comes back 404.

        An unavailable analysis is not reported as a missing cycle: it returns
        200 with ``state = "UNAVAILABLE"``, because the server does not know
        whether the cycle exists and must not imply that it does not.
        """
        context = self._context(query)
        result = adapter.get_cycle_detail(context, params['cycle_id'])
        if result is None:
            raise HttpError(404, "Cycle not found")
        return result

    def get_workspace_mapping(self, params, query):
        workspace_id = params['workspace_id']
        manager = WorkspaceManager()
        ws = manager.get_workspace(workspace_id)
        mapping = manager.get_mapping(workspace_id)
        if not mapping:
            manager.map_workspace(workspace_id)
            mapping = manager.get_mapping(workspace_id)

        return {
            "manifest": ws.manifest.to_dict(),
            "profile": manager.get_profile(workspace_id),
            "mapping": mapping,
        }

    def get_workspace_config(self, params, query):
        return workspace_configuration_view(params['workspace_id'], WorkspaceManager())

    def get_workspace_trust(self, params, query):
        from src.datasets.inference import load_trust_assessment

        workspace_id = params['workspace_id']
        manager = WorkspaceManager()
        ws = manager.get_workspace(workspace_id)
        try:
            assessment = load_trust_assessment(ws.directory)
        except (ValueError, KeyError, TypeError) as e:
            # trust.json exists but is unreadable.  This must NOT look like
            # 404 "no assessment": that is how a workspace silently loses its
            # confidence banner while its scores stay visible.
            raise HttpError(
                500, f"Trust assessment for {workspace_id} is unreadable: {e}"
            )
        if assessment is None:
            raise HttpError(404, "No trust assessment for this workspace")
        return assessment.to_dict()

    def read_json_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            raise HttpError(400, "Empty payload")
        return json.loads(self.rfile.read(content_length).decode('utf-8'))

    def post_workspace_mapping(self, params, query):
        data = self.read_json_body()
        updates = data.get("updates", [data] if "source_column" in data else [])

        # The write itself is owned by WorkspaceManager, so the mapping manifest
        # gets the same atomic persistence as every other workspace artifact.
        mapping = WorkspaceManager().apply_mapping_overrides(
            params['workspace_id'], updates
        )
        return {"status": "success", "mapping": mapping}

    def post_workspace_config(self, params, query):
        from src.datasets.manifest import (
            CurrencyPolicy,
            GeographyPolicy,
            IdentitySource,
        )

        workspace_id = params['workspace_id']
        data = self.read_json_body()
        manager = WorkspaceManager()
        ws = manager.get_workspace(workspace_id)
        config = ws.manifest.dataset_config

        # Parse every supplied field before mutating anything, so a rejected
        # value cannot leave the manifest half-updated.
        pending = {}
        for key, enum_cls in (
            ("identity_source", IdentitySource),
            ("currency_policy", CurrencyPolicy),
            ("geography_policy", GeographyPolicy),
        ):
            if key in data:
                try:
                    pending[key] = enum_cls(data[key])
                except ValueError:
                    allowed = ", ".join(m.value for m in enum_cls)
                    raise HttpError(
                        400,
                        f"Invalid {key} {data[key]!r}. Expected one of: {allowed}",
                    )

        if "single_country_code" in data:
            raw_code = data["single_country_code"]
            if raw_code in (None, ""):
                pending["single_country_code"] = None
            elif not isinstance(raw_code, str):
                raise HttpError(400, "single_country_code must be a string")
            else:
                pending["single_country_code"] = raw_code.strip()

        for key, value in pending.items():
            setattr(config, key, value)

        # dataset_config is a live object on the manifest, so the save persists
        # the mutations above into manifest.json.
        ws.update()

        # identity_source decides how the graph is rebuilt, so a cached context
        # now describes different entity semantics than the workspace declares.
        resolver.invalidate(workspace_id)

        # Re-assess against the actual upload and return the same view model
        # GET returns, so the UI updates from one response.
        view = workspace_configuration_view(workspace_id, manager)
        view["status"] = "success"
        return view

    def post_workspace_predict(self, params, query):
        # Reuse the existing inference implementation; do not duplicate any
        # inference logic here.  predict_workspace() owns the READY gate,
        # prediction-status transitions and artifact persistence.
        from src.datasets.inference import predict_workspace

        workspace_id = params['workspace_id']
        manager = WorkspaceManager()
        df_preds = predict_workspace(workspace_id, manager)
        ws = manager.get_workspace(workspace_id)

        # A fresh prediction invalidates any cached analysis context so the
        # dashboard reflects the new run rather than a stale one.
        resolver.invalidate(workspace_id)

        return {
            "status": "success",
            "workspace_id": workspace_id,
            "prediction_status": ws.manifest.prediction_status,
            "prediction_timestamp": ws.manifest.prediction_timestamp,
            "model_version": ws.manifest.model_version,
            "entity_count": int(len(df_preds)),
        }

    def post_workspace_validate(self, params, query):
        import src.datasets.canonicalizer as canonicalizer

        workspace_id = params['workspace_id']
        manager = WorkspaceManager()
        report = canonicalizer.canonicalize_and_validate(workspace_id, manager)
        ws = manager.get_workspace(workspace_id)

        # Re-canonicalization rewrites canonical.parquet, so any cached
        # AnalysisContext was built from a dataset that no longer exists.  The
        # cache is consulted before every freshness check in the resolver, so
        # it must be dropped here or the dashboard keeps serving the previous
        # canonicalization indefinitely.
        resolver.invalidate(workspace_id)

        return {
            "status": "success",
            "manifest_status": ws.manifest.status.value,
            "compatibility": report.to_dict(),
        }

    def post_upload(self, params, query):
        import tempfile
        from src.datasets.manifest import WorkspaceStatus

        name = query.get('name', ['Untitled Workspace'])[0]
        filename = query.get('filename', ['upload.csv'])[0]

        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            raise HttpError(400, "Empty upload")

        # One allowlist, owned by the parser that has to read the file.
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            raise HttpError(400, f"Unsupported extension {ext}. Allowed: {allowed}")

        file_data = self.rfile.read(content_length)

        manager = WorkspaceManager()
        ws = manager.create_workspace(name)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp_path = tmp.name
        try:
            tmp.write(file_data)
            tmp.close()

            ws = manager.ingest_file(ws.manifest.workspace_id, tmp_path)
            ws.update(original_filename=filename)
            manager.profile_workspace(ws.manifest.workspace_id)
        except Exception:
            ws.update(status=WorkspaceStatus.FAILED)
            raise
        finally:
            try:
                tmp.close()
            except Exception:
                pass
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return {"status": "success", "workspace_id": ws.manifest.workspace_id}

    # -----------------------------------------------------------------------
    # Static composition
    # -----------------------------------------------------------------------
    def serve_index(self):
        """Transport only.  ``compose_index`` is what builds the page."""
        try:
            content = compose_index()
        except Exception as e:
            self.send_error(500, str(e))
            return

        body = content.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    #: The console's tab mark.  Browsers request /favicon.ico regardless of the
    #: icon the page declares, and the static fallback answered every one of
    #: them with a 404 in the operator's terminal.  Served from here rather
    #: than a committed binary so there is no opaque file in the repository.
    _FAVICON = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="7" fill="#0B0710"/>'
        '<rect x="7" y="7" width="18" height="18" rx="4" fill="none" '
        'stroke="#CF3688" stroke-width="3"/>'
        '<rect x="13" y="13" width="6" height="6" rx="1.5" fill="#CF3688"/>'
        '</svg>'
    ).encode('utf-8')

    def serve_favicon(self):
        self.send_response(200)
        self.send_header('Content-Type', 'image/svg+xml')
        self.send_header('Content-Length', str(len(self._FAVICON)))
        self.end_headers()
        self.wfile.write(self._FAVICON)

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        # Handle CORS for local dev
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))


class _ConsoleServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """The console's HTTP server: one thread per connection.

    Handlers themselves are serialized by ``_API_LOCK``, so this changes what
    the server can do *while* a long analysis runs, not how that analysis runs.
    ``daemon_threads`` keeps Ctrl+C immediate rather than waiting on
    in-flight connections.

    Tests construct ``socketserver.TCPServer`` with ``APIHandler`` directly and
    are unaffected; nothing about the handler depends on which server class
    drives it.
    """

    daemon_threads = True


def run_server():
    print(f"Loading UI environment on {UIConfig.HOST}:{UIConfig.PORT}...")
    try:
        resolver.get_analysis_context(None)
    except Exception as e:
        print(f"Failed to load detection pipeline: {e}")
        print("Ensure 'artifacts/models/experiment3_graphsage.pt' exists.")
        sys.exit(1)

    socketserver.TCPServer.allow_reuse_address = True
    with _ConsoleServer((UIConfig.HOST, UIConfig.PORT), APIHandler) as httpd:
        print(f"Serving AML Dashboard at http://{UIConfig.HOST}:{UIConfig.PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down UI.")


if __name__ == "__main__":
    run_server()
