"""Tests for Stage 6C-2: Frontend Workspace Integration."""

import os
import pytest

def get_file_content(relative_path):
    path = os.path.join(os.path.dirname(__file__), "..", relative_path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_composed_page():
    """The console page as served.

    ``pages/index.html`` is a shell that composes the views from fragments, so
    a markup expectation stated against the raw file would only be checking one
    piece of the page.  These tests assert against what the browser receives.
    """
    from src.ui.app import compose_index
    return compose_index()

def test_frontend_has_request_version_race_protection():
    """Verify requestVersion is used to protect against asynchronous request races (Requirement 16, 17)."""
    app_js = get_file_content("src/ui/components/app.js")
    
    # State has requestVersion
    assert "requestVersion: 0" in app_js, "State missing requestVersion"
    
    # selectWorkspace increments it
    assert "this.requestVersion++" in app_js, "selectWorkspace must increment requestVersion"
    
    # loadOverview uses it
    assert "const currentReq = this.requestVersion" in app_js
    assert "if (this.requestVersion !== currentReq) return;" in app_js

def test_frontend_workspace_selector_present():
    """Verify index.html has a dropdown-like custom dataset selector (Requirement 1, 2, 18)."""
    index_html = get_composed_page()
    
    assert "workspaceDropdownOpen" in index_html, "Dropdown state missing"
    assert "Reference Dataset" in index_html, "Reference Dataset option missing"
    assert "selectWorkspace('default')" in index_html, "Missing select default workspace hook"
    assert "selectWorkspace(ws.id)" in index_html, "Missing select workspace hook for list"

def test_frontend_resets_state_on_switch():
    """Verify selectWorkspace resets dataset-specific UI state (Requirement 4, 8, 9)."""
    app_js = get_file_content("src/ui/components/app.js")
    
    # Find the selectWorkspace function body roughly
    assert "this.selectedId = null" in app_js
    assert "this.selectedEntity = null" in app_js
    assert "this.network = null" in app_js
    assert "this.searchQuery = ''" in app_js
    assert "this.entityTablePage = 1" in app_js
    assert "this.counterpartyTablePage = 1" in app_js

def test_frontend_handles_non_ready_workspaces():
    """Verify non-ready workspaces show a state screen and skip loading the dashboard (Requirement 11, 12, 13, 14)."""
    app_js = get_file_content("src/ui/components/app.js")
    index_html = get_composed_page()
    
    # app.js skips loadOverview if not READY
    assert "ws.manifest_status !== 'READY'" in app_js
    assert "this.workspaceNotReadyStatus = ws.manifest_status" in app_js
    
    # index.html shows the state screen
    assert "x-show=\"workspaceNotReadyStatus && !loading && !error\"" in index_html
    assert "Dataset Not Ready" in index_html
    
    # index.html hides normal dashboard
    assert "x-show=\"!workspaceNotReadyStatus && !loading && !error\"" in index_html

def test_frontend_api_calls_pass_workspace_id():
    """Verify overview, entity, and network APIs pass workspace_id (Requirement 5, 6, 7)."""
    app_js = get_file_content("src/ui/components/app.js")
    
    assert "url += '?workspace_id=' + encodeURIComponent(this.selectedWorkspaceId);" in app_js
    assert "params.append('workspace_id', this.selectedWorkspaceId);" in app_js

def test_frontend_retains_reference_dataset_default():
    """Verify the default behaves as expected (Requirement 1, 22)."""
    app_js = get_file_content("src/ui/components/app.js")
    
    assert "this.selectedWorkspaceId = 'default'" in app_js

def test_frontend_preserves_help():
    """Verify Help system doesn't reset unnecessarily (Requirement 17, 20)."""
    app_js = get_file_content("src/ui/components/app.js")
    
    # selectWorkspace should NOT contain "this.helpData = {}" or similar
    # (Just verifying it's not present)
    select_block = app_js.split("async selectWorkspace(wsId) {")[1].split("},")[0]
    assert "helpData" not in select_block
    assert "helpModalOpen" not in select_block


# ---------------------------------------------------------------------------
# Null-safety of template bindings
#
# Alpine evaluates every binding inside an x-show subtree even while that
# subtree is hidden, and it evaluates x-if / x-for expressions eagerly.  A
# binding that dereferences a state field which is legitimately null — before a
# workspace is opened, or after leaving a view resets it — therefore throws an
# uncaught TypeError on a perfectly normal screen.  These fields are all set to
# null by app.js, so every dereference of them must be optional-chained.
# ---------------------------------------------------------------------------

#: State fields app.js assigns ``null``, paired with the template that reads them.
NULLABLE_BINDINGS = [
    ("selectedEntity", "src/ui/components/view_dashboard.html"),
    ("selectedEntity", "src/ui/components/investigation_dossier.html"),
    ("selectedEntity", "src/ui/components/entity_table.html"),
    ("mappingActiveColumn", "src/ui/components/view_mapping.html"),
    ("validationResult", "src/ui/components/view_mapping.html"),
    ("configView", "src/ui/components/dataset_config.html"),
    ("trust", "src/ui/components/trust_banner.html"),
    ("trust", "src/ui/components/view_dashboard.html"),
    ("trust", "src/ui/components/investigation_dossier.html"),
]


@pytest.mark.parametrize("field,template", NULLABLE_BINDINGS)
def test_nullable_state_is_only_dereferenced_with_optional_chaining(field, template):
    """A nullable root must never be dereferenced with a bare dot in a template."""
    import re

    content = get_file_content(template)
    # A bare `field.` that is not already `field?.`.  The name must not be part
    # of a longer identifier (e.g. `trust.py` inside a comment is excluded by
    # requiring a non-word character, and by the .py suffix check below).
    unguarded = [
        m.group(0)
        for m in re.finditer(rf"(?<![\w?.]){re.escape(field)}\.\w+", content)
        if not m.group(0).endswith(".py")
    ]
    assert not unguarded, (
        f"{template}: {field} is dereferenced without optional chaining in "
        f"{unguarded}. Alpine evaluates hidden bindings, so this throws "
        f"whenever {field} is null."
    )


def test_nullable_fields_really_are_set_to_null():
    """Guard the list above against drift: app.js must still null these out."""
    app_js = get_file_content("src/ui/components/app.js")
    for field, _ in NULLABLE_BINDINGS:
        assert f"{field} = null" in app_js or f"{field}: null" in app_js, (
            f"{field} is no longer nulled in app.js; the null-safety "
            f"expectation for its template may be stale."
        )


def test_switching_workspace_clears_the_previous_totals():
    """The header renders entity / transaction / suspicious counts beside the
    active dataset's name.  A stale `overview` during the load labels one
    dataset's numbers as another's, so it must be cleared on the loading path
    as well as on the two early returns."""
    app_js = get_file_content("src/ui/components/app.js")

    start = app_js.index("async selectWorkspace(")
    end = app_js.index("async openDatasetHub(")
    body = app_js[start:end]

    # Cleared before the await that fetches the new dataset's totals.
    clear = body.rindex("this.overview = {}")
    load = body.index("await this.loadOverview()")
    assert clear < load, "overview must be cleared before loadOverview()"


def test_loading_state_keeps_the_page_shape():
    """A bare line of text collapses the layout; the skeleton preserves it."""
    index_html = get_composed_page()
    assert 'x-show="loading"' in index_html
    # Enough skeleton blocks to stand in for the summary tiles, the network
    # stage and the ranked table, not a single placeholder bar.
    assert index_html.count("skeleton") >= 4
    assert "Resolving analysis context" in index_html
