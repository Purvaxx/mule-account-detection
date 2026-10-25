"""
Design tokens — the single place the investigation console's visual language
is defined.
===========================================================================

Every colour the UI renders comes from this module, through exactly two
adapters:

* the **document adapter** — :func:`css_variables` renders these tokens as CSS
  custom properties, which ``src/ui/app.py`` injects into ``pages/index.html``
  at compose time.  The Tailwind config in that page resolves its palette from
  those variables, so utility classes and hand-written CSS share one source.
* the **graph adapter** — ``/api/theme`` serves :data:`THEME`, and
  ``components/visuals.js`` builds its Vis-Network options from
  ``THEME["graph"]``.

Two adapters is what makes this a real seam rather than a hypothetical one.
Previously the palette existed in three disagreeing places — this dict, an
inline ``tailwind.config`` block, and a hardcoded ``premiumColors`` literal in
``app.js`` — and this dict was the one with no consumers.  Editing this file
changed nothing on screen, which is precisely what ``docs/UI_ARCHITECTURE.md``
told a maintainer to do for a redesign.

Visual direction: a premium financial-intelligence workstation.  A true black
canvas, deep plum (``#75226E``) used as *depth* rather than as fill — washes,
glass and edges at restrained opacity — a pink primary accent spent only on
identity and the one primary action per surface, a calm analytical azure for
data affordances and graph focus, near-white body copy for readability at
small sizes, and red reserved exclusively for risk and failure so that seeing
red always means something.

The module carries more than colour.  A design system also needs radii,
elevation, rhythm and timing to be stated once, so :data:`RADII`,
:data:`SHADOWS`, :data:`SPACING` and :data:`MOTION` are emitted as custom
properties alongside the palette.  They are document-level only: the graph
adapter has no use for them, and :data:`THEME`'s shape is part of the API
contract.

This module is presentation only.  It carries no thresholds, no scores and no
analytical meaning: :data:`RISK` names the colour a risk state is drawn in, it
never decides which state applies.
"""

# ---------------------------------------------------------------------------
# Canvas — true black, with plum used as depth rather than as fill
# ---------------------------------------------------------------------------
SURFACES = {
    #: The page itself.
    "background": "#080808",
    #: Default panel.  The canvas lifted by a trace of the deep plum, so
    #: panels read as *lit* rather than as a different hue.
    "surface": "#120B12",
    #: A panel that sits above another panel.
    "surfaceElevated": "#1A101B",
    #: The deep plum itself.  Used for washes, gradient edges and the graph's
    #: depth field — never as a flat fill, which is what turns a restrained
    #: accent into a pink interface.
    "surfaceDeep": "#75226E",
    #: Deep plum glass, for overlays and bars laid over content.
    "surfaceGlass": "rgba(117, 34, 110, 0.16)",
    #: Inset wells — table headers, code blocks, the graph canvas.
    "surfaceSunken": "#0A060A",
    #: Hover state for an interactive surface.  Lifts rather than tints, so
    #: hover never reads as a selection or a status.
    "surfaceHover": "#241628",
    "border": "#2A1B2C",
    "borderStrong": "#4A2B48",
}

# ---------------------------------------------------------------------------
# Type hierarchy — white is a signal, body copy is near-white for legibility
# ---------------------------------------------------------------------------
TEXT = {
    #: Reserved for headings and the single most important value on a surface.
    "heading": "#FFFFFF",
    #: Body copy.  Near-white: this console is read at 11–13px in a dark room.
    "body": "#F3F3F3",
    #: Supporting detail, labels, units.
    "muted": "#A79FAA",
    #: Disabled state and de-emphasised metadata.  Light enough to clear 4.5:1
    #: against every surface token — this console renders explanatory metadata
    #: at 10px, and the quietest step in the hierarchy still has to be read.
    "faint": "#857A8A",
}

# ---------------------------------------------------------------------------
# Accents
# ---------------------------------------------------------------------------
ACCENTS = {
    #: Primary. Restrained: identity, the active selection, one primary action
    #: per surface. Never used to mean "danger".
    "primary": "#CF3688",
    "primarySoft": "#F286BC",
    "primaryDim": "rgba(207, 54, 136, 0.12)",
    "primaryEdge": "rgba(207, 54, 136, 0.42)",
    #: Secondary, analytical: data affordances, graph focus, links, metrics.
    #: Deliberately a calm azure rather than a saturated cyan — a bright cyan
    #: beside a pink primary reads as a gaming palette, not a forensic one.
    "secondary": "#59A5D8",
    "secondarySoft": "#A9D3EE",
    "secondaryDim": "rgba(89, 165, 216, 0.12)",
    "secondaryEdge": "rgba(89, 165, 216, 0.38)",
}

# ---------------------------------------------------------------------------
# States — red is spent only on risk and failure
# ---------------------------------------------------------------------------
STATUS = {
    "critical": "#E25555",
    "criticalDim": "rgba(226, 85, 85, 0.12)",
    "criticalEdge": "rgba(226, 85, 85, 0.42)",
    "warning": "#D9A03C",
    "warningDim": "rgba(217, 160, 60, 0.12)",
    "warningEdge": "rgba(217, 160, 60, 0.38)",
    "success": "#3FB98A",
    "successDim": "rgba(63, 185, 138, 0.12)",
    "successEdge": "rgba(63, 185, 138, 0.36)",
    "info": "#8189D8",
    "infoDim": "rgba(129, 137, 216, 0.12)",
}

# ---------------------------------------------------------------------------
# Risk presentation
# ---------------------------------------------------------------------------
#: How each state is *drawn*.  Which state applies is decided by the detection
#: layer and the trust assessment; this only names the paint.
RISK = {
    "suspicious": STATUS["critical"],
    "suspiciousDim": STATUS["criticalDim"],
    "normal": TEXT["muted"],
    #: Confidence levels from src/datasets/trust.py.  There is no HIGH, and no
    #: colour here implies one: UNVERIFIED is the ceiling and is drawn calm,
    #: not affirmative green.
    "trustLow": STATUS["critical"],
    "trustLimited": STATUS["warning"],
    "trustUnverified": TEXT["muted"],
}

# ---------------------------------------------------------------------------
# Graph encoding
# ---------------------------------------------------------------------------
GRAPH = {
    "canvas": SURFACES["surfaceSunken"],
    #: The plum wash painted behind the network to give the field depth.
    "depthWash": SURFACES["surfaceDeep"],
    "nodeColor": "#2B1E30",
    "nodeBorder": "#4A3450",
    #: The entity under investigation — analytical focus, not a risk claim.
    "selectedNodeColor": ACCENTS["secondary"],
    #: The investigated entity when the model flagged it.  Red, because this
    #: is a risk state; the pink accent never carries that meaning.
    "selectedSuspiciousColor": STATUS["critical"],
    #: A *counterparty* the model flagged.  Same meaning as the table's
    #: SUSPICIOUS badge — model output, drawn quieter than the focused node so
    #: focus and risk stay separable.
    "flaggedNodeColor": "#3E1A1F",
    "flaggedNodeBorder": STATUS["critical"],
    "edgeColor": "rgba(167, 159, 170, 0.24)",
    "edgeMuted": "rgba(167, 159, 170, 0.07)",
    "edgeHighlight": ACCENTS["secondary"],
    "labelColor": TEXT["body"],
    "labelSelected": TEXT["heading"],
    "labelMuted": TEXT["faint"],
}

TYPOGRAPHY = {
    "sans": "Inter, ui-sans-serif, system-ui, sans-serif",
    "mono": "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
}

# ---------------------------------------------------------------------------
# Form — radii, elevation, rhythm and timing
#
# Document-level only.  The graph adapter has no use for them, and THEME's
# shape is part of the /api/theme contract, so they are emitted as custom
# properties and consumed by the Tailwind config and the page's own CSS.
# ---------------------------------------------------------------------------
#: Restrained. A forensic console is squared off; only pills and the focus
#: halo are round.
RADII = {
    "xs": "2px",
    "sm": "4px",
    "md": "8px",
    "lg": "14px",
    "pill": "999px",
}

#: Elevation is depth, not glow. Shadows are black and soft; the only
#: coloured one is the focus halo, which marks the analytical selection.
SHADOWS = {
    "raise": "0 1px 0 0 rgba(255, 255, 255, 0.03) inset, 0 8px 24px -12px rgba(0, 0, 0, 0.9)",
    "panel": "0 24px 60px -40px rgba(0, 0, 0, 1), 0 2px 12px -6px rgba(0, 0, 0, 0.8)",
    "overlay": "0 40px 120px -30px rgba(0, 0, 0, 1)",
    "inset": "inset 0 1px 24px rgba(0, 0, 0, 0.55)",
    "focus": "0 0 0 1px rgba(89, 165, 216, 0.55), 0 0 0 4px rgba(89, 165, 216, 0.14)",
}

#: A single 4px rhythm, named so layout spacing is a decision made once.
SPACING = {
    "gutter": "1.5rem",
    "section": "1.75rem",
    "panel": "1.25rem",
    "tight": "0.75rem",
}

#: Motion is short and eased out. Nothing bounces; nothing takes longer than
#: it takes to read where something moved to.
MOTION = {
    "fast": "120ms",
    "base": "200ms",
    "slow": "420ms",
    "cinematic": "900ms",
    "ease": "cubic-bezier(0.22, 0.61, 0.36, 1)",
    "easeOut": "cubic-bezier(0.16, 1, 0.3, 1)",
}


def _flat_tokens() -> dict:
    """All document-level tokens in one namespace, for CSS variable output."""
    tokens = {}
    for group in (SURFACES, TEXT, ACCENTS, STATUS):
        tokens.update(group)
    for name, value in RISK.items():
        tokens[f"risk-{name}"] = value
    return tokens


def _kebab(name: str) -> str:
    out = []
    for ch in name:
        if ch.isupper():
            out.append('-')
            out.append(ch.lower())
        else:
            out.append(ch)
    return ''.join(out)


def _rgb_channels(value: str):
    """``"#CF3688"`` -> ``"207 54 136"``, or ``None`` if not a plain hex.

    Tailwind composes an opacity modifier (``bg-system-surface/50``) by
    substituting into ``rgb(<channels> / <alpha-value>)``, which needs the
    channels unwrapped.  Tokens that are already ``rgba(...)`` carry their own
    alpha and are used directly.
    """
    if not (isinstance(value, str) and value.startswith("#") and len(value) == 7):
        return None
    try:
        r, g, b = (int(value[i:i + 2], 16) for i in (1, 3, 5))
    except ValueError:
        return None
    return f"{r} {g} {b}"


def css_variables() -> str:
    """Render the tokens as a ``:root`` custom-property block.

    Injected into the served page, so the Tailwind config and any hand-written
    CSS resolve the same values this module defines — without a flash of a
    differently-styled palette while ``/api/theme`` is in flight.

    Each solid colour is emitted twice: ``--aml-<name>`` for direct use in CSS,
    and ``--aml-<name>-rgb`` as bare channels so Tailwind can apply an opacity
    modifier to it.
    """
    lines = []

    def emit(prefix: str, name: str, value: str) -> None:
        var = f"--aml-{prefix}{_kebab(name)}"
        lines.append(f"  {var}: {value};")
        channels = _rgb_channels(value)
        if channels:
            lines.append(f"  {var}-rgb: {channels};")

    for name, value in _flat_tokens().items():
        emit("", name, value)
    for name, value in GRAPH.items():
        emit("graph-", name, value)
    for name, value in RADII.items():
        emit("radius-", name, value)
    for name, value in SHADOWS.items():
        emit("shadow-", name, value)
    for name, value in SPACING.items():
        emit("space-", name, value)
    for name, value in MOTION.items():
        emit("motion-", name, value)

    lines.append(f"  --aml-font-sans: {TYPOGRAPHY['sans']};")
    lines.append(f"  --aml-font-mono: {TYPOGRAPHY['mono']};")
    return ":root {\n" + "\n".join(lines) + "\n}"


def tailwind_color(name: str) -> str:
    """The Tailwind config value for a token, opacity modifiers included."""
    return f"rgb(var(--aml-{_kebab(name)}-rgb) / <alpha-value>)"


#: Served at ``GET /api/theme``.  The graph adapter in ``app.js`` reads
#: ``graph``; ``colors`` is the same token set the CSS variables carry, so a
#: future surface can consume it without re-deriving anything.
THEME = {
    "colors": {**SURFACES, **TEXT, **ACCENTS, **STATUS},
    "risk": RISK,
    "graph": GRAPH,
    "typography": TYPOGRAPHY,
}
