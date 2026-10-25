/*
 * Canvas surfaces
 * ===============
 * The two things this console draws pixel by pixel rather than as markup: the
 * network stage, and the distribution of the model's risk scores.
 *
 * Both are pure presentation over data the adapter already returned. The graph
 * reads `predicted_class` to draw a flagged counterparty as one and the
 * histogram bands the scores it was given; neither re-scores, re-thresholds,
 * reorders or suppresses a prediction, and no threshold line is drawn because
 * the checkpoint publishes none.
 *
 * Spread into the Alpine component in `app.js`, so `this` is the component and
 * `this.$refs` resolves against the page exactly as it does there.
 */

//: How long the network is allowed to keep settling before the console stops
//: saying so.  Presentation only: physics carries on afterwards if it needs to,
//: and the projection is unaffected either way.
const GRAPH_SETTLE_CEILING_MS = 2600;

const AML_VISUALS = {
    // -------------------------------------------------------------------
    // Network stage: depth, focus and motion.
    //
    // The parallax layers sit behind the Vis-Network canvas and are
    // pointer-transparent, so hit-testing, dragging, zooming and selection
    // are exactly what Vis-Network does on a flat canvas. Depth is
    // presentation; the projection and its filters are unchanged.
    // -------------------------------------------------------------------
    onGraphPointerMove(event) {
        const stage = this.$refs.graphStage;
        if (!stage) return;
        const rect = stage.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        stage.style.setProperty('--gx', ((event.clientX - rect.left) / rect.width - 0.5).toFixed(3));
        stage.style.setProperty('--gy', ((event.clientY - rect.top) / rect.height - 0.5).toFixed(3));
    },
    resetGraphParallax() {
        const stage = this.$refs.graphStage;
        if (!stage) return;
        stage.style.setProperty('--gx', '0');
        stage.style.setProperty('--gy', '0');
    },
    //: A single expanding ring, once, where attention moved to. Created
    //: imperatively so the animation restarts on every focus change.
    fireFocusPulse() {
        const stage = document.querySelector('.graph-stage');
        if (!stage) return;
        const pulse = document.createElement('div');
        pulse.className = 'focus-pulse';
        if (this.selectedEntity && this.selectedEntity.predicted_class === 1) {
            pulse.dataset.flagged = 'true';
        }
        pulse.setAttribute('aria-hidden', 'true');
        stage.appendChild(pulse);
        setTimeout(() => pulse.remove(), 1200);
    },
    recenterNetwork() {
        if (!this.networkInstance) return;
        this.networkInstance.fit({animation: {duration: 600, easingFunction: 'easeOutQuint'}});
    },
    focusEntityInNetwork() {
        if (!this.networkInstance || !this.selectedId) return;
        const stage = this.$refs.graphStage;
        if (stage) stage.scrollIntoView({behavior: 'smooth', block: 'center'});
        this.networkInstance.focus(this.selectedId, {
            scale: 1.15,
            animation: {duration: 800, easingFunction: 'easeInOutQuad'}
        });
        this.fireFocusPulse();
    },

    renderGraph() {
        const container = document.getElementById('network-graph');
        if (!container || !this.network) return;
        // Graph adapter for the design tokens.  The encoding comes from
        // src/ui/theme.py via /api/theme, so the graph cannot drift from
        // the rest of the console the way a hardcoded palette here did.
        // Falls back to the injected CSS custom properties if the theme
        // request has not landed, and never to a literal.
        const cssToken = (name) => getComputedStyle(document.documentElement)
            .getPropertyValue(`--aml-graph-${name}`).trim();
        const g = (this.theme && this.theme.graph) || {};
        const premiumColors = {
            normalNode: g.nodeColor || cssToken('node-color'),
            normalBorder: g.nodeBorder || cssToken('node-border'),
            selectedNode: g.selectedNodeColor || cssToken('selected-node-color'),
            selectedSuspicious: g.selectedSuspiciousColor || cssToken('selected-suspicious-color'),
            flaggedNode: g.flaggedNodeColor || cssToken('flagged-node-color'),
            flaggedBorder: g.flaggedNodeBorder || cssToken('flagged-node-border'),
            edgeNormal: g.edgeColor || cssToken('edge-color'),
            edgeHighlight: g.edgeHighlight || cssToken('edge-highlight'),
            textNormal: g.labelColor || cssToken('label-color'),
            textSelected: g.labelSelected || cssToken('label-selected'),
            textMuted: g.labelMuted || cssToken('label-muted')
        };

        // Presentation only: `predicted_class` is read, never recomputed.
        const getSelectedColor = () => {
            if (!this.selectedEntity) return premiumColors.selectedNode;
            return this.selectedEntity.predicted_class === 1
                ? premiumColors.selectedSuspicious
                : premiumColors.selectedNode;
        };
        const selectedColor = getSelectedColor();
        const classOf = (id) => this.entityClassIndex ? this.entityClassIndex.get(id) : undefined;

        const nodes = new vis.DataSet(this.network.nodes.map(n => {
            const flagged = !n.is_target && classOf(n.id) === 1;
            const fill = n.is_target
                ? selectedColor
                : (flagged ? premiumColors.flaggedNode : premiumColors.normalNode);
            const edge = n.is_target
                ? selectedColor
                : (flagged ? premiumColors.flaggedBorder : premiumColors.normalBorder);
            return {
                id: n.id,
                label: n.label,
                color: {
                    background: fill,
                    border: edge,
                    highlight: { background: fill, border: selectedColor },
                    hover: { background: fill, border: selectedColor }
                },
                font: {
                    color: n.is_target ? premiumColors.textSelected : (flagged ? premiumColors.textNormal : premiumColors.textMuted),
                    size: n.is_target ? 15 : 11,
                    face: 'Inter, ui-sans-serif, system-ui, sans-serif',
                    strokeWidth: 0,
                    vadjust: 2
                },
                shape: 'dot',
                size: n.is_target ? 24 : (flagged ? 14 : 11),
                borderWidth: n.is_target ? 3 : (flagged ? 2 : 1),
                // Depth: the focused node casts, the rest recede.
                shadow: n.is_target
                    ? { enabled: true, color: selectedColor, size: 26, x: 0, y: 0 }
                    : (flagged ? { enabled: true, color: premiumColors.flaggedBorder, size: 10, x: 0, y: 0 } : false)
            };
        }));

        const edges = new vis.DataSet(this.network.edges.map(e => ({
            from: e.from,
            to: e.to,
            value: e.value,
            scaling: {
                min: 1,
                max: 7,
                label: { enabled: false }
            },
            title: e.title,
            arrows: { to: { enabled: true, scaleFactor: 0.55 } },
            color: {
                color: premiumColors.edgeNormal,
                highlight: premiumColors.edgeHighlight,
                hover: premiumColors.edgeHighlight,
                opacity: 0.75
            },
            smooth: {
                type: 'continuous',
                roundness: 0.45
            }
        })));

        const data = { nodes, edges };
        const options = {
            physics: {
                barnesHut: {
                    gravitationalConstant: -4000,
                    centralGravity: 0.3,
                    springLength: 200,
                    springConstant: 0.04,
                    damping: 0.09,
                    avoidOverlap: 0.1
                },
                stabilization: {
                    enabled: true,
                    iterations: 200,
                    updateInterval: 25
                }
            },
            interaction: {
                hover: true,
                hoverConnectedEdges: true,
                selectConnectedEdges: true,
                tooltipDelay: 200,
                zoomView: true,
                navigationButtons: false,
                keyboard: { enabled: false }
            },
            layout: {
                improvedLayout: true
            }
        };

        if (this.networkInstance) {
            this.networkInstance.destroy();
        }
        this.graphStabilizing = true;
        // Which render this is.  Alpine stores component state behind a
        // reactive Proxy, so `this.networkInstance` reads back as a *wrapper*
        // around the Vis-Network object rather than the object itself, and
        // `this.networkInstance !== instance` was therefore true even for the
        // live instance.  The settle below bailed out every time: the console
        // said "resolving layout" forever, never eased the view to fit, and
        // the resize handler -- which only refits once stabilizing is over --
        // stayed permanently disabled.  A plain integer is not proxied, so
        // comparing render generations answers the same question ("is this
        // still the current render?") without depending on object identity
        // surviving the framework.
        this.graphRenderSeq = (this.graphRenderSeq || 0) + 1;
        const renderSeq = this.graphRenderSeq;
        const instance = new vis.Network(container, data, options);
        this.networkInstance = instance;

        // Cinematic settle: the layout resolves, then the view eases to fit
        // and a single ring marks where attention moved to.
        //
        // Runs at most once per render, and only while this instance is
        // still the live one — a rapid re-selection destroys it, and
        // fit() on a destroyed network throws.  The ceiling is what keeps
        // the console from claiming it is still resolving a layout that
        // never reported itself settled.
        let settled = false;
        const settle = () => {
            if (settled) return;
            settled = true;
            if (this.graphRenderSeq !== renderSeq) return;
            this.graphStabilizing = false;
            if (this.network && this.network.nodes.length <= 1) {
                // A single node has a zero-size bounding box, which fit()
                // resolves to an arbitrary corner. Centre it explicitly.
                instance.moveTo({position: {x: 0, y: 0}, scale: 1});
            } else {
                instance.fit({animation: {duration: 900, easingFunction: 'easeOutQuint'}});
            }
            this.fireFocusPulse();
        };
        instance.once('stabilizationIterationsDone', settle);
        instance.once('stabilized', settle);
        setTimeout(settle, GRAPH_SETTLE_CEILING_MS);

        // Click a counterparty to investigate it. The selection is the same
        // one the ranked table makes; nothing about the projection changes.
        this.networkInstance.on('click', (params) => {
            const id = params.nodes && params.nodes[0];
            if (id === undefined || id === null || id === this.selectedId) return;
            this.selectEntity(id);
        });
    },

    // -------------------------------------------------------------------
    // Risk score distribution
    //
    // A histogram of the scores the model produced, each band split by the
    // model's own predicted class.  No threshold line is drawn: the
    // checkpoint publishes none, and one will not be invented here.
    // -------------------------------------------------------------------
    drawRiskSpectrum() {
        const canvas = this.$refs.riskSpectrum;
        const rows = this.overview.all_entities;
        if (!canvas || !rows || !rows.length) return;

        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) {
            // The canvas has no layout yet -- on a dataset switch the draw is
            // queued while the panel around it is still hidden.  Returning
            // here used to leave the *previous* dataset's bars on the canvas
            // under the new dataset's heading, which is the same mistake the
            // overview clears itself to avoid: one dataset's numbers shown as
            // another's.  Wipe first, then try again once the panel has been
            // laid out.  Bounded to a single retry so a permanently zero-sized
            // canvas cannot schedule frames forever.
            const ctx = canvas.getContext('2d');
            if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (!this._riskSpectrumRetry) {
                this._riskSpectrumRetry = true;
                requestAnimationFrame(() => {
                    this._riskSpectrumRetry = false;
                    this.drawRiskSpectrum();
                });
            }
            return;
        }
        this._riskSpectrumRetry = false;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.round(rect.width * dpr);
        canvas.height = Math.round(rect.height * dpr);
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rect.width, rect.height);

        const BINS = 56;
        const normal = new Array(BINS).fill(0);
        const flagged = new Array(BINS).fill(0);
        for (const row of rows) {
            const score = row.risk_score;
            if (typeof score !== 'number' || Number.isNaN(score)) continue;
            let bin = Math.floor(score * BINS);
            if (bin < 0) bin = 0;
            if (bin >= BINS) bin = BINS - 1;
            if (row.predicted_class === 1) flagged[bin]++; else normal[bin]++;
        }
        const peak = Math.max(1, ...normal.map((v, i) => v + flagged[i]));

        const token = (name) => getComputedStyle(document.documentElement)
            .getPropertyValue(`--aml-${name}`).trim();
        const colNormal = token('secondary');
        const colFlagged = token('critical');
        const colBase = token('border');

        const gap = 1;
        const slot = rect.width / BINS;
        const barW = Math.max(1, slot - gap);
        const baseline = rect.height - 1;

        // Baseline, so an empty band reads as zero rather than as absent.
        ctx.fillStyle = colBase;
        ctx.fillRect(0, baseline, rect.width, 1);

        for (let i = 0; i < BINS; i++) {
            const x = i * slot;
            const total = normal[i] + flagged[i];
            if (!total) continue;
            // Log scale: a fraud population is long-tailed, and a linear
            // axis renders every band except the mode as a flat line.
            const h = (Math.log10(1 + total) / Math.log10(1 + peak)) * (rect.height - 6);
            const flaggedH = total ? h * (flagged[i] / total) : 0;
            const normalH = h - flaggedH;

            if (normalH > 0) {
                ctx.globalAlpha = 0.7;
                ctx.fillStyle = colNormal;
                ctx.fillRect(x, baseline - normalH, barW, normalH);
            }
            if (flaggedH > 0) {
                ctx.globalAlpha = 0.95;
                ctx.fillStyle = colFlagged;
                ctx.fillRect(x, baseline - h, barW, flaggedH);
            }
        }
        ctx.globalAlpha = 1;
    }
};
