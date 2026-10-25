document.addEventListener('alpine:init', () => {
    Alpine.data('amlApp', () => ({
        currentView: 'dashboard',
        mappingWorkspaceId: null,
        mappingData: null,
        mappingProfile: null,
        mappingManifest: null,
        mappingActiveColumn: null,
        validationResult: null,
        validating: false,
        newDatasetName: '',
        uploadStatus: 'SELECTING',
        uploadProgressMsg: '',
        uploadError: null,
        dragActive: false,
        uploadedDatasetSummary: null,
        requestVersion: 0,
        workspaceNotReadyStatus: null,
        workspaceDropdownOpen: false,
        // Run Analysis state. READY (compatibility) and COMPLETE (predictions
        // exist) are deliberately distinct: reaching READY does not mean any
        // prediction has been produced.
        analysisRunning: null,
        analysisError: null,
        analysisErrorId: null,
        // Explicit dataset configuration (identity / currency / geography).
        // configView holds the single view model the API returns for both GET
        // and POST, so the panel never reconciles two shapes.
        configView: null,
        configSaving: false,
        configError: null,
        configDirty: false,
        // Prediction trust for the open workspace. Null when the run predates
        // trust assessment or no run exists.
        trust: null,
        trustExpanded: false,
        // Dossier progressive disclosure: one expanded heuristic flag at a time.
        openFlag: 0,
        // Graph state.  Layout settling is presentation only; it says nothing
        // about the analysis.
        graphStabilizing: false,
        //: Which graph render is current.  Declared here so it is part of the
        //: component's own state; renderGraph() bumps it and the settle
        //: callback compares generations rather than object identity, which
        //: Alpine's reactive Proxy does not preserve.
        graphRenderSeq: 0,
        //: entity_id -> predicted_class, built from the overview the model
        //: already returned.  The graph reads it so a flagged counterparty is
        //: drawn as one; it is the same prediction the table shows, never a
        //: second opinion computed here.
        entityClassIndex: null,

        // The design system's status vocabulary and the two canvas surfaces,
        // each stated once in its own module and spread in here so templates
        // keep calling them as ordinary component methods.
        ...AML_PRESENTATION,
        ...AML_VISUALS,

        // READY is eligibility; COMPLETE is "results exist". A run in flight in
        // this session overrides the persisted status, and that rule is written
        // once rather than in every template that shows an analysis state.
        analysisState(ws) {
            if (this.analysisRunning === ws.id) return 'ANALYZING';
            return ws.prediction_status || 'NOT_RUN';
        },
        analysisLabel(ws) {
            const state = this.analysisState(ws);
            return state === 'RUNNING' ? 'Analyzing' : this.statusLabel(state).toLowerCase()
                .replace(/^./, c => c.toUpperCase());
        },
        analysisClass(ws) {
            return this.toneBadgeClass(this.analysisState(ws));
        },

        // -------------------------------------------------------------------
        // Dataset configuration
        // -------------------------------------------------------------------
        async loadConfig() {
            this.configError = null;
            try {
                const res = await fetch(`/api/workspaces/${this.mappingWorkspaceId}/config`);
                if (!res.ok) throw new Error(await res.text());
                this.configView = await res.json();
                this.configDirty = false;
            } catch (e) {
                this.configError = (e && e.message) ? e.message : 'Could not load configuration';
            }
        },
        // Local edit only. Nothing is persisted until saveConfig() runs, so the
        // displayed assessment always describes the saved configuration.
        setConfig(key, value) {
            if (!this.configView) return;
            this.configView.config[key] = value;
            this.configDirty = true;
        },
        async saveConfig() {
            if (!this.configView) return;
            this.configSaving = true;
            this.configError = null;
            try {
                const c = this.configView.config;
                const res = await fetch(`/api/workspaces/${this.mappingWorkspaceId}/config`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        identity_source: c.identity_source,
                        currency_policy: c.currency_policy,
                        geography_policy: c.geography_policy,
                        single_country_code: c.single_country_code || null
                    })
                });
                if (!res.ok) throw new Error(await res.text());
                this.configView = await res.json();
                this.configDirty = false;
                // The saved configuration changes what validation would decide,
                // so a previous result no longer describes this workspace.
                this.validationResult = null;
            } catch (e) {
                this.configError = (e && e.message) ? e.message : 'Could not save configuration';
            } finally {
                this.configSaving = false;
            }
        },
        //: The three declarations the system will not guess.  One descriptor per
        //: dimension, so the configuration panel is one card rendered three
        //: times rather than three cards that can drift apart.
        configDimensions: [
            {
                key: 'identity_source',
                assessment: 'identity',
                title: 'Identity',
                consequence: 'Decides what a node in the graph is.'
            },
            {
                key: 'currency_policy',
                assessment: 'currency',
                title: 'Currency',
                consequence: 'Decides what the amount features mean. Nothing is converted.'
            },
            {
                key: 'geography_policy',
                assessment: 'geography',
                title: 'Geography',
                consequence: 'Decides whether the 151 country features carry anything at all.'
            }
        ],
        //: The saved verdict for a dimension, as returned by src/datasets/policy.py.
        configVerdict(dimension) {
            return this.configView?.assessment?.[dimension.assessment] || null;
        },
        configOption(dimension, value) {
            const opts = this.configView?.options?.[dimension] || [];
            return opts.find(o => o.value === value) || null;
        },
        // Concepts a policy option needs that the current mapping does not
        // supply. Drives the inline "mapping does not provide" hint.
        missingFor(option) {
            if (!option || !this.configView) return [];
            const have = this.configView.mapped_concepts || [];
            return (option.requires || []).filter(r => !have.includes(r));
        },

        // -------------------------------------------------------------------
        // Prediction trust
        // -------------------------------------------------------------------
        async loadTrust(workspaceId) {
            this.trust = null;
            this.trustExpanded = false;
            if (!workspaceId || workspaceId === 'default') return;
            try {
                const res = await fetch(`/api/workspaces/${workspaceId}/trust`);
                if (res.ok) {
                    this.trust = await res.json();
                }
            } catch (e) {}
        },

        // -------------------------------------------------------------------
        // Run Analysis
        // -------------------------------------------------------------------
        async runAnalysis(workspaceId) {
            this.analysisRunning = workspaceId;
            this.analysisError = null;
            this.analysisErrorId = null;
            try {
                const res = await fetch(`/api/workspaces/${workspaceId}/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: '{}'
                });
                if (!res.ok) {
                    const detail = await res.text();
                    throw new Error(detail || `Analysis failed (${res.status})`);
                }
                await res.json();
                // Refresh hub state so prediction_status / has_predictions and
                // the resulting action buttons reflect the completed run.
                await this.refreshWorkspaces();
                // Open the freshly predicted workspace dashboard.
                await this.selectWorkspace(workspaceId);
            } catch (e) {
                this.analysisError = (e && e.message) ? e.message : 'Analysis failed';
                this.analysisErrorId = workspaceId;
                await this.refreshWorkspaces();
            } finally {
                this.analysisRunning = null;
            }
        },
        async refreshWorkspaces() {
            try {
                const wsRes = await fetch('/api/workspaces');
                if (wsRes.ok) {
                    this.workspaces = await wsRes.json();
                }
            } catch (e) {}
        },
        async selectWorkspace(wsId) {
            this.selectedWorkspaceId = wsId;
            this.currentView = 'dashboard';
            this.requestVersion++;
            // Reset dataset-specific state
            this.selectedId = null;
            this.selectedEntity = null;
            this.network = null;
            this.entityClassIndex = null;
            this.openFlag = 0;
            if (this.networkInstance) {
                this.networkInstance.destroy();
                this.networkInstance = null;
            }
            this.searchQuery = '';
            this.entityTablePage = 1;
            this.counterpartySearchQuery = '';
            this.counterpartyTablePage = 1;
            this.error = null;
            this.trust = null;
            this.trustExpanded = false;
            if (wsId && wsId !== 'default') {
                const ws = this.workspaces.find(w => w.id === wsId);
                if (ws && ws.manifest_status !== 'READY') {
                    this.workspaceNotReadyStatus = ws.manifest_status;
                    this.overview = {};
                    return;
                }
                // Compatibility READY is not the same as "predictions exist".
                // Without a completed run there is nothing to display.
                if (ws && ws.prediction_status !== 'COMPLETE') {
                    this.workspaceNotReadyStatus = 'NO_ANALYSIS';
                    this.overview = {};
                    return;
                }
            }
            this.workspaceNotReadyStatus = null;
            // Clear the previous dataset's totals before the new ones arrive.
            // The header renders entity / transaction / suspicious counts beside
            // the active dataset's name, so a stale `overview` during the load
            // labels one dataset's numbers as another's.  The two early returns
            // above already do this; the loading path must too.
            this.overview = {};
            await this.loadTrust(wsId);
            await this.loadOverview();
        },
        async openDatasetHub() {
            this.currentView = 'hub';
            this.workspaceDropdownOpen = false;
            await this.refreshWorkspaces();
        },

        // -------------------------------------------------------------------
        // Schema mapping
        // -------------------------------------------------------------------
        openMappingReview(workspaceId) {
            if (!workspaceId) return;
            this.mappingWorkspaceId = workspaceId;
            this.currentView = 'mapping';
            this.validationResult = null;
            this.configView = null;
            this.mappingData = null;
            this.mappingActiveColumn = null;
            this.loadMappingData();
            this.loadConfig();
        },
        async loadMappingData() {
            try {
                const response = await fetch(`/api/workspaces/${this.mappingWorkspaceId}/mapping`);
                const data = await response.json();
                this.mappingManifest = data.manifest;
                this.mappingProfile = data.profile;
                this.mappingData = data.mapping;
                if (this.mappingData && this.mappingData.mappings && this.mappingData.mappings.length > 0) {
                    this.mappingActiveColumn = this.mappingData.mappings[0];
                }
            } catch (e) {
                console.error("Failed to load mapping", e);
            }
        },
        selectMappingColumn(mapping) {
            this.mappingActiveColumn = mapping;
        },
        get mappedColumns() {
            return (this.mappingData?.mappings || []).filter(m => m.status !== 'UNMAPPED');
        },
        get unmappedColumns() {
            return (this.mappingData?.mappings || []).filter(m => m.status === 'UNMAPPED');
        },
        get mappingCounts() {
            const all = this.mappingData?.mappings || [];
            const count = (s) => all.filter(m => m.status === s).length;
            return {
                confirmed: count('AUTO_ACCEPTED'),
                review: count('NEEDS_CONFIRMATION'),
                conflict: count('CONFLICT'),
                unmapped: count('UNMAPPED')
            };
        },
        async confirmMapping(sourceColumn, concept, status) {
            try {
                const response = await fetch(`/api/workspaces/${this.mappingWorkspaceId}/mapping`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        source_column: sourceColumn,
                        selected_concept: concept,
                        status: status
                    })
                });
                if (response.ok) {
                    const data = await response.json();
                    this.mappingData = data.mapping;
                    // update active column reference
                    if (this.mappingActiveColumn) {
                        this.mappingActiveColumn = this.mappingData.mappings.find(m => m.source_column === this.mappingActiveColumn.source_column);
                    }
                    // Which concepts are mapped is exactly what the
                    // configuration is assessed against, so its verdict is
                    // now stale.
                    await this.loadConfig();
                }
            } catch (e) {
                console.error("Failed to confirm mapping", e);
            }
        },
        async validateDataset() {
            this.validating = true;
            try {
                const response = await fetch(`/api/workspaces/${this.mappingWorkspaceId}/validate`, {
                    method: 'POST'
                });
                const data = await response.json();
                this.validationResult = data;
            } catch (e) {
                console.error("Failed to validate", e);
            } finally {
                this.validating = false;
            }
        },
        //: The compatibility report's field states, in the order an analyst
        //: reads them: what is there, what can be derived, what needs a
        //: decision, what is absent and required, what is absent and not.
        fieldGroups: [
            {state: 'AVAILABLE',    label: 'Detected',     tone: 'READY'},
            {state: 'DERIVABLE',    label: 'Derivable',    tone: 'ANALYZING'},
            {state: 'NEEDS_REVIEW', label: 'Needs review', tone: 'NEEDS_REVIEW'},
            {state: 'MISSING',      label: 'Missing',      tone: 'INCOMPATIBLE'},
            {state: 'NOT_SUPPLIED', label: 'Not supplied', tone: null}
        ],
        fieldsIn(state) {
            return (this.validationResult?.compatibility?.field_statuses || []).filter(f => f.state === state);
        },

        // -------------------------------------------------------------------
        // Upload
        // -------------------------------------------------------------------
        //: The ingestion workflow, as the operator experiences it. Purely a
        //: display of where the upload has got to; it drives nothing.
        uploadSteps: [
            {key: 'select',   label: 'Select'},
            {key: 'inspect',  label: 'Inspect'},
            {key: 'map',      label: 'Map'},
            {key: 'validate', label: 'Validate'},
            {key: 'analyze',  label: 'Analyze'}
        ],
        get uploadStepIndex() {
            if (this.uploadStatus === 'SELECTING') return 0;
            if (this.uploadStatus === 'UPLOADING' || this.uploadStatus === 'PROFILING') return 1;
            if (this.uploadStatus === 'FAILED') return 1;
            if (this.uploadStatus === 'NEEDS_MAPPING') return 2;
            if (this.uploadStatus === 'READY') return 3;
            return 2;
        },
        openNewDataset() {
            this.currentView = 'new_dataset';
            this.newDatasetName = '';
            this.resetUpload();
        },
        resetUpload() {
            this.uploadStatus = 'SELECTING';
            this.uploadedDatasetSummary = null;
            this.uploadProgressMsg = '';
            this.uploadError = null;
            this.dragActive = false;
        },
        handleDroppedFile(event) {
            const file = event?.dataTransfer?.files?.[0];
            if (file) this.ingestFile(file);
        },
        async handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            await this.ingestFile(file);
        },
        async ingestFile(file) {
            this.uploadError = null;
            const ext = file.name.split('.').pop().toLowerCase();
            // Convenience only. The server imports parsers.ALLOWED_EXTENSIONS
            // and is authoritative; the browser cannot import it.
            if (!['csv', 'xlsx', 'xls'].includes(ext)) {
                this.uploadError = `Unsupported file type ".${ext}". Upload a CSV, XLSX or XLS file.`;
                return;
            }
            this.uploadStatus = 'UPLOADING';
            this.uploadProgressMsg = 'Uploading ' + file.name;
            try {
                const url = `/api/workspaces/upload?name=${encodeURIComponent(this.newDatasetName || 'Untitled')}&filename=${encodeURIComponent(file.name)}`;
                const res = await fetch(url, {
                    method: 'POST',
                    body: file
                });
                if (!res.ok) {
                    const errText = await res.text();
                    throw new Error(errText);
                }
                const data = await res.json();
                this.uploadStatus = 'PROFILING';
                this.uploadProgressMsg = 'Profiling columns and inferring a mapping';
                // Refresh workspaces to get the summary
                const wsRes = await fetch('/api/workspaces');
                this.workspaces = await wsRes.json();
                this.uploadedDatasetSummary = this.workspaces.find(ws => ws.id === data.workspace_id);
                if (this.uploadedDatasetSummary) {
                    this.mappingWorkspaceId = this.uploadedDatasetSummary.id;
                    this.uploadStatus = this.uploadedDatasetSummary.manifest_status;
                } else {
                    this.uploadStatus = 'FAILED';
                    this.uploadError = 'The workspace was created but could not be read back.';
                }
            } catch (err) {
                this.uploadStatus = 'FAILED';
                this.uploadError = (err && err.message) ? err.message : 'Upload failed';
                this.uploadProgressMsg = '';
            }
        },

        // -------------------------------------------------------------------
        // Dashboard state
        // -------------------------------------------------------------------
        loading: true,
        error: null,
        theme: null,
        overview: {},
        selectedId: null,
        selectedEntity: null,
        network: null,
        networkInstance: null,
        //: Totals for the built-in reference dataset, captured the first time
        //: it loads so the hub can state them without hardcoding a number.
        referenceStats: {entities: null, transactions: null, suspicious: null},

        get activeWorkspaceName() {
            if (!this.selectedWorkspaceId || this.selectedWorkspaceId === 'default') return 'Reference Dataset';
            return this.workspaces.find(ws => ws.id === this.selectedWorkspaceId)?.name || 'Dataset';
        },
        get flaggedShareLabel() {
            const total = this.overview.total_entities;
            const flagged = this.overview.suspicious_count;
            if (!total || flagged === null || flagged === undefined) return 'Model output, not a finding';
            return (flagged / total * 100).toFixed(1) + '% of entities';
        },
        //: Why the projection is empty.  "Tighten your filters" is wrong advice
        //: when the filters are already at their widest and the entity simply
        //: has no direct counterparty edges in the graph.
        get graphEmptyReason() {
            const narrowed = Number(this.vizFilters.minTx) > 1 || this.vizFilters.reciprocalOnly;
            return narrowed
                ? 'Lower the minimum transaction count, or turn off "reciprocal only", to widen the projection.'
                : 'This entity has no direct counterparty in the observed transaction graph.';
        },

        // Table 1 (Entity Ranking) State
        searchQuery: '',
        entityTablePage: 1,
        entityTablePageSize: 10,
        get filteredEntities() {
            if (!this.overview.all_entities) return [];
            if (!this.searchQuery) return this.overview.all_entities;
            const q = this.searchQuery.toLowerCase();
            return this.overview.all_entities.filter(e => String(e.entity_id).toLowerCase().includes(q));
        },
        get paginatedEntities() {
            const start = (this.entityTablePage - 1) * this.entityTablePageSize;
            return this.filteredEntities.slice(start, start + this.entityTablePageSize);
        },
        get totalEntityPages() {
            if (!this.overview.all_entities) return 0;
            return Math.ceil(this.filteredEntities.length / this.entityTablePageSize);
        },

        // Table 2 (Counterparty Relationships) State
        counterpartySearchQuery: '',
        counterpartyTablePage: 1,
        counterpartyTablePageSize: 10,
        get filteredRelationships() {
            if (!this.network || !this.network.relationships) return [];
            if (!this.counterpartySearchQuery) return this.network.relationships;
            const q = this.counterpartySearchQuery.toLowerCase();
            return this.network.relationships.filter(r => String(r.counterparty).toLowerCase().includes(q));
        },
        get paginatedRelationships() {
            const start = (this.counterpartyTablePage - 1) * this.counterpartyTablePageSize;
            return this.filteredRelationships.slice(start, start + this.counterpartyTablePageSize);
        },
        get totalCounterpartyPages() {
            if (!this.network || !this.network.relationships) return 0;
            return Math.ceil(this.filteredRelationships.length / this.counterpartyTablePageSize);
        },

        // Help System
        helpData: {},
        helpModalOpen: false,
        helpModalContent: null,
        openHelp(key) {
            if (this.helpData[key]) {
                this.helpModalContent = this.helpData[key];
                this.helpModalOpen = true;
            } else {
                console.warn("Help key not found:", key);
            }
        },

        vizFilters: {
            maxNeighbors: 15,
            minTx: 1,
            reciprocalOnly: false
        },
        workspaces: [],
        selectedWorkspaceId: null,

        async loadOverview() {
            this.loading = true;
            this.error = null;
            const currentReq = this.requestVersion;
            try {
                let url = '/api/overview';
                if (this.selectedWorkspaceId && this.selectedWorkspaceId !== 'default') {
                    url += '?workspace_id=' + encodeURIComponent(this.selectedWorkspaceId);
                }
                const res = await fetch(url);
                if (!res.ok) throw new Error("Failed to load overview data.");
                const data = await res.json();
                if (this.requestVersion !== currentReq) return;
                this.overview = data;
                this.indexEntityClasses();
                if (this.selectedWorkspaceId === 'default') {
                    this.referenceStats = {
                        entities: data.total_entities,
                        transactions: data.total_transactions,
                        suspicious: data.suspicious_count
                    };
                }
                if (this.overview.top_entities && this.overview.top_entities.length > 0) {
                    this.selectEntity(this.overview.top_entities[0].entity_id);
                } else {
                    this.selectedId = null;
                    this.selectedEntity = null;
                    this.network = null;
                }
            } catch (err) {
                if (this.requestVersion !== currentReq) return;
                this.error = err.message;
            } finally {
                if (this.requestVersion === currentReq) {
                    this.loading = false;
                    this.$nextTick(() => this.drawRiskSpectrum());
                }
            }
        },
        //: Presentation index over predictions the adapter already returned.
        //: Nothing is recomputed: predicted_class is read, never derived.
        indexEntityClasses() {
            const rows = this.overview.all_entities;
            if (!rows) { this.entityClassIndex = null; return; }
            const index = new Map();
            for (const row of rows) index.set(row.entity_id, row.predicted_class);
            this.entityClassIndex = index;
        },

        async init() {
            try {
                // Load Theme
                const themeRes = await fetch('/api/theme');
                this.theme = await themeRes.json();
                // Fetch help content
                const helpResponse = await fetch('/api/help');
                if (helpResponse.ok) {
                    this.helpData = await helpResponse.json();
                }
                // Load workspaces
                try {
                    const wsRes = await fetch('/api/workspaces');
                    if (wsRes.ok) {
                        this.workspaces = await wsRes.json();
                    } else {
                        console.error("Failed to fetch workspaces", await wsRes.text());
                    }
                } catch(e) {
                    console.error("Network error fetching workspaces", e);
                }
                // Two surfaces do not reflow on their own.  The distribution is
                // drawn to a canvas, which has no intrinsic responsive
                // behaviour; and Vis-Network redraws on resize but keeps its
                // previous viewport, which leaves the network off-centre in the
                // new box.
                let resizeTimer = null;
                window.addEventListener('resize', () => {
                    clearTimeout(resizeTimer);
                    resizeTimer = setTimeout(() => {
                        this.drawRiskSpectrum();
                        if (this.networkInstance && !this.graphStabilizing) {
                            this.networkInstance.fit();
                        }
                    }, 160);
                });
                // Load Data
                this.selectedWorkspaceId = 'default';
                await this.loadOverview();
            } catch (err) {
                this.error = err.message;
                this.loading = false;
            }
        },

        async selectEntity(id) {
            this.selectedId = id;
            this.counterpartySearchQuery = '';
            this.counterpartyTablePage = 1;
            this.openFlag = 0;
            const currentReq = this.requestVersion;
            try {
                let url = '/api/entity/' + encodeURIComponent(id);
                if (this.selectedWorkspaceId && this.selectedWorkspaceId !== 'default') {
                    url += '?workspace_id=' + encodeURIComponent(this.selectedWorkspaceId);
                }
                const res = await fetch(url);
                if (!res.ok) throw new Error("Failed to load entity data.");
                const data = await res.json();
                if (this.requestVersion !== currentReq) return;
                this.selectedEntity = data;
                await this.refreshNetwork();
            } catch(err) {
                if (this.requestVersion !== currentReq) return;
                this.error = err.message;
            }
        },
        //: Open an entity as an investigation: select it, then bring the
        //: dossier into view. Selection from the graph deliberately does not
        //: scroll, so exploring the network does not move the page underneath.
        investigate(id) {
            if (!id) return;
            this.selectEntity(id);
            this.scrollToDossier();
        },
        scrollToDossier() {
            this.$nextTick(() => {
                const el = this.$refs.dossier;
                if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
            });
        },
        toggleFlag(index) {
            this.openFlag = (this.openFlag === index) ? -1 : index;
        },

        async refreshNetwork() {
            const currentReq = this.requestVersion;
            try {
                const params = new URLSearchParams({
                    max_neighbors: this.vizFilters.maxNeighbors,
                    min_tx: this.vizFilters.minTx,
                    reciprocal: this.vizFilters.reciprocalOnly
                });
                if (this.selectedWorkspaceId && this.selectedWorkspaceId !== 'default') {
                    params.append('workspace_id', this.selectedWorkspaceId);
                }
                const netRes = await fetch(`/api/network/${encodeURIComponent(this.selectedId)}?${params.toString()}`);
                if (!netRes.ok) throw new Error("Failed to load network data.");
                const data = await netRes.json();
                if (this.requestVersion !== currentReq) return;
                this.network = data;
                this.$nextTick(() => {
                    if (this.requestVersion === currentReq) {
                        this.renderGraph();
                    }
                });
            } catch(err) {
                if (this.requestVersion !== currentReq) return;
                this.error = err.message;
            }
        }
    }))
});
