/*
 * Presentation vocabulary
 * =======================
 * How the console *paints* a value, kept apart from what the console does.
 *
 * Every status this UI renders — workspace status, compatibility, readiness,
 * analysis state, prediction confidence, a mapping decision, a profiler
 * confidence band — collapses to one of five tones. Stating that once means a
 * status cannot be painted green on the hub and amber on the dashboard, and it
 * removes the six near-identical lookup tables that said the same thing in
 * six places.
 *
 * These members are spread into the Alpine component in `app.js`, so templates
 * call them exactly as before. Nothing here reads or writes analysis state:
 * it maps a value to a class string, and never decides which value applies.
 */

//: The five tones. `neutral` is the default and carries no claim: an unknown
//: status is drawn quiet rather than guessed at.
const AML_TONES = ['success', 'warning', 'critical', 'accent', 'neutral'];

//: status -> tone.  Looked up case-insensitively, so the profiler's `High`
//: band and the trust ladder's `LOW` level resolve through the same table.
//:
//: Red is spent only on risk and failure, which is why `LOW` confidence and
//: `INCOMPATIBLE` share a tone while `UNVERIFIED` and `NOT_RUN` do not appear
//: here at all — they fall through to neutral, because the confidence ladder
//: has no affirmative state to draw.
const AML_STATUS_TONE = {
    // Compatibility and workspace status
    READY: 'success',
    NATIVE: 'success',
    FULL: 'success',
    NEEDS_MAPPING: 'warning',
    PARTIAL: 'warning',
    NEEDS_REVIEW: 'warning',
    DEGRADED: 'warning',
    INCOMPATIBLE: 'critical',
    BLOCKED: 'critical',
    FAILED: 'critical',
    UPLOADED: 'accent',
    ANALYZING: 'accent',
    // Analysis state
    COMPLETE: 'success',
    RUNNING: 'accent',
    NO_ANALYSIS: 'accent',
    // Prediction confidence
    LIMITED: 'warning',
    LOW: 'critical',
    // Mapping decisions
    AUTO_ACCEPTED: 'success',
    NEEDS_CONFIRMATION: 'warning',
    CONFLICT: 'critical',
    // Profiler confidence bands
    HIGH: 'success',
    MEDIUM: 'warning'
};

//: Badge and pill: foreground plus a matching hairline.
const AML_TONE_BADGE = {
    success: 'text-system-success border-system-success/40',
    warning: 'text-system-warning border-system-warning/40',
    critical: 'text-system-critical border-system-critical/45',
    accent: 'text-system-accent border-system-accent/40',
    neutral: 'text-system-muted border-system-border'
};

//: Foreground only, for a value rendered as text rather than as a pill.
const AML_TONE_TEXT = {
    success: 'text-system-success',
    warning: 'text-system-warning',
    critical: 'text-system-critical',
    accent: 'text-system-accent',
    neutral: 'text-system-muted'
};

//: Edge only, for a panel that carries the state on its own border.  Neutral
//: contributes nothing, so an ordinary panel keeps the default border.
const AML_TONE_EDGE = {
    success: 'border-system-success/45',
    warning: 'border-system-warning/45',
    critical: 'border-system-critical/45',
    accent: 'border-system-accent/45',
    neutral: ''
};

const AML_PRESENTATION = {
    // -----------------------------------------------------------------------
    // Values
    // -----------------------------------------------------------------------
    //: Presentation only — no rounding that changes a value's meaning, and no
    //: unit invented for a number that has none.  An absent number is an
    //: em dash, never a zero.
    fmtInt(value) {
        if (value === null || value === undefined || Number.isNaN(value)) return '—';
        return Number(value).toLocaleString();
    },
    fmtUsd(value) {
        if (value === null || value === undefined || Number.isNaN(value)) return '—';
        return '$' + Number(value).toLocaleString(undefined, {
            minimumFractionDigits: 2, maximumFractionDigits: 2
        });
    },

    // -----------------------------------------------------------------------
    // Tones
    // -----------------------------------------------------------------------
    tone(status) {
        if (!status) return 'neutral';
        return AML_STATUS_TONE[String(status).toUpperCase()] || 'neutral';
    },
    toneBadgeClass(status) {
        return AML_TONE_BADGE[this.tone(status)];
    },
    toneTextClass(status) {
        return AML_TONE_TEXT[this.tone(status)];
    },
    toneEdgeClass(status) {
        return AML_TONE_EDGE[this.tone(status)];
    },
    //: A bordered icon well that carries the state in both border and glyph.
    toneOutlineClass(status) {
        const tone = this.tone(status);
        return `${AML_TONE_EDGE[tone]} ${AML_TONE_TEXT[tone]}`.trim();
    },

    // -----------------------------------------------------------------------
    // Labels
    // -----------------------------------------------------------------------
    //: A status reads better as a word than as a SCREAMING_SNAKE constant, and
    //: an unknown one is passed through rather than mangled.
    statusLabel(status) {
        return String(status || '').replace(/_/g, ' ');
    },
    mappingStatusLabel(status) {
        return {
            AUTO_ACCEPTED: '✓ Confirmed',
            NEEDS_CONFIRMATION: 'Review',
            CONFLICT: 'Conflict',
            UNMAPPED: 'Unmapped'
        }[status] || this.statusLabel(status);
    },
    //: The profiler's own score, banded for reading.  The bands are display
    //: only; the score itself is shown on each candidate.
    confidenceLabel(mapping) {
        const top = mapping && mapping.candidates && mapping.candidates[0];
        if (!top) return '—';
        if (top.score >= 0.8) return 'High';
        if (top.score >= 0.5) return 'Medium';
        return 'Low';
    }
};
