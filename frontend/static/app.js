/* ================================================================
   Detective AI — Frontend App v10
   Single-page dark-mode application
   ================================================================ */

const API = window.location.origin;
let currentInvCaseId = null;   // tracks the currently running investigation

/* ── Tab Navigation ──────────────────────────────────────────── */

function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => {
        const active = t.dataset.tab === tabName;
        t.classList.toggle('active', active);
        t.setAttribute('aria-selected', active);
    });
    document.querySelectorAll('.page').forEach(p => {
        p.classList.toggle('active', p.id === `page-${tabName}`);
    });
    if (tabName === 'dashboard') refreshDashboard();
    if (tabName === 'reports')   refreshReports();
}

document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

/* ── Sub-tab (within Ingest) ─────────────────────────────────── */

function switchSubTab(name, btn) {
    document.querySelectorAll('.sub-page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.sub-tab').forEach(b => b.classList.remove('active'));
    const page = document.getElementById(`sub-${name}`);
    if (page) page.classList.add('active');
    if (btn)  btn.classList.add('active');
}

/* ── API Helpers ─────────────────────────────────────────────── */

async function apiGet(path) {
    try {
        const r = await fetch(`${API}${path}`);
        return await r.json();
    } catch (e) {
        console.error('GET', path, e);
        return null;
    }
}

async function apiPost(path, data) {
    try {
        const r = await fetch(`${API}${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return await r.json();
    } catch (e) {
        console.error('POST', path, e);
        return null;
    }
}

/* ── Toast ───────────────────────────────────────────────────── */

function toast(msg, type = 'info') {
    const c = document.getElementById('toastContainer');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    c.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transform = 'translateX(16px)';
        el.style.transition = '250ms ease';
        setTimeout(() => el.remove(), 260);
    }, 3800);
}

/* ── Escape ──────────────────────────────────────────────────── */

function esc(s) {
    const d = document.createElement('div');
    d.textContent = String(s || '');
    return d.innerHTML;
}

/* ── Health Check ────────────────────────────────────────────── */

async function checkHealth() {
    const dot  = document.getElementById('apiStatus');
    const lbl  = document.getElementById('apiStatusText');
    try {
        const d = await apiGet('/api/health');
        if (d && d.status === 'healthy') {
            dot.className = 'status-dot connected';
            lbl.textContent = 'API Connected';
        } else {
            dot.className = 'status-dot disconnected';
            lbl.textContent = 'API Error';
        }
    } catch {
        dot.className = 'status-dot disconnected';
        lbl.textContent = 'API Offline';
    }
}

/* ── Dashboard ───────────────────────────────────────────────── */

async function refreshDashboard() {
    const d = await apiGet('/api/v1/cases');
    if (!d || !d.cases) return;
    const cases = d.cases;

    document.getElementById('totalCases').textContent     = cases.length;
    document.getElementById('activeCases').textContent    = cases.filter(c => c.status === 'running').length;
    document.getElementById('completedCases').textContent = cases.filter(c => c.status === 'completed').length;
    document.getElementById('failedCases').textContent    = cases.filter(c => c.status === 'failed').length;

    const list = document.getElementById('casesList');
    if (!cases.length) {
        list.innerHTML = `<div class="empty-state">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.3-4.3"></path></svg>
            <p>No investigations yet</p>
            <span>Ingest evidence then start your first investigation</span>
        </div>`;
        return;
    }
    list.innerHTML = cases.map(c => {
        const badgeClass = c.status === 'completed'         ? 'badge-completed'
                         : c.status === 'running'           ? 'badge-running'
                         : c.status === 'failed'            ? 'badge-failed'
                         : c.status === 'evidence_ingested' ? 'badge-ingested'
                         : 'badge-pending';
        const badgeLabel = c.status === 'evidence_ingested' ? 'Evidence Ready'
                         : c.status.replace(/_/g, ' ');
        const date = c.created_at ? new Date(c.created_at).toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric'}) : 'N/A';
        const canInvestigate = ['evidence_ingested','pending','failed','completed'].includes(c.status);
        const canReport = c.status === 'completed';
        return `<div class="case-item">
            <div style="flex:1;min-width:0" onclick="openCaseReport('${esc(c.id)}')" style="cursor:pointer">
                <div class="case-item-title">${esc(c.title)}</div>
                <div class="case-item-meta">${c.id.substring(0,12)}… &middot; ${date}</div>
            </div>
            <div style="display:flex;gap:0.5rem;align-items:center;flex-shrink:0">
                ${canInvestigate ? `<button class="btn-ghost btn-sm" style="font-size:0.72rem;padding:0.25rem 0.6rem" onclick="investigateCase('${esc(c.id)}','${esc(c.title)}')">Investigate</button>` : ''}
                ${canReport ? `<button class="btn-ghost btn-sm" style="font-size:0.72rem;padding:0.25rem 0.6rem" onclick="openCaseReport('${esc(c.id)}')">Report</button>` : ''}
                <span class="badge ${badgeClass}">${badgeLabel}</span>
            </div>
        </div>`;
    }).join('');
}

async function clearHistory() {
    if (!confirm('Are you sure you want to clear ALL data? This will delete all cases, evidence, statements, and reports. This action cannot be undone.')) return;
    toast('Clearing all data…', 'info');
    try {
        const r = await fetch(`${API}/api/v1/clear`, { method: 'DELETE' });
        const result = await r.json();
        if (result && result.status === 'success') {
            toast('✓ All data cleared successfully', 'success');
            refreshDashboard();
        } else {
            toast(result?.message || 'Failed to clear data', 'error');
        }
    } catch (e) {
        toast('Failed to clear data — is the server running?', 'error');
    }
}

function openCaseReport(caseId) {
    switchTab('reports');
    setTimeout(() => loadReport(caseId), 100);
}

function investigateCase(caseId, title) {
    // Switch to investigate tab and pre-fill with case details
    switchTab('investigate');
    const caseIdField = document.getElementById('invCaseId');
    const titleField  = document.getElementById('invTitle');
    if (caseIdField) caseIdField.value = caseId;
    if (titleField && title) titleField.value = title;
    // Scroll to the launch button
    setTimeout(() => {
        const btn = document.getElementById('launchBtn');
        if (btn) btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 200);
    toast(`Case ${caseId.substring(0,8)}… loaded — set a title and click Launch`, 'info');
}

/* ── Ingest: Video ───────────────────────────────────────────── */

async function ingestVideo() {
    const caseId = document.getElementById('videoCaseId').value.trim();
    const file = document.getElementById('videoFile').files[0];

    if (!file) { toast('Please select a video file', 'error'); return; }

    const btn = document.getElementById('videoUploadBtn');
    btn.disabled = true;
    btn.textContent = 'Uploading…';

    const prog = document.getElementById('uploadProgress');
    const fill = document.getElementById('uploadFill');
    const lbl  = document.getElementById('uploadLabel');
    prog.style.display = 'flex';

    // Simulate progress during upload
    let pct = 0;
    const ticker = setInterval(() => {
        pct = Math.min(pct + 4, 85);
        fill.style.width = pct + '%';
        lbl.textContent = `Uploading… ${pct}%`;
    }, 200);

    const fd = new FormData();
    fd.append('file', file);
    if (caseId) fd.append('case_id', caseId);

    try {
        const r = await fetch(`${API}/api/v1/ingest/video`, { method:'POST', body:fd });
        const result = await r.json();
        clearInterval(ticker);
        fill.style.width = '100%';
        lbl.textContent = 'Processing complete';

        if (result && result.status === 'success') {
            const assignedCaseId = result.data?.case_id || '';
            toast(`✓ ${result.message}`, 'success');
            const displayData = {
                ...result,
                data: {
                    ...result.data,
                    note: assignedCaseId ? `Linked to Case ID: ${assignedCaseId}` : ''
                }
            };
            showResult('videoResult', displayData, 'success');
            // Auto-fill the case ID field for reference
            if (assignedCaseId) {
                document.getElementById('videoCaseId').value = assignedCaseId;
            }
            // Refresh dashboard so the new case appears
            refreshDashboard();
        } else {
            toast(result?.message || 'Video ingestion failed', 'error');
            showResult('videoResult', result, 'error');
        }
    } catch (e) {
        clearInterval(ticker);
        toast('Upload failed — is the server running?', 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg> Upload &amp; Process Video`;
    }
}

/* ── Ingest: Logs ────────────────────────────────────────────── */

async function ingestLogs() {
    const content = document.getElementById('logContent').value.trim();
    const source  = document.getElementById('logSource').value.trim() || 'badge_system';
    const caseId  = document.getElementById('logCaseId').value.trim();

    if (!content) { toast('Please enter log data', 'error'); return; }
    try { JSON.parse(content); } catch { toast('Invalid JSON — please check your log data format', 'error'); return; }

    toast('Processing access logs…', 'info');
    const result = await apiPost('/api/v1/ingest/logs', { content, format:'json', source, case_id: caseId || undefined });

    if (result && result.status === 'success') {
        toast(`✓ ${result.message}`, 'success');
        showResult('logsResult', result, 'success');
    } else {
        toast(result?.message || 'Ingestion failed', 'error');
        showResult('logsResult', result, 'error');
    }
}

/* ── Ingest: Statement ───────────────────────────────────────── */

async function ingestStatement() {
    const text        = document.getElementById('stmtText').value.trim();
    const source      = document.getElementById('stmtSource').value.trim();
    const timestamp   = document.getElementById('stmtTimestamp').value;
    const eventTime   = document.getElementById('stmtEventTime').value;
    const caseId      = document.getElementById('stmtCaseId').value.trim();

    if (!text)   { toast('Please enter the statement text', 'error'); return; }
    if (!source) { toast('Please enter the witness name', 'error'); return; }

    toast('Processing witness statement…', 'info');
    const result = await apiPost('/api/v1/ingest/statements', {
        text, source,
        timestamp:        timestamp  || new Date().toISOString(),
        event_time:       eventTime  || null,
        case_id:          caseId    || undefined,
    });

    if (result && result.status === 'success') {
        toast(`✓ ${result.message}`, 'success');
        showResult('stmtResult', result, 'success');
    } else {
        toast(result?.message || 'Ingestion failed', 'error');
        showResult('stmtResult', result, 'error');
    }
}

/* ── Shared Result Renderer ──────────────────────────────────── */

function showResult(id, result, type) {
    const el = document.getElementById(id);
    el.style.display = 'block';
    el.className = `result-card ${type}`;
    const title = type === 'success' ? (result?.message || 'Success') : (result?.message || 'Error');
    const data  = result?.data ? JSON.stringify(result.data, null, 2) : '';
    el.innerHTML = `
        <div class="result-title">${type === 'success' ? '✓ ' : '✗ '}${esc(title)}</div>
        ${data ? `<pre class="result-pre">${esc(data)}</pre>` : ''}
    `;
}

/* ── Investigation ───────────────────────────────────────────── */

async function startInvestigation() {
    const title       = document.getElementById('invTitle').value.trim();
    const description = document.getElementById('invDescription').value.trim();
    const maxRounds   = parseInt(document.getElementById('invMaxRounds').value) || 3;
    const caseId      = document.getElementById('invCaseId').value.trim();

    if (!title) { toast('Please enter an investigation title', 'error'); return; }

    const btn = document.getElementById('launchBtn');
    btn.disabled = true;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg> Launching…`;

    toast('Launching investigation pipeline…', 'info');
    const result = await apiPost('/api/v1/investigate/', {
        case_id: caseId || undefined, title, description, max_rounds: maxRounds,
    });

    if (result && result.status === 'success') {
        const id = result.data.case_id;
        currentInvCaseId = id;
        toast(`Investigation started — ${id.substring(0,8)}…`, 'success');
        showPipeline(id);
        pollStatus(id);
    } else {
        toast(result?.message || 'Failed to start investigation', 'error');
        btn.disabled = false;
        btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Launch Investigation`;
    }
}

function showPipeline(caseId) {
    const card = document.getElementById('pipelineCard');
    card.style.display = 'block';
    document.getElementById('pipelineCaseId').textContent = `Case: ${caseId}`;
    document.getElementById('pipelineLog').textContent = `> Investigation ${caseId.substring(0,8)}… launched\n`;
    document.getElementById('viewReportAction').style.display = 'none';
    // Reset all steps
    ['investigation','trajectory','critic','verifier','reporter','completed'].forEach(s => {
        const el = document.getElementById(`ps-${s}`);
        if (el) el.className = 'pipeline-step';
    });
    // Mark first step active
    setPipelineStep('investigation');
}

const PHASE_MAP = {
    'running':        'investigation',
    'investigation':  'investigation',
    'investigator':   'investigation',
    'trajectory':     'trajectory',
    'critic':         'critic',
    'verifier':       'verifier',
    'reporter':       'reporter',
    'completed':      'completed',
};

const STEP_ORDER = ['investigation','trajectory','critic','verifier','reporter','completed'];

function setPipelineStep(current) {
    const idx = STEP_ORDER.indexOf(current);
    STEP_ORDER.forEach((s, i) => {
        const el = document.getElementById(`ps-${s}`);
        if (!el) return;
        el.className = 'pipeline-step' + (i < idx ? ' done' : i === idx ? ' active' : '');
    });
}

async function pollStatus(caseId) {
    const log = document.getElementById('pipelineLog');
    let prevPhase = '';
    let pollCount = 0;
    const MAX_POLLS = 200; // safety

    const timer = setInterval(async () => {
        pollCount++;
        if (pollCount > MAX_POLLS) { clearInterval(timer); return; }

        const d = await apiGet(`/api/v1/investigate/${caseId}`);
        if (!d || !d.data) return;
        const { status, phase, current_round } = d.data;

        const mappedPhase = PHASE_MAP[phase] || phase;
        if (phase !== prevPhase) {
            log.textContent += `> [Round ${current_round || '?'}] ${phase}\n`;
            log.scrollTop = log.scrollHeight;
            setPipelineStep(mappedPhase);
            prevPhase = phase;
        }

        if (status === 'completed') {
            clearInterval(timer);
            setPipelineStep('completed');
            log.textContent += `> ✓ Investigation completed\n`;
            log.scrollTop = log.scrollHeight;
            toast('Investigation complete — report ready!', 'success');
            document.getElementById('viewReportAction').style.display = 'flex';
            document.getElementById('launchBtn').disabled = false;
            document.getElementById('launchBtn').innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Launch Investigation`;
        } else if (status === 'failed') {
            clearInterval(timer);
            log.textContent += `> ✗ Investigation failed\n`;
            log.scrollTop = log.scrollHeight;
            toast('Investigation failed. Check server logs.', 'error');
            document.getElementById('launchBtn').disabled = false;
            document.getElementById('launchBtn').innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Launch Investigation`;
        }
    }, 3000);
}

function goToReport() {
    if (!currentInvCaseId) return;
    switchTab('reports');
    setTimeout(() => loadReport(currentInvCaseId), 150);
}

/* ── Reports ─────────────────────────────────────────────────── */

async function refreshReports() {
    const d = await apiGet('/api/v1/cases');
    if (!d || !d.cases) return;
    const completed = d.cases.filter(c => c.status === 'completed');
    const list = document.getElementById('reportsCaseList');

    if (!completed.length) {
        list.innerHTML = `<div class="empty-state">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
            <p>No completed reports</p><span>Finish an investigation first</span>
        </div>`;
        return;
    }

    list.innerHTML = completed.map(c => `
        <div class="report-case-item" id="rci-${c.id}" onclick="loadReport('${esc(c.id)}')">
            <div>
                <div class="report-case-item-title">${esc(c.title)}</div>
                <div class="report-case-item-id">${c.id.substring(0,14)}…</div>
            </div>
        </div>
    `).join('');
}

async function loadReport(caseId) {
    // Highlight selected case
    document.querySelectorAll('.report-case-item').forEach(el => {
        el.classList.toggle('selected', el.id === `rci-${caseId}`);
    });

    document.getElementById('reportEmpty').style.display = 'none';
    const viewer = document.getElementById('reportContent');
    viewer.style.display = 'block';
    viewer.innerHTML = `<div class="card"><div class="empty-state"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg><p>Loading report…</p></div></div>`;

    const d = await apiGet(`/api/v1/report/${caseId}`);
    if (!d || d.status !== 'success' || !d.data?.report) {
        viewer.innerHTML = `<div class="card"><div class="empty-state"><p style="color:var(--red)">Report not available: ${esc(d?.message || 'Unknown error')}</p></div></div>`;
        return;
    }
    renderReport(d.data.report, caseId);
}

function renderReport(report, caseId) {
    const viewer = document.getElementById('reportContent');
    const confidence = report.confidence_assessment?.overall_confidence
        ?? report.primary_conclusion?.confidence ?? 0;
    const pct = Math.round(confidence * 100);
    const confClass = pct >= 70 ? 'conf-high' : pct >= 40 ? 'conf-med' : 'conf-low';
    const confColor = pct >= 70 ? 'var(--green)' : pct >= 40 ? 'var(--amber)' : 'var(--red)';

    // Build timeline section
    let timelineHtml = '';
    if (report.timeline?.length) {
        timelineHtml = `<div class="report-section">
            <div class="report-section-title"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg> Evidence Timeline</div>
            ${report.timeline.map(item => {
                const isInferred  = item.is_inferred  === true || item.inferred === true;
                const isConfirmed = item.is_confirmed !== false;
                const tag = isInferred
                    ? `<span class="tag tag-inferred">INFERRED</span>`
                    : `<span class="tag tag-observed">OBSERVED</span>`;
                const unconf = !isConfirmed ? `<span class="tag tag-unconfirmed">UNCONFIRMED</span>` : '';
                return `<div class="timeline-item">
                    <div class="timeline-time">${esc(item.time || item.timestamp || '?')}</div>
                    <div class="timeline-event">${esc(item.event || item.description || '')}${tag}${unconf}</div>
                </div>`;
            }).join('')}
        </div>`;
    }

    // Critic findings / unresolved objections
    let objHtml = '';
    const objections = report.unresolved_objections || report.critic_findings || [];
    if (objections.length) {
        objHtml = `<div class="report-section">
            <div class="report-section-title"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg> Critic Findings &amp; Unresolved Objections</div>
            ${objections.map(o => {
                const sev = o.severity ? `<span class="objection-severity">[${esc(o.severity)}]</span>` : '';
                const text = o.objection || o.objection_text || o.finding || '';
                return `<div class="objection-card">${sev}${esc(text)}</div>`;
            }).join('')}
        </div>`;
    }

    // Alternative / rejected hypotheses
    let altHtml = '';
    const alts = report.alternative_hypotheses || report.rejected_hypotheses || [];
    if (alts.length) {
        altHtml = `<div class="report-section">
            <div class="report-section-title"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg> Considered &amp; Rejected Hypotheses</div>
            ${alts.map(h => `
                <div class="hypothesis-card">
                    <div class="hypothesis-title">${esc(h.hypothesis || h.title || 'Hypothesis')}</div>
                    <div class="hypothesis-reason">${esc(h.rejection_reason || h.reason || '')}</div>
                </div>
            `).join('')}
        </div>`;
    }

    // Metadata
    let metaHtml = '';
    if (report.metadata) {
        metaHtml = `<div class="report-section">
            <div class="report-section-title"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"></rect><path d="M3 9h18M9 21V9"></path></svg> Analysis Metadata</div>
            <div class="meta-grid">${Object.entries(report.metadata).map(([k,v]) => `
                <div class="meta-item">
                    <div class="meta-key">${esc(k.replace(/_/g,' '))}</div>
                    <div class="meta-val">${esc(v)}</div>
                </div>
            `).join('')}</div>
        </div>`;
    }

    viewer.innerHTML = `
        <div class="card report-section">
            <div class="report-section-title"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg> ${esc(report.title || 'Investigation Report')}</div>
            <p style="color:var(--text-secondary);line-height:1.7;font-size:0.88rem;margin-bottom:1.25rem">${esc(report.summary || 'No summary available.')}</p>

            <div class="report-section-title"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg> Primary Conclusion</div>
            <p style="font-size:0.9rem;font-weight:600;color:var(--text-primary);margin-bottom:1rem">${esc(report.primary_conclusion?.hypothesis || report.primary_conclusion?.title || 'No conclusion reached.')}</p>

            <div class="confidence-row">
                <div class="confidence-num" style="color:${confColor}">${pct}%</div>
                <div style="flex:1">
                    <div style="font-size:0.73rem;color:var(--text-muted);margin-bottom:0.35rem">Overall Confidence</div>
                    <div class="confidence-track"><div class="confidence-fill ${confClass}" style="width:${pct}%"></div></div>
                </div>
            </div>
        </div>

        ${timelineHtml ? `<div class="card">${timelineHtml}</div>` : ''}
        ${objHtml      ? `<div class="card">${objHtml}</div>`      : ''}
        ${altHtml      ? `<div class="card">${altHtml}</div>`      : ''}
        ${metaHtml     ? `<div class="card">${metaHtml}</div>`     : ''}

        <div style="font-size:0.73rem;color:var(--text-muted);text-align:right;margin-top:0.5rem;font-family:'JetBrains Mono',monospace">
            Case: ${esc(caseId)}
        </div>
    `;
}

/* ── Counterfactual ──────────────────────────────────────────── */

async function runCounterfactual() {
    const caseId     = document.getElementById('cfCaseId').value.trim();
    const evidenceId = document.getElementById('cfEvidenceId').value.trim();

    if (!caseId || !evidenceId) {
        toast('Please enter both a Case ID and Evidence ID', 'error');
        return;
    }

    toast('Running what-if analysis…', 'info');
    const result = await apiPost('/api/v1/counterfactual/', {
        case_id: caseId, removed_evidence_id: evidenceId,
    });

    const panel = document.getElementById('cfResult');
    panel.style.display = 'block';

    if (result && result.status === 'success') {
        const d = result.data;
        const origConf = ((d.original_leading?.confidence || 0) * 100).toFixed(0);
        const cfConf   = ((d.counterfactual_leading?.confidence || 0) * 100).toFixed(0);
        const changed  = d.conclusion_changed;

        panel.innerHTML = `
            <div class="card">
                <div class="report-section-title" style="margin-bottom:1rem">What-If Result</div>
                <div class="cf-grid">
                    <div class="cf-box">
                        <div class="cf-box-label">Original Conclusion</div>
                        <div class="cf-box-title">${esc(d.original_leading?.title || d.original_leading?.hypothesis || 'N/A')}</div>
                        <div class="cf-confidence" style="color:var(--accent)">${origConf}%</div>
                    </div>
                    <div class="cf-box">
                        <div class="cf-box-label">Without Evidence ${esc(evidenceId)}</div>
                        <div class="cf-box-title">${esc(d.counterfactual_leading?.title || d.counterfactual_leading?.hypothesis || 'N/A')}</div>
                        <div class="cf-confidence" style="color:var(--purple)">${cfConf}%</div>
                    </div>
                </div>
                <div class="cf-verdict ${changed ? 'cf-changed' : 'cf-unchanged'}">
                    ${changed
                        ? '⚠️  Conclusion CHANGED — this evidence is critical to the investigation'
                        : '✓  Conclusion unchanged — the investigation is robust to this evidence removal'}
                </div>
            </div>
        `;
    } else {
        panel.innerHTML = `<div class="card"><p style="color:var(--red)">Analysis failed: ${esc(result?.message || 'Unknown error')}</p></div>`;
    }
}

/* ── Drop Zone ───────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('videoDropZone');
    const fileInput = document.getElementById('videoFile');
    const dropText  = document.getElementById('dropText');

    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', e => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                dropText.innerHTML = `<strong>${esc(e.dataTransfer.files[0].name)}</strong> selected`;
            }
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                dropText.innerHTML = `<strong>${esc(fileInput.files[0].name)}</strong> selected`;
            }
        });
    }

    checkHealth();
    refreshDashboard();
    setInterval(checkHealth, 30000);
});
