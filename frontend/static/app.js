/* ═══════════════════════════════════════════════════════════════
   Detective AI — Frontend Application Logic
   ═══════════════════════════════════════════════════════════════ */

const API_BASE = window.location.origin;

// ── View Switching ───────────────────────────────────────────

function switchView(viewName) {
    // Hide all views
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    // Show target view
    const target = document.getElementById(`view-${viewName}`);
    if (target) target.classList.add('active');

    // Update nav
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const navItem = document.querySelector(`.nav-item[data-view="${viewName}"]`);
    if (navItem) navItem.classList.add('active');

    // Refresh data for specific views
    if (viewName === 'dashboard') refreshDashboard();
    if (viewName === 'reports') refreshReports();
}

// ── Ingest Tab Switching ─────────────────────────────────────

function switchIngestTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

    const tab = document.getElementById(`tab-${tabName}`);
    if (tab) tab.classList.add('active');

    event.target.classList.add('active');
}

// ── Toast Notifications ──────────────────────────────────────

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ── API Helpers ──────────────────────────────────────────────

async function apiGet(path) {
    try {
        const res = await fetch(`${API_BASE}${path}`);
        return await res.json();
    } catch (e) {
        console.error(`GET ${path} failed:`, e);
        return null;
    }
}

async function apiPost(path, data) {
    try {
        const res = await fetch(`${API_BASE}${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        return await res.json();
    } catch (e) {
        console.error(`POST ${path} failed:`, e);
        return null;
    }
}

// ── Dashboard ────────────────────────────────────────────────

async function refreshDashboard() {
    const data = await apiGet('/api/v1/cases');
    if (!data || !data.cases) return;

    const cases = data.cases;
    document.getElementById('totalCases').textContent = cases.length;
    document.getElementById('activeCases').textContent = cases.filter(c => c.status === 'running').length;
    document.getElementById('completedCases').textContent = cases.filter(c => c.status === 'completed').length;

    const list = document.getElementById('casesList');
    if (cases.length === 0) {
        list.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.3-4.3"></path></svg>
                <p>No investigations yet</p>
                <span>Ingest evidence and start your first investigation</span>
            </div>`;
        return;
    }

    list.innerHTML = cases.map(c => `
        <div class="case-card" onclick="viewCase('${c.id}')">
            <div class="case-info">
                <h4>${escapeHtml(c.title)}</h4>
                <span class="case-meta">
                    ${c.id.substring(0, 8)}... · Round ${c.current_round} · 
                    ${c.created_at ? new Date(c.created_at).toLocaleDateString() : 'N/A'}
                </span>
            </div>
            <span class="case-status ${c.status}">${c.status}</span>
        </div>
    `).join('');
}

async function viewCase(caseId) {
    const data = await apiGet(`/api/v1/report/${caseId}`);
    if (data && data.data && data.data.report) {
        renderReport(data.data.report, caseId);
        switchView('reports');
    } else {
        showToast('Report not yet available', 'info');
    }
}

// ── Evidence Ingestion ───────────────────────────────────────

async function ingestLogs() {
    const content = document.getElementById('logContent').value.trim();
    const source = document.getElementById('logSource').value.trim();
    const caseId = document.getElementById('logCaseId').value.trim();

    if (!content) {
        showToast('Please enter log data', 'error');
        return;
    }

    showToast('Processing access logs...', 'info');

    const result = await apiPost('/api/v1/ingest/logs', {
        content: content,
        format: 'json',
        source: source || 'badge_system',
        case_id: caseId,
    });

    if (result && result.status === 'success') {
        showToast(result.message, 'success');
        showIngestResult(result);
    } else {
        showToast(result?.message || 'Ingestion failed', 'error');
    }
}

async function ingestStatement() {
    const text = document.getElementById('stmtText').value.trim();
    const source = document.getElementById('stmtSource').value.trim();
    const timestamp = document.getElementById('stmtTimestamp').value;
    const eventTime = document.getElementById('stmtEventTime').value;
    const reliability = parseFloat(document.getElementById('stmtReliability').value);
    const caseId = document.getElementById('stmtCaseId').value.trim();

    if (!text || !source) {
        showToast('Please fill in witness name and statement text', 'error');
        return;
    }

    showToast('Processing witness statement...', 'info');

    const result = await apiPost('/api/v1/ingest/statements', {
        text: text,
        source: source,
        timestamp: timestamp || new Date().toISOString(),
        event_time: eventTime || null,
        reliability_score: reliability || 0.7,
        case_id: caseId,
    });

    if (result && result.status === 'success') {
        showToast(result.message, 'success');
        showIngestResult(result);
    } else {
        showToast(result?.message || 'Ingestion failed', 'error');
    }
}

async function ingestVideo() {
    showToast('Video ingestion requires file upload — use the API directly at /api/docs', 'info');
}

function showIngestResult(result) {
    const panel = document.getElementById('ingestResult');
    panel.style.display = 'block';
    panel.className = `result-panel ${result.status}`;
    panel.innerHTML = `
        <h4 style="margin-bottom: 0.5rem; color: var(--accent-green);">✓ ${result.message}</h4>
        <pre style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--text-secondary);">${JSON.stringify(result.data, null, 2)}</pre>
    `;
}

// ── Investigation ────────────────────────────────────────────

async function startInvestigation() {
    const title = document.getElementById('invTitle').value.trim();
    const description = document.getElementById('invDescription').value.trim();
    const maxRounds = parseInt(document.getElementById('invMaxRounds').value);
    const caseId = document.getElementById('invCaseId').value.trim();

    if (!title) {
        showToast('Please enter an investigation title', 'error');
        return;
    }

    showToast('Launching investigation pipeline...', 'info');

    const result = await apiPost('/api/v1/investigate/', {
        case_id: caseId || undefined,
        title: title,
        description: description,
        max_rounds: maxRounds || 3,
    });

    if (result && result.status === 'success') {
        showToast(`Investigation started: ${result.data.case_id}`, 'success');
        showProgressPanel(result.data.case_id);
        pollInvestigationStatus(result.data.case_id);
    } else {
        showToast(result?.message || 'Failed to start investigation', 'error');
    }
}

function showProgressPanel(caseId) {
    const panel = document.getElementById('investigationProgress');
    panel.style.display = 'block';
    document.getElementById('progressLog').innerHTML = `> Investigation ${caseId.substring(0, 8)}... launched\n`;
    
    // Activate first step
    document.getElementById('step-ingestion').classList.add('active');
}

function updateProgress(phase) {
    const steps = ['ingestion', 'hypothesis', 'gathering', 'trajectory', 'critic', 'verification', 'report'];
    const phaseMap = {
        'investigator': 'gathering',
        'hypothesis_formation': 'hypothesis',
        'evidence_gathering': 'gathering',
        'trajectory': 'trajectory',
        'critic': 'critic',
        'verifier': 'verification',
        'reporter': 'report',
        'completed': 'report',
    };
    
    const currentStep = phaseMap[phase] || phase;
    const currentIndex = steps.indexOf(currentStep);

    steps.forEach((step, i) => {
        const el = document.getElementById(`step-${step}`);
        if (!el) return;
        el.classList.remove('active', 'done');
        if (i < currentIndex) el.classList.add('done');
        else if (i === currentIndex) el.classList.add('active');
    });
}

async function pollInvestigationStatus(caseId) {
    const log = document.getElementById('progressLog');
    let prevPhase = '';

    const poll = setInterval(async () => {
        const data = await apiGet(`/api/v1/investigate/${caseId}`);
        if (!data || !data.data) return;

        const { status, phase } = data.data;

        if (phase !== prevPhase) {
            log.innerHTML += `> Phase: ${phase} (Round ${data.data.current_round || '?'})\n`;
            log.scrollTop = log.scrollHeight;
            updateProgress(phase);
            prevPhase = phase;
        }

        if (status === 'completed') {
            clearInterval(poll);
            log.innerHTML += `> ✓ Investigation completed successfully\n`;
            showToast('Investigation completed! View the report.', 'success');
            updateProgress('completed');
        } else if (status === 'failed') {
            clearInterval(poll);
            log.innerHTML += `> ✗ Investigation failed\n`;
            showToast('Investigation failed. Check logs.', 'error');
        }
    }, 3000);
}

// ── Reports ──────────────────────────────────────────────────

async function refreshReports() {
    const data = await apiGet('/api/v1/cases');
    if (!data || !data.cases) return;

    const completed = data.cases.filter(c => c.status === 'completed');
    const container = document.getElementById('reportContent');

    if (completed.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
                <p>No completed reports</p>
                <span>Complete an investigation to generate a report</span>
            </div>`;
        return;
    }

    // Show first completed report
    const report = await apiGet(`/api/v1/report/${completed[0].id}`);
    if (report?.data?.report) {
        renderReport(report.data.report, completed[0].id);
    }
}

function renderReport(report, caseId) {
    const container = document.getElementById('reportContent');

    const confidence = report.confidence_assessment?.overall_confidence || 
                       report.primary_conclusion?.confidence || 0;
    const confLevel = confidence > 0.7 ? 'high' : confidence > 0.4 ? 'medium' : 'low';

    container.innerHTML = `
        <div class="report-section">
            <h3>📋 ${escapeHtml(report.title || 'Investigation Report')}</h3>
            <p style="color: var(--text-secondary); line-height: 1.7;">${escapeHtml(report.summary || 'No summary available.')}</p>
        </div>

        <div class="report-section">
            <h3>🎯 Primary Conclusion</h3>
            <p style="margin-bottom: 0.5rem;">${escapeHtml(report.primary_conclusion?.hypothesis || 'No conclusion reached.')}</p>
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.1rem;">${(confidence * 100).toFixed(0)}%</span>
                <div class="confidence-bar" style="flex: 1;">
                    <div class="confidence-fill confidence-${confLevel}" style="width: ${confidence * 100}%"></div>
                </div>
            </div>
        </div>

        ${report.timeline ? `
        <div class="report-section">
            <h3>🕐 Timeline</h3>
            ${report.timeline.map(item => `
                <div style="display: flex; gap: 0.75rem; margin-bottom: 0.75rem; padding: 0.5rem; background: var(--bg-primary); border-radius: var(--radius-sm);">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--accent-cyan); min-width: 60px;">${item.time || '?'}</span>
                    <div>
                        <span>${escapeHtml(item.event || '')}</span>
                        ${item.is_inferred ? '<span class="tag inferred">inferred</span>' : ''}
                        ${!item.is_confirmed ? '<span class="tag unconfirmed">unconfirmed</span>' : ''}
                    </div>
                </div>
            `).join('')}
        </div>` : ''}

        ${report.unresolved_objections?.length ? `
        <div class="report-section">
            <h3>⚠️ Unresolved Objections</h3>
            ${report.unresolved_objections.map(obj => `
                <div class="objection-card ${obj.severity === 'CRITICAL' ? 'critical' : ''}">
                    <strong style="color: var(--accent-amber);">[${obj.severity}]</strong> 
                    ${escapeHtml(obj.objection || obj.objection_text || '')}
                </div>
            `).join('')}
        </div>` : ''}

        ${report.alternative_hypotheses?.length ? `
        <div class="report-section">
            <h3>🚫 Considered & Rejected</h3>
            ${report.alternative_hypotheses.map(h => `
                <div style="padding: 0.5rem; margin-bottom: 0.5rem; background: var(--bg-primary); border-radius: var(--radius-sm);">
                    <strong>${escapeHtml(h.hypothesis || h.title || '')}</strong>
                    <p style="font-size: 0.82rem; color: var(--text-muted); margin-top: 0.25rem;">${escapeHtml(h.rejection_reason || '')}</p>
                </div>
            `).join('')}
        </div>` : ''}

        ${report.metadata ? `
        <div class="report-section">
            <h3>📊 Metadata</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.5rem;">
                ${Object.entries(report.metadata).map(([k, v]) => `
                    <div style="padding: 0.5rem; background: var(--bg-primary); border-radius: var(--radius-sm);">
                        <span style="font-size: 0.72rem; color: var(--text-muted); display: block;">${k.replace(/_/g, ' ')}</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600;">${v}</span>
                    </div>
                `).join('')}
            </div>
        </div>` : ''}
    `;
}

// ── Counterfactual ───────────────────────────────────────────

async function runCounterfactual() {
    const caseId = document.getElementById('cfCaseId').value.trim();
    const evidenceId = document.getElementById('cfEvidenceId').value.trim();

    if (!caseId || !evidenceId) {
        showToast('Please enter both Case ID and Evidence ID', 'error');
        return;
    }

    showToast('Running counterfactual analysis...', 'info');

    const result = await apiPost('/api/v1/counterfactual/', {
        case_id: caseId,
        removed_evidence_id: evidenceId,
    });

    const panel = document.getElementById('cfResult');
    panel.style.display = 'block';

    if (result && result.status === 'success') {
        const data = result.data;
        panel.innerHTML = `
            <h4 style="margin-bottom: 1rem;">Counterfactual Result</h4>
            <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
                <div style="flex:1; padding: 1rem; background: var(--bg-primary); border-radius: var(--radius-sm);">
                    <span style="font-size: 0.75rem; color: var(--text-muted);">Original Conclusion</span>
                    <p style="font-weight: 600;">${escapeHtml(data.original_leading?.title || 'None')}</p>
                    <span style="font-family: 'JetBrains Mono', monospace; color: var(--accent-cyan);">${((data.original_leading?.confidence || 0) * 100).toFixed(0)}%</span>
                </div>
                <div style="flex:1; padding: 1rem; background: var(--bg-primary); border-radius: var(--radius-sm);">
                    <span style="font-size: 0.75rem; color: var(--text-muted);">Counterfactual Conclusion</span>
                    <p style="font-weight: 600;">${escapeHtml(data.counterfactual_leading?.title || 'None')}</p>
                    <span style="font-family: 'JetBrains Mono', monospace; color: var(--accent-purple);">${((data.counterfactual_leading?.confidence || 0) * 100).toFixed(0)}%</span>
                </div>
            </div>
            <div style="padding: 0.75rem; background: ${data.conclusion_changed ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)'}; border-radius: var(--radius-sm); font-weight: 600;">
                ${data.conclusion_changed ? '⚠️ Conclusion CHANGED — this evidence is critical!' : '✓ Conclusion unchanged — robust to this evidence removal.'}
            </div>
        `;
    } else {
        panel.innerHTML = `<p style="color: var(--accent-red);">Analysis failed: ${result?.message || 'Unknown error'}</p>`;
    }
}

// ── Utilities ────────────────────────────────────────────────

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

// ── File Upload Handler ──────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('videoDropZone');
    const fileInput = document.getElementById('videoFile');

    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', e => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--accent-blue)';
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.style.borderColor = 'var(--border-medium)';
        });
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--border-medium)';
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                dropZone.querySelector('p').textContent = e.dataTransfer.files[0].name;
            }
        });
    }

    // Check API health
    checkApiHealth();
    // Load dashboard
    refreshDashboard();
});

async function checkApiHealth() {
    const dot = document.getElementById('apiStatus');
    const text = document.getElementById('apiStatusText');

    try {
        const data = await apiGet('/api/health');
        if (data && data.status === 'healthy') {
            dot.className = 'status-dot connected';
            text.textContent = 'API Connected';
        } else {
            dot.className = 'status-dot disconnected';
            text.textContent = 'API Error';
        }
    } catch {
        dot.className = 'status-dot disconnected';
        text.textContent = 'API Offline';
    }
}
