/**
 * SatQuery AI - Monitoring & Alert System API Service
 * Handles CRUD for monitoring jobs, manual on-demand checks,
 * run status polling, and alert management.
 */

import { API_BASE_URL } from './agentApi.js';

/**
 * Fetch list of all active/paused monitoring jobs
 * @param {string} [sessionId] - Optional session ID filter
 * @returns {Promise<Array<Object>>}
 */
export async function fetchMonitors(sessionId = null) {
  const url = new URL(`${API_BASE_URL}/api/monitoring`);
  if (sessionId) {
    url.searchParams.append('session_id', sessionId);
  }

  const res = await fetch(url.toString(), {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch monitors: ${res.statusText}`);
  }

  return await res.json();
}

/**
 * Get detailed information and run history for a specific monitor
 * @param {string} monitorId
 * @returns {Promise<Object>}
 */
export async function getMonitor(monitorId) {
  const res = await fetch(`${API_BASE_URL}/api/monitoring/${monitorId}`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch monitor details: ${res.statusText}`);
  }

  return await res.json();
}

/**
 * Create a new monitoring job
 * @param {Object} monitorData
 * @returns {Promise<Object>}
 */
export async function createMonitor(monitorData) {
  const res = await fetch(`${API_BASE_URL}/api/monitoring`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(monitorData),
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Failed to create monitor: ${res.statusText}`);
  }

  return await res.json();
}

/**
 * Update monitor attributes (e.g. status: 'paused' | 'active', threshold, name)
 * @param {string} monitorId
 * @param {Object} updates
 * @returns {Promise<Object>}
 */
export async function updateMonitor(monitorId, updates) {
  const res = await fetch(`${API_BASE_URL}/api/monitoring/${monitorId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(updates),
  });

  if (!res.ok) {
    throw new Error(`Failed to update monitor: ${res.statusText}`);
  }

  return await res.json();
}

/**
 * Delete a monitor job
 * @param {string} monitorId
 * @returns {Promise<{ deleted: boolean }>}
 */
export async function deleteMonitor(monitorId) {
  const res = await fetch(`${API_BASE_URL}/api/monitoring/${monitorId}`, {
    method: 'DELETE',
    headers: { 'Accept': 'application/json' },
  });

  if (!res.ok) {
    throw new Error(`Failed to delete monitor: ${res.statusText}`);
  }

  return await res.json();
}

/**
 * Trigger an immediate on-demand satellite check for a monitor
 * @param {string} monitorId
 * @returns {Promise<{ job_id: string, status: string, message: string, monitor_id: string }>}
 */
export async function triggerCheckNow(monitorId) {
  const res = await fetch(`${API_BASE_URL}/api/monitoring/${monitorId}/check-now`, {
    method: 'POST',
    headers: { 'Accept': 'application/json' },
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Failed to trigger check: ${res.statusText}`);
  }

  return await res.json();
}

/**
 * Retrieve execution status of an asynchronous monitoring run
 * @param {string} runId
 * @returns {Promise<Object>}
 */
export async function getRunStatus(runId) {
  const res = await fetch(`${API_BASE_URL}/api/monitoring/jobs/${runId}`, {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });

  if (!res.ok) {
    throw new Error(`Failed to get run status: ${res.statusText}`);
  }

  return await res.json();
}

/**
 * Poll a check-now run until finished or timeout
 * @param {string} runId
 * @param {Object} [options]
 * @param {number} [options.intervalMs=2000]
 * @param {number} [options.maxWaitMs=95000]
 * @param {Function} [options.onProgress]
 * @returns {Promise<Object>}
 */
export async function pollRunUntilComplete(runId, { intervalMs = 2000, maxWaitMs = 95000, onProgress = null } = {}) {
  const startTime = Date.now();

  while (Date.now() - startTime < maxWaitMs) {
    const run = await getRunStatus(runId);
    if (onProgress) {
      onProgress(run);
    }

    if (run.status === 'completed' || run.status === 'failed' || run.status === 'skipped') {
      return run;
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new Error(`Satellite check run ${runId} timed out after ${Math.round(maxWaitMs / 1000)} seconds.`);
}

/**
 * Fetch system alerts
 * @param {Object} [params]
 * @param {string} [params.monitorId]
 * @param {boolean} [params.unreadOnly]
 * @param {number} [params.limit]
 * @returns {Promise<Array<Object>>}
 */
export async function fetchAlerts({ monitorId = null, unreadOnly = false, limit = 50 } = {}) {
  const url = new URL(`${API_BASE_URL}/api/alerts`);
  if (monitorId) url.searchParams.append('monitor_id', monitorId);
  if (unreadOnly) url.searchParams.append('unread_only', 'true');
  if (limit) url.searchParams.append('limit', String(limit));

  const res = await fetch(url.toString(), {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch alerts: ${res.statusText}`);
  }

  return await res.json();
}

/**
 * Fetch unread alert count
 * @param {string} [sessionId]
 * @returns {Promise<{ unread_count: number }>}
 */
export async function fetchUnreadAlertsCount(sessionId = null) {
  const url = new URL(`${API_BASE_URL}/api/alerts/unread-count`);
  if (sessionId) {
    url.searchParams.append('session_id', sessionId);
  }

  const res = await fetch(url.toString(), {
    method: 'GET',
    headers: { 'Accept': 'application/json' },
  });

  if (!res.ok) {
    return { unread_count: 0 };
  }

  return await res.json();
}

/**
 * Mark an alert as read
 * @param {string} alertId
 * @returns {Promise<Object>}
 */
export async function markAlertRead(alertId) {
  const res = await fetch(`${API_BASE_URL}/api/alerts/${alertId}/read`, {
    method: 'PATCH',
    headers: { 'Accept': 'application/json' },
  });

  if (!res.ok) {
    throw new Error(`Failed to mark alert read: ${res.statusText}`);
  }

  return await res.json();
}

/**
 * Delete / dismiss an alert
 * @param {string} alertId
 * @returns {Promise<{ deleted: boolean }>}
 */
export async function deleteAlert(alertId) {
  const res = await fetch(`${API_BASE_URL}/api/alerts/${alertId}`, {
    method: 'DELETE',
    headers: { 'Accept': 'application/json' },
  });

  if (!res.ok) {
    throw new Error(`Failed to delete alert: ${res.statusText}`);
  }

  return await res.json();
}
