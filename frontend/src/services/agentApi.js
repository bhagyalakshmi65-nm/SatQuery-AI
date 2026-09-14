/**
 * SatQuery AI - Real Agent API Client Service
 * Connects directly to FastAPI backend (POST /api/agent/analyze).
 * Retains raw File objects and dispatches multipart/form-data.
 */

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Check backend health status
 * @returns {Promise<{ online: boolean, specialists: string[], error?: string }>}
 */
export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) {
      return { online: false, specialists: [], error: `HTTP ${res.status}: ${res.statusText}` };
    }
    const data = await res.json();
    return {
      online: data.status === 'healthy',
      specialists: data.available_specialists || [],
    };
  } catch (err) {
    return {
      online: false,
      specialists: [],
      error: err.message || 'Cannot reach SatQuery backend service',
    };
  }
}

/**
 * Execute real agent query analysis via multipart/form-data
 * @param {string} query - Natural language query string
 * @param {File[]} files - Array of raw JavaScript File objects
 * @param {Object} [options] - Optional settings (sensorType, metadata, timeoutMs)
 * @returns {Promise<{
 *   answer: string,
 *   task: string,
 *   evidence: Object,
 *   confidence: number|null,
 *   execution_trace: { task: string, tools_used: string[], steps: string[], execution_time_ms: number },
 *   error: string|null
 * }>}
 */
export async function analyzeWithAgent(query, files = [], options = {}) {
  const trimmedQuery = (query || '').trim();
  if (!trimmedQuery) {
    throw new Error('Please provide a query for satellite image analysis.');
  }

  const formData = new FormData();
  formData.append('query', trimmedQuery);

  // Append each File object to 'images' key
  if (Array.isArray(files)) {
    for (const file of files) {
      if (file instanceof File) {
        formData.append('images', file, file.name);
      }
    }
  }

  if (options.sensorType) {
    formData.append('sensor_type', options.sensorType);
  }

  if (options.metadata) {
    formData.append('metadata', typeof options.metadata === 'string' ? options.metadata : JSON.stringify(options.metadata));
  }

  const timeoutMs = options.timeoutMs || 120000; // 2 minutes for deep learning pipeline

  try {
    // CRITICAL: Do NOT set Content-Type header manually for FormData.
    // The browser automatically sets multipart/form-data with the boundary parameter.
    const res = await fetch(`${API_BASE_URL}/api/agent/analyze`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(timeoutMs),
    });

    if (!res.ok) {
      let detail = `Server responded with HTTP ${res.status} (${res.statusText})`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) {
          detail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        }
      } catch {
        // Fall back to HTTP status message
      }
      throw new Error(detail);
    }

    const data = await res.json();

    // Check if backend payload returned an error
    if (data.error && !data.answer) {
      throw new Error(data.error);
    }

    return {
      answer: data.answer || 'No analysis generated.',
      task: data.task || 'unknown',
      evidence: data.evidence || {},
      confidence: data.confidence !== undefined ? data.confidence : null,
      execution_trace: data.execution_trace || {
        task: data.task || 'unknown',
        tools_used: [],
        steps: [],
        execution_time_ms: 0,
      },
      error: data.error || null,
    };
  } catch (err) {
    if (err.name === 'TimeoutError') {
      throw new Error('Analysis timed out. The specialist model execution exceeded the allowed duration.');
    }
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error(`Failed to connect to backend server at ${API_BASE_URL}. Ensure FastAPI is running on port 8000.`);
    }
    throw err;
  }
}

