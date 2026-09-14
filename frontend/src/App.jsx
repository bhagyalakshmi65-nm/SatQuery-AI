import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header/Header.jsx';
import ImageViewer from './components/ImageViewer/ImageViewer.jsx';
import QueryInput from './components/QueryInput/QueryInput.jsx';
import AnalysisProgress from './components/AnalysisProgress/AnalysisProgress.jsx';
import ResultsPanel from './components/ResultsPanel/ResultsPanel.jsx';
import EvidencePanel from './components/EvidencePanel/EvidencePanel.jsx';
import ExecutionTrace from './components/ExecutionTrace/ExecutionTrace.jsx';
import ArchitectureSection from './components/ArchitectureSection/ArchitectureSection.jsx';
import { analyzeWithAgent, checkBackendHealth } from './services/agentApi.js';

export default function App() {
  const [files, setFiles] = useState([]);
  const [query, setQuery] = useState('');
  const [lastQuery, setLastQuery] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState({ online: false, specialists: [] });

  // Probe backend health periodically
  const checkHealth = useCallback(async () => {
    const status = await checkBackendHealth();
    setBackendStatus(status);
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  const handleSubmit = async () => {
    if (files.length === 0) {
      setError('Please upload at least one satellite image file first.');
      return;
    }
    if (!query.trim()) {
      setError('Please enter a natural language question about the satellite scene.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setLastQuery(query);

    try {
      const result = await analyzeWithAgent(query, files);
      setResponse(result);
    } catch (err) {
      console.error('Agent analysis error:', err);
      setError(err.message || 'An unexpected error occurred during agent analysis.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="satquery-app">
      <Header backendStatus={backendStatus} />

      <main className="app-main">
        <div className="workspace-grid">
          {/* Left Column: Satellite Viewer & HUD Stage */}
          <ImageViewer
            files={files}
            onFilesChange={setFiles}
            isAnalyzing={isAnalyzing}
            evidence={response?.evidence}
          />

          {/* Right Column: Query & Evidence-Grounded Results */}
          <div className="query-results-column" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <QueryInput
              query={query}
              onQueryChange={setQuery}
              onSubmit={handleSubmit}
              isAnalyzing={isAnalyzing}
              hasImages={files.length > 0}
            />

            <AnalysisProgress isAnalyzing={isAnalyzing} />

            <ResultsPanel
              response={response}
              lastQuery={lastQuery}
              error={error}
            />

            {response?.evidence && (
              <EvidencePanel evidence={response.evidence} />
            )}

            {response?.execution_trace && (
              <ExecutionTrace trace={response.execution_trace} />
            )}
          </div>
        </div>

        {/* System Architecture Section */}
        <ArchitectureSection />
      </main>
    </div>
  );
}

