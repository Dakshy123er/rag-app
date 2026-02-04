import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function App() {
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const [activeTab, setActiveTab] = useState('upload');

  // Upload state
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [title, setTitle] = useState('');
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState({ message: '', type: '' });

  // Query state
  const [query, setQuery] = useState('');
  const [queryLoading, setQueryLoading] = useState(false);
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [timing, setTiming] = useState(null);
  const [tokenEstimate, setTokenEstimate] = useState(null);
  const [queryError, setQueryError] = useState('');

  // Stats
  const [stats, setStats] = useState(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  const formatMs = (seconds) =>
    typeof seconds === 'number' ? `${(seconds * 1000).toFixed(0)}ms` : '—';

  const formatCost = (usd) =>
    typeof usd === 'number' ? `$${usd.toFixed(6)}` : '—';

  const renderAnswer = (text) => {
    if (!text) return null;

    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h3: ({ children }) => <h3 className="md-heading">{children}</h3>,
          p: ({ children }) => <p className="md-paragraph">{children}</p>,
          li: ({ children }) => <li className="md-list-item">{children}</li>,
          strong: ({ children }) => <strong className="md-bold">{children}</strong>,
          ul: ({ children }) => <ul className="md-list">{children}</ul>,
          ol: ({ children }) => <ol className="md-list">{children}</ol>
        }}
      >
        {text}
      </ReactMarkdown>
    );
  };

  // Fetch stats
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/stats`);
      if (!res.ok) throw new Error('Stats fetch failed');
      const data = await res.json();
      setStats(data);
    } catch (err) {
      console.error('Stats fetch failed:', err);
      setStats(null);
    }
  }, [API_URL]);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  // Upload handler
  const handleUpload = async (e) => {
    e.preventDefault();
    setUploadLoading(true);
    setUploadStatus({ message: '', type: '' });

    const formData = new FormData();
    if (file) formData.append('file', file);
    if (text) formData.append('text', text);
    formData.append('title', title || 'Untitled');
    formData.append('source', 'web_upload');

    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      setUploadStatus({
        message: `✓ ${data.message} (${data.total_tokens.toLocaleString()} tokens)`,
        type: 'success'
      });
      setText('');
      setFile(null);
      setTitle('');
      
      // Clear file input
      const fileInput = document.querySelector('input[type="file"]');
      if (fileInput) fileInput.value = '';
      
      fetchStats();
    } catch (err) {
      setUploadStatus({
        message: `✗ ${err.message}`,
        type: 'error'
      });
    } finally {
      setUploadLoading(false);
    }
  };

  // Query handler
  const handleQuery = async (e) => {
    e.preventDefault();
    setQueryLoading(true);
    setAnswer('');
    setSources([]);
    setTiming(null);
    setTokenEstimate(null);
    setQueryError('');

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          top_k: 15,
          rerank_top_n: 5
        })
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Query failed');
      }

      setAnswer(data.answer || 'No answer generated.');
      setSources(data.sources || []);
      setTiming(data.timing || {});
      setTokenEstimate(data.token_estimate || {});
    } catch (err) {
      setQueryError(err.message);
      setAnswer('');
    } finally {
      setQueryLoading(false);
    }
  };

  // Clear handler
  const handleClear = async () => {
    try {
      const res = await fetch(`${API_URL}/clear`, { method: 'DELETE' });
      
      if (!res.ok) throw new Error('Clear failed');
      
      setUploadStatus({ 
        message: '✓ All documents cleared successfully', 
        type: 'success' 
      });
      setShowClearConfirm(false);
      fetchStats();
      
      // Clear query results if any
      setAnswer('');
      setSources([]);
      setTiming(null);
      setTokenEstimate(null);
    } catch (err) {
      setUploadStatus({ 
        message: `✗ ${err.message}`, 
        type: 'error' 
      });
    }
  };

  return (
    <div className="App">
      <div className="bg-gradient"></div>

      <header className="header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">◆</div>
            <div>
              <h1>Mini RAG System</h1>
            </div>
          </div>

          {stats && (
            <div className="stats">
              <div className="stat-item">
                <span className="stat-value">
                  {stats.total_vectors != null ? stats.total_vectors.toLocaleString() : '—'}
                </span>
                <span className="stat-label">Vectors</span>
              </div>
              <div className="stat-divider"></div>
              <div className="stat-item">
                <span className="stat-value">
                  {stats.dimension != null ? stats.dimension.toLocaleString() : '—'}
                </span>
                <span className="stat-label">Dimension</span>
              </div>
            </div>
          )}
        </div>
      </header>

      <div className="container">
        <div className="tabs">
          <button
            className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
             Upload
          </button>
          <button
            className={`tab ${activeTab === 'query' ? 'active' : ''}`}
            onClick={() => setActiveTab('query')}
          >
             Query
          </button>
        </div>

        <div className="tab-content-wrapper">
          {activeTab === 'upload' && (
            <div className="tab-content upload-content">
              <h2>Add Documents</h2>
              <p className="section-description">
                Upload PDF, TXT, or MD files, or paste text directly
              </p>

              <form onSubmit={handleUpload}>
                <div className="form-group">
                  <label htmlFor="title">Document Title (Optional)</label>
                  <input
                    id="title"
                    type="text"
                    placeholder="e.g., Research Paper 2024"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="file">Upload File</label>
                  <input
                    id="file"
                    type="file"
                    accept=".pdf,.txt,.md"
                    onChange={(e) => setFile(e.target.files[0])}
                  />
                  {file && (
                    <div className="file-info">
                      Selected: {file.name} ({(file.size / 1024).toFixed(1)} KB)
                    </div>
                  )}
                </div>

                <div className="form-group">
                  <label htmlFor="text">Or Paste Text</label>
                  <textarea
                    id="text"
                    rows={8}
                    placeholder="Paste your text content here..."
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                  />
                  {text && (
                    <div className="text-info">
                      Characters: {text.length.toLocaleString()}
                    </div>
                  )}
                </div>

                <button 
                  type="submit"
                  className="btn-primary"
                  disabled={uploadLoading || (!file && !text)}
                >
                  {uploadLoading ? '⏳ Processing...' : '✓ Upload & Index'}
                </button>
              </form>

              {uploadStatus.message && (
                <div className={`status-message ${uploadStatus.type}`}>
                  {uploadStatus.message}
                </div>
              )}

              <div className="danger-zone">
                <h3>Danger Zone</h3>
                {!showClearConfirm ? (
                  <button 
                    className="btn-danger" 
                    onClick={() => setShowClearConfirm(true)}
                  >
                    Clear All Documents
                  </button>
                ) : (
                  <div className="confirm-clear">
                    <p>⚠️ This will permanently delete all indexed documents. Continue?</p>
                    <div className="confirm-buttons">
                      <button 
                        className="btn-danger-confirm" 
                        onClick={handleClear}
                      >
                        Yes, Delete All
                      </button>
                      <button 
                        className="btn-cancel" 
                        onClick={() => setShowClearConfirm(false)}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'query' && (
            <div className="tab-content query-content">
              <h2>Ask Questions</h2>

              <form onSubmit={handleQuery}>
                <div className="form-group">
                  <label htmlFor="query">Your Question</label>
                  <textarea
                    id="query"
                    rows={4}
                    placeholder="What would you like to know?"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </div>

                <button 
                  type="submit"
                  className="btn-primary"
                  disabled={queryLoading || !query.trim()}
                >
                  {queryLoading ? 'Searching...' : 'Search'}
                </button>
              </form>

              {queryError && (
                <div className="status-message error">
                  ✗ {queryError}
                </div>
              )}

              {answer && (
                <div className="results-section">
                  <div className="answer-section">
                    <h3>Answer</h3>
                    <div className="answer-content">
                      {renderAnswer(answer)}
                    </div>

                    <div className="metadata-grid">
                      {/* Performance Metrics */}
                      {timing && Object.keys(timing).length > 0 && (
                        <div className="metrics-card">
                          <h4>Performance</h4>
                          <div className="metrics-list">
                            {Object.entries(timing).map(([key, value]) => (
                              <div key={key} className="metric-row">
                                <span className="metric-label">
                                  {key.charAt(0).toUpperCase() + key.slice(1)}:
                                </span>
                                <span className="metric-value">{formatMs(value)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Token Usage */}
                      {tokenEstimate && tokenEstimate.total_tokens && (
                        <div className="cost-card">
                          <h4>Token Usage</h4>
                          <div className="metrics-list">
                            <div className="metric-row">
                              <span className="metric-label">Embedding:</span>
                              <span className="metric-value">
                                {tokenEstimate.embedding_tokens?.toLocaleString() || '—'}
                              </span>
                            </div>
                            <div className="metric-row">
                              <span className="metric-label">LLM Input:</span>
                              <span className="metric-value">
                                {tokenEstimate.llm_input_tokens?.toLocaleString() || '—'}
                              </span>
                            </div>
                            <div className="metric-row">
                              <span className="metric-label">LLM Output:</span>
                              <span className="metric-value">
                                {tokenEstimate.llm_output_tokens?.toLocaleString() || '—'}
                              </span>
                            </div>
                            <div className="metric-row metric-total">
                              <span className="metric-label">Total:</span>
                              <span className="metric-value">
                                {tokenEstimate.total_tokens?.toLocaleString() || '—'}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Cost Breakdown */}
                      {tokenEstimate?.costs && (
                        <div className="cost-card">
                          <h4>Cost Breakdown</h4>
                          <div className="metrics-list">
                            <div className="metric-row">
                              <span className="metric-label">Embedding:</span>
                              <span className="metric-value">
                                {formatCost(tokenEstimate.costs.embedding_usd)}
                              </span>
                            </div>
                            <div className="metric-row">
                              <span className="metric-label">LLM Input:</span>
                              <span className="metric-value">
                                {formatCost(tokenEstimate.costs.llm_input_usd)}
                              </span>
                            </div>
                            <div className="metric-row">
                              <span className="metric-label">LLM Output:</span>
                              <span className="metric-value">
                                {formatCost(tokenEstimate.costs.llm_output_usd)}
                              </span>
                            </div>
                            <div className="metric-row">
                              <span className="metric-label">Reranking:</span>
                              <span className="metric-value">
                                {formatCost(tokenEstimate.costs.rerank_usd)}
                              </span>
                            </div>
                            <div className="metric-row metric-total">
                              <span className="metric-label">Total:</span>
                              <span className="metric-value">
                                {formatCost(tokenEstimate.costs.total_usd)}
                              </span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Sources */}
                  {sources.length > 0 && (
                    <div className="sources-section">
                      <h3>Sources ({sources.length})</h3>
                      <p className="sources-description">
                        Retrieved and reranked chunks used to generate the answer
                      </p>
                      {sources.map((source, idx) => (
                        <div key={idx} className="source-card">
                          <div className="source-header">
                            <strong className="source-title">
                              [{source.id}] {source.metadata?.title || 'Untitled Document'}
                            </strong>
                            <span className="source-score">
                              Score: {(source.score * 100).toFixed(1)}%
                            </span>
                          </div>
                          <p className="source-preview">{source.preview || source.text}</p>
                          <div className="source-metadata">
                            <span className="metadata-item">
                              📄 Chunk {source.metadata?.position || '—'}
                            </span>
                            <span className="metadata-item">
                              🔢 {source.metadata?.token_count?.toLocaleString() || '—'} tokens
                            </span>
                            {source.metadata?.source && (
                              <span className="metadata-item">
                                📁 {source.metadata.source}
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <footer className="footer">
        <div className="footer-content">
          <span>Built with React & FastAPI</span>
          <span className="footer-divider">•</span>
          <span>Powered by OpenAI, Pinecone & Cohere</span>
        </div>
      </footer>
    </div>
  );
}

export default App;