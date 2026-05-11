import { useState } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Scatter, ScatterChart,
  Legend
} from 'recharts';
import './index.css';

const API_URL = 'https://ml-comparator-api.onrender.com/api';

const DATASET_OPTIONS = [
  { value: 'sine', label: '🌊 Sine Wave' },
  { value: 'quadratic', label: '📈 Quadratic' },
  { value: 'cubic', label: '〰️ Cubic' },
  { value: 'linear', label: '📏 Linear' },
];

const KERNEL_OPTIONS = [
  { value: 'rbf', label: 'RBF (Radial Basis Function)' },
  { value: 'linear', label: 'Linear' },
  { value: 'poly', label: 'Polynomial' },
  { value: 'sigmoid', label: 'Sigmoid' },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: '#1a2236', border: '1px solid #2a3550',
        borderRadius: '10px', padding: '10px 14px', fontSize: '0.8rem'
      }}>
        <p style={{ color: '#8b99b5', marginBottom: '6px' }}>x = {Number(label).toFixed(3)}</p>
        {payload.map((p) => (
          <p key={p.name} style={{ color: p.color, margin: '2px 0', fontFamily: 'JetBrains Mono, monospace' }}>
            {p.name}: {Number(p.value).toFixed(4)}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function App() {
  const [config, setConfig] = useState({
    datasetType: 'sine',
    nSamples: 120,
    noiseLevel: 0.5,
    polyDegree: 3,
    svrC: 1.0,
    svrEpsilon: 0.1,
    svrKernel: 'rbf',
  });

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predX, setPredX] = useState(0);
  const [predResult, setPredResult] = useState(null);
  const [predLoading, setPredLoading] = useState(false);
  const [toast, setToast] = useState(null);
  const [apiStatus, setApiStatus] = useState('online');

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  };

  const handleConfig = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const trainModels = async () => {
    setLoading(true);
    setPredResult(null);
    try {
      const { data } = await axios.post(`${API_URL}/train`, config);
      setResults(data);
      setApiStatus('online');
      showToast('✅ Models trained successfully!');
    } catch (err) {
      setApiStatus('offline');
      showToast('❌ Backend not reachable. Start Flask server.');
    } finally {
      setLoading(false);
    }
  };

  const predictSingle = async () => {
    setPredLoading(true);
    try {
      const { data } = await axios.post(`${API_URL}/predict`, { xValue: predX, ...config });
      setPredResult(data);
    } catch {
      showToast('❌ Prediction failed.');
    } finally {
      setPredLoading(false);
    }
  };

  // Build chart data
  const chartData = results
    ? results.xValues.map((x, i) => ({
      x: Number(x.toFixed(3)),
      Actual: Number(results.yActual[i].toFixed(4)),
      Linear: Number(results.yLinear[i].toFixed(4)),
      Polynomial: Number(results.yPoly[i].toFixed(4)),
      SVR: Number(results.ySVR[i].toFixed(4)),
    }))
    : [];

  const getR2Color = (r2) => {
    if (r2 >= 0.9) return '#10b981';
    if (r2 >= 0.7) return '#f59e0b';
    return '#ef4444';
  };

  const getBestModel = () => {
    if (!results) return null;
    const best = results.metrics.reduce((a, b) => (a.r2 > b.r2 ? a : b));
    return best.model;
  };

  return (
    <>
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon">🤖</div>
            <div className="logo-text">
              <h1>ML Model Comparator</h1>
              <p>Linear · Polynomial · SVR</p>
            </div>
          </div>
          <div className="header-badge">
            <span className="status-dot" style={{ background: apiStatus === 'online' ? '#10b981' : '#ef4444' }} />
            {apiStatus === 'online' ? 'API Connected' : 'API Offline'}
          </div>
        </div>
      </header>

      <main className="main">
        {/* Hero Banner */}
        <div className="hero-banner">
          <div className="hero-text">
            <h2>Regression Model Comparison</h2>
            <p>Train and compare Linear, Polynomial, and SVR models on various datasets.
              Tune hyperparameters, visualize predictions, and evaluate performance metrics interactively.</p>
            <div className="hero-badges">
              <span className="model-badge linear">📏 Linear Regression</span>
              <span className="model-badge poly">🔢 Polynomial Regression</span>
              <span className="model-badge svr">⚙️ Support Vector Regression</span>
            </div>
          </div>
          <div className="hero-stats">
            <div className="stat-box">
              <div className="val">3</div>
              <div className="lbl">Models</div>
            </div>
            <div className="stat-box">
              <div className="val">4</div>
              <div className="lbl">Metrics</div>
            </div>
            <div className="stat-box">
              <div className="val">4</div>
              <div className="lbl">Datasets</div>
            </div>
          </div>
        </div>

        {/* LEFT: Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Dataset Config */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-icon blue">📊</div>
              <div>
                <div className="panel-title">Dataset Configuration</div>
                <div className="panel-subtitle">Choose data source & parameters</div>
              </div>
            </div>
            <div className="panel-body">
              <div className="control-group">
                <div className="control-label">Dataset Type</div>
                <select className="select-input" value={config.datasetType}
                  onChange={e => handleConfig('datasetType', e.target.value)}>
                  {DATASET_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>

              <div className="control-group">
                <div className="control-label">
                  <span>Number of Samples</span>
                  <span className="control-value">{config.nSamples}</span>
                </div>
                <input type="range" className="range-input" min="30" max="300" step="10"
                  value={config.nSamples} onChange={e => handleConfig('nSamples', Number(e.target.value))} />
              </div>

              <div className="control-group">
                <div className="control-label">
                  <span>Noise Level</span>
                  <span className="control-value">{config.noiseLevel}</span>
                </div>
                <input type="range" className="range-input" min="0" max="3" step="0.1"
                  value={config.noiseLevel} onChange={e => handleConfig('noiseLevel', Number(e.target.value))} />
              </div>
            </div>
          </div>

          {/* Model Hyperparams */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-icon purple">⚙️</div>
              <div>
                <div className="panel-title">Model Hyperparameters</div>
                <div className="panel-subtitle">Tune each model</div>
              </div>
            </div>
            <div className="panel-body">
              <div className="section-label">Polynomial Regression</div>
              <div className="control-group">
                <div className="control-label">
                  <span>Degree</span>
                  <span className="control-value">{config.polyDegree}</span>
                </div>
                <input type="range" className="range-input" min="1" max="8" step="1"
                  value={config.polyDegree} onChange={e => handleConfig('polyDegree', Number(e.target.value))} />
              </div>

              <div className="section-divider" />
              <div className="section-label">Support Vector Regression (SVR)</div>

              <div className="control-group">
                <div className="control-label">Kernel</div>
                <select className="select-input" value={config.svrKernel}
                  onChange={e => handleConfig('svrKernel', e.target.value)}>
                  {KERNEL_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>

              <div className="control-group">
                <div className="control-label">
                  <span>C (Regularization)</span>
                  <span className="control-value">{config.svrC}</span>
                </div>
                <input type="range" className="range-input" min="0.1" max="10" step="0.1"
                  value={config.svrC} onChange={e => handleConfig('svrC', Number(e.target.value))} />
              </div>

              <div className="control-group">
                <div className="control-label">
                  <span>Epsilon (ε)</span>
                  <span className="control-value">{config.svrEpsilon}</span>
                </div>
                <input type="range" className="range-input" min="0.01" max="1" step="0.01"
                  value={config.svrEpsilon} onChange={e => handleConfig('svrEpsilon', Number(e.target.value))} />
              </div>

              <button className="btn-primary" onClick={trainModels} disabled={loading}>
                {loading ? <><span>⏳</span> Training Models...</> : <><span>🚀</span> Train All Models</>}
              </button>
            </div>
          </div>

          {/* Single Prediction */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-icon orange">🎯</div>
              <div>
                <div className="panel-title">Point Prediction</div>
                <div className="panel-subtitle">Predict for a custom X value</div>
              </div>
            </div>
            <div className="panel-body">
              <div className="control-group">
                <div className="control-label">X Input Value</div>
                <div className="predict-row">
                  <input type="number" className="number-input" value={predX} step="0.1"
                    onChange={e => setPredX(Number(e.target.value))} placeholder="e.g. 1.5" />
                  <button className="btn-predict" onClick={predictSingle} disabled={predLoading}>
                    {predLoading ? '⏳' : '→ Predict'}
                  </button>
                </div>
              </div>

              {predResult && (
                <div className="predict-results fade-in">
                  <div className="predict-card linear">
                    <span className="name">📏 Linear</span>
                    <span className="value">{predResult.linearPred}</span>
                  </div>
                  <div className="predict-card poly">
                    <span className="name">🔢 Polynomial</span>
                    <span className="value">{predResult.polyPred}</span>
                  </div>
                  <div className="predict-card svr">
                    <span className="name">⚙️ SVR</span>
                    <span className="value">{predResult.svrPred}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT: Charts + Metrics */}
        <div className="right-col">
          {/* Chart */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-icon blue">📈</div>
              <div>
                <div className="panel-title">Model Predictions vs Actual</div>
                <div className="panel-subtitle">
                  {results ? `${results.nSamples} samples · ${results.datasetType} dataset` : 'Train models to see results'}
                </div>
              </div>
              {results && getBestModel() && (
                <div style={{
                  marginLeft: 'auto', fontSize: '0.75rem',
                  background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)',
                  borderRadius: '6px', padding: '0.3rem 0.7rem', color: '#10b981'
                }}>
                  🏆 Best: {getBestModel()}
                </div>
              )}
            </div>
            <div className="panel-body">
              {!results && !loading && (
                <div className="empty-state">
                  <div className="empty-icon">📉</div>
                  <h3>No Results Yet</h3>
                  <p>Configure your dataset and hyperparameters, then click "Train All Models" to see predictions.</p>
                </div>
              )}

              {loading && (
                <div className="loading-overlay">
                  <div className="spinner" />
                  <p className="loading-text">Training Linear, Polynomial & SVR models…</p>
                </div>
              )}

              {results && !loading && (
                <>
                  <div className="chart-legend">
                    {[
                      { key: 'Actual', color: '#f59e0b', dash: '4 4' },
                      { key: 'Linear', color: '#3b82f6' },
                      { key: 'Polynomial', color: '#8b5cf6' },
                      { key: 'SVR', color: '#10b981' },
                    ].map(l => (
                      <div key={l.key} className="legend-item">
                        <div className="legend-line" style={{ background: l.color }} />
                        {l.key}
                      </div>
                    ))}
                  </div>
                  <div className="chart-wrapper fade-in">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2a3550" />
                        <XAxis dataKey="x" stroke="#4b5a72" tick={{ fontSize: 11 }} />
                        <YAxis stroke="#4b5a72" tick={{ fontSize: 11 }} />
                        <Tooltip content={<CustomTooltip />} />
                        <Line type="monotone" dataKey="Actual" stroke="#f59e0b"
                          strokeDasharray="5 5" strokeWidth={1.5} dot={false} />
                        <Line type="monotone" dataKey="Linear" stroke="#3b82f6"
                          strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="Polynomial" stroke="#8b5cf6"
                          strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="SVR" stroke="#10b981"
                          strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Metrics */}
          {results && !loading && (
            <div className="panel fade-in">
              <div className="panel-header">
                <div className="panel-icon green">📊</div>
                <div>
                  <div className="panel-title">Performance Metrics</div>
                  <div className="panel-subtitle">R², MSE, RMSE, MAE comparison</div>
                </div>
              </div>
              <div className="panel-body">
                <div className="metrics-grid">
                  {results.metrics.map((m, i) => {
                    const classes = ['linear', 'poly', 'svr'];
                    const cls = classes[i];
                    const r2Pct = Math.max(0, Math.min(100, m.r2 * 100));
                    return (
                      <div key={m.model} className={`metric-card ${cls}`}>
                        <div className="metric-card-header">
                          <div className={`model-dot ${cls}`} />
                          <span className="model-name">{m.model}</span>
                        </div>
                        <div className="metric-rows">
                          <div className="metric-row">
                            <span className="metric-key">R² Score</span>
                            <span className="metric-val" style={{ color: getR2Color(m.r2) }}>{m.r2}</span>
                          </div>
                          <div className="metric-row">
                            <span className="metric-key">MSE</span>
                            <span className="metric-val">{m.mse}</span>
                          </div>
                          <div className="metric-row">
                            <span className="metric-key">RMSE</span>
                            <span className="metric-val">{m.rmse}</span>
                          </div>
                          <div className="metric-row">
                            <span className="metric-key">MAE</span>
                            <span className="metric-val">{m.mae}</span>
                          </div>
                        </div>
                        <div className="r2-bar">
                          <div className="r2-label">R² Score: {(r2Pct).toFixed(1)}%</div>
                          <div className="r2-track">
                            <div className={`r2-fill ${cls}`} style={{ width: `${r2Pct}%` }} />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {toast && (
        <div className="toast">{toast}</div>
      )}
    </>
  );
}
