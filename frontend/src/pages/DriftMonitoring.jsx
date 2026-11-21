import React, { useState } from 'react';
import { 
  LineChart as LineChartIcon, 
  ShieldCheck, 
  AlertTriangle, 
  AlertOctagon, 
  Play, 
  FileText, 
  BarChart2, 
  Filter 
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine, Cell 
} from 'recharts';
import { triggerDriftAnalysis } from '../services/api';

export default function DriftMonitoring({ driftReport, onRefresh }) {
  const [activeSubTab, setActiveSubTab] = useState('metrics'); // 'metrics' or 'evidently'
  const [evaluating, setEvaluating] = useState(false);
  const [feedback, setFeedback] = useState(null);

  const report = driftReport?.report;
  const status = report?.status || 'Healthy';
  const isHealthy = status === 'Healthy';
  const isWarning = status === 'Warning';
  const isCritical = status === 'Critical';

  // Prepare PSI chart data
  const featureMetrics = report?.feature_metrics || {};
  const psiChartData = Object.keys(featureMetrics).map(f => ({
    feature: f,
    psi: featureMetrics[f]?.psi || 0.0,
    status: featureMetrics[f]?.psi_status || 'HEALTHY',
    p_value: featureMetrics[f]?.ks_pvalue || 1.0,
    ks_drifted: featureMetrics[f]?.ks_drifted || false
  })).sort((a, b) => b.psi - a.psi).slice(0, 15); // Top 15 features

  const handleRunAdHocDrift = async () => {
    setEvaluating(true);
    setFeedback(null);
    try {
      const res = await triggerDriftAnalysis({ sample_size: 2000, generate_evidently_html: true });
      setFeedback({ type: 'success', message: `Drift evaluation completed! Status: ${res.status}, Drift Score: ${res.drift_score}` });
      onRefresh();
    } catch (err) {
      setFeedback({ type: 'error', message: `Evaluation failed: ${err.message}` });
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
            <LineChartIcon className="w-6 h-6 text-sky-400" />
            <span>Data Drift & Statistical Distribution Monitoring</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Detects input feature distribution shifts P(X) using Population Stability Index (PSI), Two-Sample KS Tests, and Evidently AI.
          </p>
        </div>

        <button
          disabled={evaluating}
          onClick={handleRunAdHocDrift}
          className="mt-4 sm:mt-0 px-4 py-2 text-sm font-medium rounded-xl bg-sky-600 hover:bg-sky-500 text-white flex items-center space-x-2 shadow-lg shadow-sky-600/20 disabled:opacity-50"
        >
          <Play className="w-4 h-4" />
          <span>{evaluating ? 'Analyzing Distributions...' : 'Run Ad-hoc Drift Check'}</span>
        </button>
      </div>

      {feedback && (
        <div className={`p-4 rounded-xl text-sm ${
          feedback.type === 'success' ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30' : 'bg-rose-500/15 text-rose-300 border border-rose-500/30'
        }`}>
          {feedback.message}
        </div>
      )}

      {/* Drift Status Banner */}
      <div className={`p-6 rounded-2xl border flex flex-col md:flex-row md:items-center justify-between shadow-xl ${
        isHealthy ? 'bg-emerald-950/20 border-emerald-500/30' :
        isWarning ? 'bg-amber-950/20 border-amber-500/30' :
        'bg-rose-950/20 border-rose-500/30'
      }`}>
        <div className="flex items-center space-x-4">
          <div className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${
            isHealthy ? 'bg-emerald-500/20 text-emerald-400' :
            isWarning ? 'bg-amber-500/20 text-amber-400' :
            'bg-rose-500/20 text-rose-400'
          }`}>
            {isHealthy && <ShieldCheck className="w-7 h-7" />}
            {isWarning && <AlertTriangle className="w-7 h-7" />}
            {isCritical && <AlertOctagon className="w-7 h-7" />}
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xl font-bold text-white">Status: {status}</span>
              <span className="text-xs px-2 py-0.5 rounded font-mono bg-slate-800 text-slate-300">
                Score: {report?.drift_score || 0.0}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              {report?.drifted_features_count || 0} of {report?.total_features_count || 30} features exhibit significant distribution divergence.
            </p>
          </div>
        </div>

        <div className="mt-4 md:mt-0 flex items-center space-x-4 text-xs text-slate-400">
          <div>
            <span className="font-semibold text-slate-200">Warning Threshold:</span> PSI &ge; 0.20
          </div>
          <div>
            <span className="font-semibold text-slate-200">Critical Threshold:</span> PSI &ge; 0.30
          </div>
        </div>
      </div>

      {/* Sub Tabs */}
      <div className="flex space-x-2 border-b border-slate-800">
        <button
          onClick={() => setActiveSubTab('metrics')}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-all flex items-center space-x-2 ${
            activeSubTab === 'metrics'
              ? 'border-sky-500 text-sky-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <BarChart2 className="w-4 h-4" />
          <span>Statistical Metrics (PSI & KS Test)</span>
        </button>
        <button
          onClick={() => setActiveSubTab('evidently')}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-all flex items-center space-x-2 ${
            activeSubTab === 'evidently'
              ? 'border-sky-500 text-sky-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <FileText className="w-4 h-4" />
          <span>Evidently AI Interactive Report</span>
        </button>
      </div>

      {/* Tab 1: PSI & KS Metrics */}
      {activeSubTab === 'metrics' && (
        <div className="space-y-6">
          {/* PSI Bar Chart */}
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800">
            <h3 className="text-base font-semibold text-white mb-1">Top Feature Population Stability Index (PSI)</h3>
            <p className="text-xs text-slate-400 mb-4">Features sorted by greatest population divergence vs baseline.</p>

            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={psiChartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="feature" stroke="#64748b" fontSize={11} angle={-30} textAnchor="end" />
                  <YAxis stroke="#64748b" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.75rem', color: '#f8fafc' }} 
                  />
                  <ReferenceLine y={0.20} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: 'Warning (0.20)', fill: '#f59e0b', fontSize: 10 }} />
                  <ReferenceLine y={0.30} stroke="#f43f5e" strokeDasharray="3 3" label={{ value: 'Critical (0.30)', fill: '#f43f5e', fontSize: 10 }} />
                  <Bar dataKey="psi" name="Feature PSI" radius={[4, 4, 0, 0]}>
                    {psiChartData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.psi >= 0.30 ? '#f43f5e' : entry.psi >= 0.20 ? '#f59e0b' : '#0ea5e9'} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Feature Breakdown Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
            <div className="p-4 border-b border-slate-800 bg-slate-950/40">
              <span className="text-sm font-semibold text-slate-200">Detailed Feature Drift Statistics</span>
            </div>
            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-left text-sm text-slate-300">
                <thead className="bg-slate-800/60 text-xs font-semibold uppercase text-slate-400 sticky top-0">
                  <tr>
                    <th className="py-3 px-4">Feature</th>
                    <th className="py-3 px-4">PSI Score</th>
                    <th className="py-3 px-4">PSI Status</th>
                    <th className="py-3 px-4">KS Statistic</th>
                    <th className="py-3 px-4">KS p-value</th>
                    <th className="py-3 px-4">Distribution Divergence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80 font-mono text-xs">
                  {Object.keys(featureMetrics).map((feat) => {
                    const m = featureMetrics[feat];
                    const isCrit = m.psi >= 0.30;
                    const isWarn = m.psi >= 0.20 && m.psi < 0.30;
                    return (
                      <tr key={feat} className="hover:bg-slate-800/40">
                        <td className="py-2.5 px-4 font-sans font-medium text-slate-200">{feat}</td>
                        <td className="py-2.5 px-4 font-bold">{m.psi.toFixed(4)}</td>
                        <td className="py-2.5 px-4">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            isCrit ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' :
                            isWarn ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                            'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                          }`}>
                            {m.psi_status}
                          </span>
                        </td>
                        <td className="py-2.5 px-4">{m.ks_statistic.toFixed(4)}</td>
                        <td className="py-2.5 px-4">{m.ks_pvalue.toExponential(2)}</td>
                        <td className="py-2.5 px-4">
                          {m.is_drifted ? (
                            <span className="text-rose-400 font-sans font-semibold">Shift Detected</span>
                          ) : (
                            <span className="text-slate-500 font-sans">Stable</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Evidently AI Viewer */}
      {activeSubTab === 'evidently' && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden p-4 shadow-xl">
          {report?.report_id ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Rendering Report ID: <span className="font-mono text-sky-400">{report.report_id}</span></span>
                <a 
                  href={`/reports/html/${report.report_id}`} 
                  target="_blank" 
                  rel="noreferrer"
                  className="text-sky-400 hover:text-sky-300 underline"
                >
                  Open in New Tab &rarr;
                </a>
              </div>
              <iframe
                src={`/reports/html/${report.report_id}`}
                className="w-full h-[700px] rounded-xl border border-slate-800 bg-white"
                title="Evidently Report"
              />
            </div>
          ) : (
            <div className="p-12 text-center text-slate-400">
              No Evidently HTML report generated yet. Click "Run Ad-hoc Drift Check" above.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

