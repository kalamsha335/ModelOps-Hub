import React, { useState, useEffect } from 'react';
import { 
  PlayCircle, 
  Send, 
  ShieldCheck, 
  ShieldAlert, 
  Clock, 
  Cpu, 
  RefreshCw, 
  Sliders 
} from 'lucide-react';
import { predictSingle, getPredictionLogs, submitFeedback } from '../services/api';

const NORMAL_TX = {
  Time: 406.0,
  V1: 1.234, V2: -0.123, V3: 0.456, V4: -0.789, V5: 0.234,
  V6: -0.567, V7: 0.890, V8: -0.111, V9: 0.222, V10: -0.333,
  V11: 0.444, V12: -0.555, V13: 0.666, V14: 0.777, V15: -0.888,
  V16: 0.999, V17: -0.123, V18: 0.234, V19: -0.345, V20: 0.456,
  V21: -0.567, V22: 0.678, V23: -0.789, V24: 0.890, V25: -0.901,
  V26: 0.112, V27: -0.223, V28: 0.034,
  Amount: 49.99
};

const FRAUD_TX = {
  Time: 406.0,
  V1: -2.312, V2: 1.951, V3: -1.609, V4: 3.997, V5: -0.522,
  V6: -1.426, V7: -2.537, V8: 1.391, V9: -2.770, V10: -2.772,
  V11: 3.202, V12: -2.899, V13: -0.595, V14: -4.289, V15: 0.389,
  V16: -1.140, V17: -2.830, V18: -0.016, V19: 0.416, V20: 0.126,
  V21: 0.517, V22: -0.035, V23: -0.465, V24: 0.320, V25: 0.044,
  V26: 0.177, V27: 0.261, V28: -0.143,
  Amount: 0.0
};

export default function PredictionPlayground({ modelInfo }) {
  const [formData, setFormData] = useState(NORMAL_TX);
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState([]);
  const [logsLoading, setLogsLoading] = useState(false);

  const fetchLogs = async () => {
    setLogsLoading(true);
    try {
      const res = await getPredictionLogs(10, 0);
      setLogs(res.data.records || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLogsLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    try {
      const res = await predictSingle(formData);
      setPredictionResult(res.data);
      fetchLogs();
    } catch (err) {
      alert(`Inference failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (logId, groundTruth) => {
    try {
      await submitFeedback({ prediction_log_id: logId, ground_truth: groundTruth });
      fetchLogs();
    } catch (err) {
      alert(`Feedback submission failed: ${err.message}`);
    }
  };

  const isFraud = predictionResult?.prediction === 1;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="pb-4 border-b border-slate-800">
        <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
          <PlayCircle className="w-6 h-6 text-sky-400" />
          <span>Real-Time Prediction Playground & Serving Tester</span>
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Sends transactions through the dynamic production serving model, calculates calibrated fraud probabilities, measures latency, and logs predictions.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Form & Presets */}
        <div className="lg:col-span-2 p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <h3 className="font-semibold text-white flex items-center space-x-2">
              <Sliders className="w-4 h-4 text-sky-400" />
              <span>Transaction Features</span>
            </h3>

            {/* Presets */}
            <div className="flex items-center space-x-2">
              <span className="text-xs text-slate-400">Presets:</span>
              <button
                type="button"
                onClick={() => setFormData(NORMAL_TX)}
                className="px-2.5 py-1 text-xs font-medium rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 transition-all"
              >
                Legitimate Tx
              </button>
              <button
                type="button"
                onClick={() => setFormData(FRAUD_TX)}
                className="px-2.5 py-1 text-xs font-medium rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 transition-all"
              >
                Known Fraud Tx
              </button>
            </div>
          </div>

          <form onSubmit={handlePredict} className="space-y-4">
            {/* Primary Features */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
                  Time (seconds)
                </label>
                <input
                  type="number"
                  step="any"
                  value={formData.Time}
                  onChange={(e) => setFormData({ ...formData, Time: parseFloat(e.target.value) || 0 })}
                  className="w-full px-3 py-2 rounded-xl bg-slate-800 border border-slate-700 text-white text-xs font-mono focus:outline-none focus:border-sky-500"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
                  Amount ($)
                </label>
                <input
                  type="number"
                  step="any"
                  value={formData.Amount}
                  onChange={(e) => setFormData({ ...formData, Amount: parseFloat(e.target.value) || 0 })}
                  className="w-full px-3 py-2 rounded-xl bg-slate-800 border border-slate-700 text-white text-xs font-mono focus:outline-none focus:border-sky-500"
                />
              </div>
            </div>

            {/* PCA Feature Matrix (V1 to V28) */}
            <div>
              <span className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                PCA Latent Features (V1 to V28)
              </span>
              <div className="grid grid-cols-4 sm:grid-cols-7 gap-2 max-h-48 overflow-y-auto p-2 rounded-xl bg-slate-950 border border-slate-800">
                {Array.from({ length: 28 }, (_, i) => i + 1).map((idx) => {
                  const key = `V${idx}`;
                  return (
                    <div key={key} className="space-y-0.5">
                      <label className="text-[10px] text-slate-500 font-mono">{key}</label>
                      <input
                        type="number"
                        step="any"
                        value={formData[key]}
                        onChange={(e) => setFormData({ ...formData, [key]: parseFloat(e.target.value) || 0 })}
                        className="w-full px-1.5 py-1 rounded bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-300 focus:outline-none focus:border-sky-500"
                      />
                    </div>
                  );
                })}
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 rounded-xl bg-sky-600 hover:bg-sky-500 text-white text-sm font-semibold transition-all shadow-lg shadow-sky-600/20 flex items-center justify-center space-x-2 disabled:opacity-50"
            >
              <Send className="w-4 h-4" />
              <span>{loading ? 'Evaluating...' : 'Score Transaction'}</span>
            </button>
          </form>
        </div>

        {/* Right Col: Prediction Result Card */}
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl flex flex-col justify-between">
          <div>
            <h3 className="font-semibold text-white mb-4">Inference Result</h3>

            {predictionResult ? (
              <div className="space-y-4">
                <div className={`p-5 rounded-2xl border text-center space-y-2 ${
                  isFraud ? 'bg-rose-500/15 border-rose-500/30' : 'bg-emerald-500/15 border-emerald-500/30'
                }`}>
                  <div className="flex justify-center">
                    {isFraud ? (
                      <ShieldAlert className="w-10 h-10 text-rose-400" />
                    ) : (
                      <ShieldCheck className="w-10 h-10 text-emerald-400" />
                    )}
                  </div>
                  <div className={`text-xl font-bold uppercase tracking-wider ${
                    isFraud ? 'text-rose-400' : 'text-emerald-400'
                  }`}>
                    {predictionResult.prediction_label}
                  </div>
                  <div className="text-xs text-slate-300">
                    Decision: Class {predictionResult.prediction} ({isFraud ? 'Fraud Detected' : 'Legitimate Transaction'})
                  </div>
                </div>

                {/* Probability Bar */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs font-medium">
                    <span className="text-slate-400">Fraud Probability</span>
                    <span className="font-mono text-white">
                      {(predictionResult.fraud_probability * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="w-full h-3 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 rounded-full ${
                        isFraud ? 'bg-rose-500' : 'bg-emerald-500'
                      }`}
                      style={{ width: `${Math.min(100, Math.max(2, predictionResult.fraud_probability * 100))}%` }}
                    />
                  </div>
                </div>

                {/* Serving Metadata */}
                <div className="p-3 rounded-xl bg-slate-800/40 border border-slate-700/60 text-xs space-y-2 font-mono">
                  <div className="flex justify-between text-slate-400">
                    <span>Model:</span>
                    <span className="text-slate-200">{predictionResult.model_name} (v{predictionResult.model_version})</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Latency:</span>
                    <span className="text-emerald-400 font-bold">{predictionResult.latency_ms} ms</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Log ID:</span>
                    <span className="text-sky-400">#{predictionResult.prediction_id || 'N/A'}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-slate-500 text-xs">
                Submit a transaction to inspect live classification, fraud probability, and latency.
              </div>
            )}
          </div>

          <div className="text-[11px] text-slate-500 pt-4 border-t border-slate-800">
            Powered by active model: <span className="text-slate-300 font-mono">{modelInfo?.algorithm || 'XGBClassifier'}</span>
          </div>
        </div>
      </div>

      {/* Recent Prediction Logs Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
          <span className="text-sm font-semibold text-slate-200">Recent Prediction Logs</span>
          <button onClick={fetchLogs} className="text-xs text-slate-400 hover:text-white flex items-center space-x-1">
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Refresh</span>
          </button>
        </div>

        <div className="overflow-x-auto max-h-72">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-800/60 text-slate-400 uppercase font-semibold sticky top-0">
              <tr>
                <th className="py-2.5 px-4">Log ID</th>
                <th className="py-2.5 px-4">Prediction</th>
                <th className="py-2.5 px-4">Fraud Probability</th>
                <th className="py-2.5 px-4">Latency</th>
                <th className="py-2.5 px-4">Ground Truth</th>
                <th className="py-2.5 px-4 text-right">Feedback Ingestion</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80 font-mono">
              {logs.map((log) => {
                const isFraudLog = log.prediction === 1;
                return (
                  <tr key={log.id} className="hover:bg-slate-800/40">
                    <td className="py-2.5 px-4 font-bold text-white">#{log.id}</td>
                    <td className="py-2.5 px-4">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-sans font-bold ${
                        isFraudLog ? 'bg-rose-500/20 text-rose-400' : 'bg-emerald-500/20 text-emerald-400'
                      }`}>
                        {log.prediction_label}
                      </span>
                    </td>
                    <td className="py-2.5 px-4">{(log.fraud_probability * 100).toFixed(2)}%</td>
                    <td className="py-2.5 px-4 text-slate-400">{log.latency_ms} ms</td>
                    <td className="py-2.5 px-4">
                      {log.ground_truth !== null ? (
                        <span className={`font-sans font-semibold ${log.ground_truth === 1 ? 'text-rose-400' : 'text-emerald-400'}`}>
                          {log.ground_truth === 1 ? 'Fraud (1)' : 'Legit (0)'}
                        </span>
                      ) : (
                        <span className="text-slate-500 font-sans">Pending</span>
                      )}
                    </td>
                    <td className="py-2.5 px-4 text-right space-x-1">
                      {log.ground_truth === null && (
                        <>
                          <button
                            onClick={() => handleFeedback(log.id, 0)}
                            className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 font-sans"
                            title="Mark as confirmed legitimate"
                          >
                            Legit
                          </button>
                          <button
                            onClick={() => handleFeedback(log.id, 1)}
                            className="px-2 py-0.5 rounded bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 font-sans"
                            title="Mark as confirmed fraud chargeback"
                          >
                            Fraud
                          </button>
                        </>
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
  );
}
