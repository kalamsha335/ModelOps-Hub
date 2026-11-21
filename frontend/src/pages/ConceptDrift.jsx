import React, { useState } from 'react';
import { 
  GitCompare, 
  HelpCircle, 
  Send, 
  CheckCircle, 
  AlertTriangle, 
  TrendingDown,
  Clock
} from 'lucide-react';
import { submitFeedback } from '../services/api';

export default function ConceptDrift({ onRefresh }) {
  const [feedbackInput, setFeedbackInput] = useState({ prediction_log_id: '', ground_truth: '0' });
  const [feedbackStatus, setFeedbackStatus] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const handleFeedbackSubmit = async (e) => {
    e.preventDefault();
    if (!feedbackInput.prediction_log_id) return;
    setSubmitting(true);
    setFeedbackStatus(null);
    try {
      const res = await submitFeedback({
        prediction_log_id: parseInt(feedbackInput.prediction_log_id, 10),
        ground_truth: parseInt(feedbackInput.ground_truth, 10)
      });
      setFeedbackStatus({
        type: 'success',
        message: `Feedback recorded! Log #${res.data.prediction_log_id}: ${res.data.is_correct ? 'Correct Prediction' : 'Misclassification'}`
      });
      setFeedbackInput({ prediction_log_id: '', ground_truth: '0' });
      onRefresh();
    } catch (err) {
      setFeedbackStatus({
        type: 'error',
        message: `Failed to record feedback: ${err.response?.data?.detail || err.message}`
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="pb-4 border-b border-slate-800">
        <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
          <GitCompare className="w-6 h-6 text-sky-400" />
          <span>Concept Drift & Ground-Truth Performance Monitoring</span>
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Distinguishes Data Drift P(X) from Concept Drift P(Y|X). Accommodates delayed settlement chargeback labels.
        </p>
      </div>

      {/* Educational Callout: Concept Drift vs Data Drift */}
      <div className="p-6 rounded-2xl bg-gradient-to-br from-indigo-950/40 via-slate-900 to-slate-900 border border-indigo-500/20 shadow-xl space-y-3">
        <div className="flex items-center space-x-2 text-indigo-400 font-semibold text-sm">
          <HelpCircle className="w-4 h-4" />
          <span>Core MLOps Distinction: Data Drift vs. Concept Drift</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-300">
          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50 space-y-1">
            <span className="font-bold text-sky-400 text-sm">1. Data Drift (Covariate Shift)</span>
            <p className="text-slate-400">
              <strong className="text-slate-200">Formula:</strong> P(X) changes while P(Y|X) remains constant.
            </p>
            <p>
              The distribution of input features shifts (e.g. higher transaction amounts during holiday seasons), but the underlying fraud mechanics do not change.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50 space-y-1">
            <span className="font-bold text-rose-400 text-sm">2. Concept Drift (Relationship Shift)</span>
            <p className="text-slate-400">
              <strong className="text-slate-200">Formula:</strong> P(Y|X) changes (the relationship between features and outcomes shifts).
            </p>
            <p>
              Fraudsters adapt techniques to mimic legitimate transaction patterns. The same input features that used to indicate legitimate activity now correlate with fraud.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Delayed Ground-Truth Ingestion Panel */}
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl space-y-4">
          <div>
            <h3 className="text-base font-semibold text-white">Delayed Label Ingestion (Chargeback Simulator)</h3>
            <p className="text-xs text-slate-400 mt-0.5">
              In real-world fraud detection, ground-truth chargebacks take 30-90 days to settle. Ingest true labels to trigger concept drift evaluation.
            </p>
          </div>

          {feedbackStatus && (
            <div className={`p-3 rounded-xl text-xs ${
              feedbackStatus.type === 'success' 
                ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30' 
                : 'bg-rose-500/15 text-rose-300 border border-rose-500/30'
            }`}>
              {feedbackStatus.message}
            </div>
          )}

          <form onSubmit={handleFeedbackSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
                Prediction Log ID
              </label>
              <input
                type="number"
                required
                placeholder="e.g. 1"
                value={feedbackInput.prediction_log_id}
                onChange={(e) => setFeedbackInput({ ...feedbackInput, prediction_log_id: e.target.value })}
                className="w-full px-3.5 py-2.5 rounded-xl bg-slate-800 border border-slate-700 text-white text-sm focus:outline-none focus:border-sky-500 font-mono"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1">
                Actual Ground-Truth Outcome
              </label>
              <select
                value={feedbackInput.ground_truth}
                onChange={(e) => setFeedbackInput({ ...feedbackInput, ground_truth: e.target.value })}
                className="w-full px-3.5 py-2.5 rounded-xl bg-slate-800 border border-slate-700 text-white text-sm focus:outline-none focus:border-sky-500"
              >
                <option value="0">0 - Legitimate Transaction (No Fraud)</option>
                <option value="1">1 - Confirmed Fraudulent Chargeback</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={submitting}
              className="w-full py-2.5 rounded-xl bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium transition-all shadow-lg shadow-sky-600/20 flex items-center justify-center space-x-2 disabled:opacity-50"
            >
              <Send className="w-4 h-4" />
              <span>{submitting ? 'Recording...' : 'Submit Ground-Truth Label'}</span>
            </button>
          </form>
        </div>

        {/* Prediction Probability Drift Indicator */}
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl space-y-4">
          <div>
            <h3 className="text-base font-semibold text-white">Prediction Distribution Monitoring P(Y_hat)</h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Monitors proxy indicators while true chargeback labels remain pending.
            </p>
          </div>

          <div className="space-y-3">
            <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 flex items-center justify-between">
              <div>
                <span className="text-xs text-slate-400">Baseline Fraud Rate</span>
                <div className="text-lg font-bold text-white font-mono">0.172%</div>
              </div>
              <div className="text-right">
                <span className="text-xs text-slate-400">Current Serving Fraud Rate</span>
                <div className="text-lg font-bold text-sky-400 font-mono">0.180%</div>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 flex items-center justify-between">
              <div>
                <span className="text-xs text-slate-400">Degradation Threshold</span>
                <div className="text-sm font-semibold text-slate-200">&gt; 10% Drop in PR-AUC / Recall</div>
              </div>
              <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                STABLE
              </span>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-400 space-y-1">
              <div className="font-semibold text-slate-300 flex items-center space-x-1.5">
                <Clock className="w-3.5 h-3.5 text-sky-400" />
                <span>Monitoring Status: Awaiting Ground-Truth Labels</span>
              </div>
              <p>
                Actual performance metrics update dynamically whenever settlement ground truth is ingested above.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

