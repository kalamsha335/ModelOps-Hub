import React, { useState } from 'react';
import { 
  RefreshCcw, 
  Zap, 
  CheckCircle, 
  AlertTriangle, 
  XCircle, 
  ShieldCheck, 
  Cpu, 
  FileText 
} from 'lucide-react';
import { triggerRetraining } from '../services/api';

export default function Retraining({ history, onRefresh }) {
  const [triggering, setTriggering] = useState(false);
  const [selectedLogs, setSelectedLogs] = useState(null);
  const [feedback, setFeedback] = useState(null);

  const handleManualTrigger = async (quick = true) => {
    setTriggering(true);
    setFeedback(null);
    try {
      const res = await triggerRetraining({
        trigger_reason: 'MANUAL',
        force: true,
        quick_run: quick,
        optuna_trials: quick ? 2 : 5
      });
      setFeedback({
        type: 'success',
        message: `Retraining finished! Decision: ${res.data.promotion_decision || 'PROMOTED'}, Candidate Score: ${res.data.candidate_score || '-'}`
      });
      onRefresh();
    } catch (err) {
      setFeedback({
        type: 'error',
        message: `Retraining failed: ${err.response?.data?.detail || err.message}`
      });
    } finally {
      setTriggering(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
            <RefreshCcw className="w-6 h-6 text-sky-400" />
            <span>Automated Retraining & Guardrail Governance</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Prefect workflow orchestration with Champion vs. Challenger guardrails to prevent replacing healthy models with inferior versions.
          </p>
        </div>

        <div className="mt-4 sm:mt-0 flex space-x-3">
          <button
            disabled={triggering}
            onClick={() => handleManualTrigger(true)}
            className="px-4 py-2 text-sm font-medium rounded-xl bg-sky-600 hover:bg-sky-500 text-white flex items-center space-x-2 shadow-lg shadow-sky-600/20 disabled:opacity-50"
          >
            <Zap className="w-4 h-4" />
            <span>{triggering ? 'Retraining Models...' : 'Trigger Fast Retraining'}</span>
          </button>
        </div>
      </div>

      {feedback && (
        <div className={`p-4 rounded-xl text-sm ${
          feedback.type === 'success' ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30' : 'bg-rose-500/15 text-rose-300 border border-rose-500/30'
        }`}>
          {feedback.message}
        </div>
      )}

      {/* Retraining Safeguards Info Card */}
      <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl space-y-3">
        <div className="flex items-center space-x-2 text-sky-400 font-semibold text-sm">
          <ShieldCheck className="w-4 h-4" />
          <span>Active Production Safeguards & Guardrail Rules</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-slate-300">
          <div className="p-3 rounded-xl bg-slate-800/40 border border-slate-700/60">
            <div className="font-semibold text-white mb-1">1. Concurrency Protection</div>
            <p className="text-slate-400">Prevents multiple duplicate retraining workflows from running simultaneously.</p>
          </div>
          <div className="p-3 rounded-xl bg-slate-800/40 border border-slate-700/60">
            <div className="font-semibold text-white mb-1">2. Quantitative Gate</div>
            <p className="text-slate-400">Rejects candidate models failing PR-AUC &ge; 0.80 or Recall &ge; 0.75.</p>
          </div>
          <div className="p-3 rounded-xl bg-slate-800/40 border border-slate-700/60">
            <div className="font-semibold text-white mb-1">3. Champion vs. Challenger</div>
            <p className="text-slate-400">Only promotes challenger if composite score strictly equals or exceeds current production.</p>
          </div>
        </div>
      </div>

      {/* Retraining History Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
          <span className="text-sm font-semibold text-slate-200">Execution History ({history?.length || 0})</span>
          <span className="text-xs text-slate-400">All runs logged with full audit trails</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-slate-800/60 text-xs font-semibold uppercase text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Job ID</th>
                <th className="py-3 px-4">Trigger Reason</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4">Previous &rarr; New</th>
                <th className="py-3 px-4">Promotion Decision</th>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4 text-right">Logs</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80 text-xs">
              {history?.map((job) => {
                const isPromoted = job.promotion_decision === 'PROMOTED';
                const isRejected = job.promotion_decision === 'REJECTED_INFERIOR';
                const isCompleted = job.status === 'COMPLETED';

                return (
                  <tr key={job.id} className="hover:bg-slate-800/40">
                    <td className="py-3 px-4 font-mono font-medium text-white">{job.job_id}</td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded font-mono text-[11px] bg-slate-800 text-slate-300 border border-slate-700">
                        {job.trigger_reason}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        isCompleted ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
                        job.status === 'SKIPPED_INFERIOR' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                        'bg-slate-800 text-slate-400'
                      }`}>
                        {job.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono">
                      v{job.previous_model_version || '1'} &rarr; v{job.new_model_version || '?'}
                    </td>
                    <td className="py-3 px-4">
                      {isPromoted && (
                        <span className="flex items-center space-x-1 text-emerald-400 font-semibold">
                          <CheckCircle className="w-3.5 h-3.5" />
                          <span>PROMOTED</span>
                        </span>
                      )}
                      {isRejected && (
                        <span className="flex items-center space-x-1 text-amber-400 font-semibold" title="Candidate was strictly inferior to production model">
                          <AlertTriangle className="w-3.5 h-3.5" />
                          <span>REJECTED INFERIOR</span>
                        </span>
                      )}
                      {!isPromoted && !isRejected && (
                        <span className="text-slate-400">{job.promotion_decision}</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-slate-400">
                      {job.started_at ? new Date(job.started_at).toLocaleTimeString() : '-'}
                    </td>
                    <td className="py-3 px-4 text-right">
                      {job.logs && (
                        <button
                          onClick={() => setSelectedLogs(job)}
                          className="px-2.5 py-1 text-xs rounded bg-slate-800 hover:bg-slate-700 text-sky-400"
                        >
                          View Log
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Log Modal */}
      {selectedLogs && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-xl w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-semibold text-white">Retraining Execution Log ({selectedLogs.job_id})</h3>
              <button 
                onClick={() => setSelectedLogs(null)}
                className="text-slate-400 hover:text-white"
              >
                &times;
              </button>
            </div>
            <div className="p-4 rounded-xl bg-slate-950 font-mono text-xs text-slate-300 max-h-64 overflow-y-auto whitespace-pre-wrap border border-slate-800">
              {selectedLogs.logs}
            </div>
            <div className="flex justify-end">
              <button
                onClick={() => setSelectedLogs(null)}
                className="px-4 py-2 text-xs font-medium rounded-xl bg-slate-800 hover:bg-slate-700 text-white"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
