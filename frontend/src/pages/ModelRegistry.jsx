import React, { useState } from 'react';
import { 
  Layers, 
  CheckCircle, 
  AlertCircle, 
  ArrowUpRight, 
  RotateCcw, 
  ShieldAlert, 
  Award,
  Sparkles,
  ExternalLink
} from 'lucide-react';
import { promoteModel, rollbackModel } from '../services/api';

export default function ModelRegistry({ versions, onRefresh }) {
  const [selectedVersion, setSelectedVersion] = useState(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [feedbackMsg, setFeedbackMsg] = useState(null);

  const handlePromote = async (version, stage) => {
    setActionLoading(true);
    setFeedbackMsg(null);
    try {
      const res = await promoteModel({
        model_name: 'fraud-detection-model',
        version: version,
        target_stage: stage,
        reason: `Promoted to ${stage} via ModelOps Hub dashboard`,
        enforce_validation: true
      });
      setFeedbackMsg({ type: 'success', message: `Model v${version} successfully promoted to ${stage}!` });
      onRefresh();
    } catch (err) {
      const errDetail = err.response?.data?.detail || err.message;
      setFeedbackMsg({ type: 'error', message: `Promotion rejected: ${errDetail}` });
    } finally {
      setActionLoading(false);
    }
  };

  const handleRollback = async (targetVersion) => {
    if (!window.confirm(`Are you sure you want to rollback production to version ${targetVersion}?`)) return;
    setActionLoading(true);
    setFeedbackMsg(null);
    try {
      const res = await rollbackModel({
        model_name: 'fraud-detection-model',
        target_version: targetVersion,
        reason: `Manual rollback to v${targetVersion} via dashboard`
      });
      setFeedbackMsg({ type: 'success', message: `Rollback completed! Active production is now v${targetVersion}.` });
      onRefresh();
    } catch (err) {
      const errDetail = err.response?.data?.detail || err.message;
      setFeedbackMsg({ type: 'error', message: `Rollback failed: ${errDetail}` });
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
            <Layers className="w-6 h-6 text-sky-400" />
            <span>MLflow Model Registry & Stage Governance</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Registered Model: <span className="text-sky-400 font-mono">fraud-detection-model</span> • Enforcing strict PR-AUC &ge; 0.80 and Recall &ge; 0.75 production criteria.
          </p>
        </div>
      </div>

      {/* Alert / Feedback message */}
      {feedbackMsg && (
        <div className={`p-4 rounded-xl flex items-center space-x-3 text-sm ${
          feedbackMsg.type === 'success' 
            ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30' 
            : 'bg-rose-500/15 text-rose-300 border border-rose-500/30'
        }`}>
          {feedbackMsg.type === 'success' ? (
            <CheckCircle className="w-5 h-5 shrink-0 text-emerald-400" />
          ) : (
            <ShieldAlert className="w-5 h-5 shrink-0 text-rose-400" />
          )}
          <span>{feedbackMsg.message}</span>
        </div>
      )}

      {/* Versions Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
          <span className="text-sm font-semibold text-slate-200">Registered Model Versions ({versions?.length || 0})</span>
          <span className="text-xs text-slate-400">Stages: Development &rarr; Staging &rarr; Production</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-slate-800/60 text-xs font-semibold uppercase text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-3.5 px-4">Version</th>
                <th className="py-3.5 px-4">Algorithm</th>
                <th className="py-3.5 px-4">Lifecycle Stage</th>
                <th className="py-3.5 px-4">Validation Gate</th>
                <th className="py-3.5 px-4">PR-AUC</th>
                <th className="py-3.5 px-4">Recall</th>
                <th className="py-3.5 px-4">F1 Score</th>
                <th className="py-3.5 px-4 text-right">Governance Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {versions?.map((ver) => {
                const isProd = ver.is_current_production;
                const isStaging = ver.stage === 'Staging';
                const isDev = ver.stage === 'Development';
                const isValid = ver.validation_status === 'PASSED';
                const isFailed = ver.validation_status === 'FAILED';

                const prAuc = ver.metrics?.pr_auc ? (ver.metrics.pr_auc * 100).toFixed(1) : '-';
                const recall = ver.metrics?.recall ? (ver.metrics.recall * 100).toFixed(1) : '-';
                const f1 = ver.metrics?.f1_score ? (ver.metrics.f1_score * 100).toFixed(1) : '-';

                return (
                  <tr key={ver.version} className={`hover:bg-slate-800/40 transition-colors ${isProd ? 'bg-emerald-500/5' : ''}`}>
                    {/* Version */}
                    <td className="py-3.5 px-4 font-mono font-medium text-white flex items-center space-x-2">
                      <span>v{ver.version}</span>
                      {isProd && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center space-x-1">
                          <Award className="w-3 h-3 inline" />
                          <span>CHAMPION</span>
                        </span>
                      )}
                    </td>

                    {/* Algorithm */}
                    <td className="py-3.5 px-4 font-medium text-slate-200">{ver.algorithm}</td>

                    {/* Stage */}
                    <td className="py-3.5 px-4">
                      {isProd && (
                        <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                          Production
                        </span>
                      )}
                      {isStaging && (
                        <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30">
                          Staging
                        </span>
                      )}
                      {isDev && (
                        <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-sky-500/20 text-sky-400 border border-sky-500/30">
                          Development
                        </span>
                      )}
                      {!isProd && !isStaging && !isDev && (
                        <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-800 text-slate-400">
                          {ver.stage}
                        </span>
                      )}
                    </td>

                    {/* Validation Gate */}
                    <td className="py-3.5 px-4">
                      {isValid && (
                        <span className="flex items-center space-x-1 text-xs text-emerald-400">
                          <CheckCircle className="w-3.5 h-3.5" />
                          <span>PASSED</span>
                        </span>
                      )}
                      {isFailed && (
                        <span className="flex items-center space-x-1 text-xs text-rose-400" title="Did not meet PR-AUC >= 0.80 or Recall >= 0.75">
                          <AlertCircle className="w-3.5 h-3.5" />
                          <span>REJECTED</span>
                        </span>
                      )}
                      {!isValid && !isFailed && (
                        <span className="text-xs text-slate-400 font-mono">{ver.validation_status}</span>
                      )}
                    </td>

                    {/* Metrics */}
                    <td className="py-3.5 px-4 font-mono">{prAuc}%</td>
                    <td className="py-3.5 px-4 font-mono">{recall}%</td>
                    <td className="py-3.5 px-4 font-mono">{f1}%</td>

                    {/* Actions */}
                    <td className="py-3.5 px-4 text-right space-x-2">
                      {!isProd && (
                        <>
                          {isDev && (
                            <button
                              disabled={actionLoading}
                              onClick={() => handlePromote(ver.version, 'Staging')}
                              className="px-2.5 py-1 text-xs font-medium rounded-lg bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30 transition-all"
                            >
                              To Staging
                            </button>
                          )}
                          <button
                            disabled={actionLoading}
                            onClick={() => handlePromote(ver.version, 'Production')}
                            className="px-2.5 py-1 text-xs font-medium rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/30 transition-all"
                          >
                            Promote to Prod
                          </button>
                        </>
                      )}

                      {!isProd && (isValid || ver.validation_status === 'OVERRIDDEN') && (
                        <button
                          disabled={actionLoading}
                          onClick={() => handleRollback(ver.version)}
                          className="px-2.5 py-1 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-all"
                          title="Rollback active production pointer to this previous validated model"
                        >
                          <RotateCcw className="w-3.5 h-3.5 inline mr-1" />
                          Rollback Here
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
    </div>
  );
}
