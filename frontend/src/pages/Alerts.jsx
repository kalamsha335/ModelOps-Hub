import React, { useState } from 'react';
import { AlertOctagon, CheckCircle2, ShieldAlert, AlertTriangle } from 'lucide-react';
import { resolveDriftAlert } from '../services/api';

export default function Alerts({ alerts, onRefresh }) {
  const [resolvingId, setResolvingId] = useState(null);

  const handleResolve = async (id) => {
    setResolvingId(id);
    try {
      await resolveDriftAlert(id);
      onRefresh();
    } catch (err) {
      alert(`Could not resolve alert: ${err.message}`);
    } finally {
      setResolvingId(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="pb-4 border-b border-slate-800">
        <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
          <AlertOctagon className="w-6 h-6 text-rose-400" />
          <span>Active Drift Alerts & Operational Remediations</span>
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Surfaces violated feature distribution thresholds (PSI &ge; 0.20/0.30 or KS p-value &lt; 0.05).
        </p>
      </div>

      {alerts?.length === 0 ? (
        <div className="p-12 text-center rounded-2xl bg-slate-900 border border-slate-800 space-y-3">
          <CheckCircle2 className="w-12 h-12 text-emerald-400 mx-auto" />
          <h3 className="text-lg font-semibold text-white">No Active Drift Alerts</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            All continuous feature distributions are stable within operational tolerances.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {alerts?.map((alert) => {
            const isCrit = alert.severity === 'CRITICAL';
            return (
              <div 
                key={alert.id}
                className={`p-5 rounded-2xl border flex flex-col sm:flex-row sm:items-center justify-between gap-4 transition-all ${
                  isCrit ? 'bg-rose-950/20 border-rose-500/30' : 'bg-amber-950/20 border-amber-500/30'
                }`}
              >
                <div className="flex items-start space-x-3">
                  <div className={`p-2 rounded-xl mt-0.5 ${
                    isCrit ? 'bg-rose-500/20 text-rose-400' : 'bg-amber-500/20 text-amber-400'
                  }`}>
                    {isCrit ? <ShieldAlert className="w-5 h-5" /> : <AlertTriangle className="w-5 h-5" />}
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold text-white text-base">{alert.affected_feature}</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        isCrit ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                      }`}>
                        {alert.severity}
                      </span>
                      <span className="text-xs font-mono text-slate-400">
                        {alert.drift_metric}: {alert.metric_value.toFixed(4)} (Threshold: {alert.threshold_value})
                      </span>
                    </div>
                    <p className="text-xs text-slate-300">
                      <strong>Recommended Action:</strong> {alert.recommended_action}
                    </p>
                    <div className="text-[11px] text-slate-500">
                      Detected: {alert.detected_at ? new Date(alert.detected_at).toLocaleString() : 'Recent'}
                    </div>
                  </div>
                </div>

                <button
                  disabled={resolvingId === alert.id}
                  onClick={() => handleResolve(alert.id)}
                  className="px-3.5 py-1.5 text-xs font-medium rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-all shrink-0 self-end sm:self-center"
                >
                  {resolvingId === alert.id ? 'Resolving...' : 'Acknowledge & Resolve'}
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

