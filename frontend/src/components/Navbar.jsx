import React from 'react';
import { Activity, ShieldCheck, Cpu, AlertTriangle, CheckCircle, RefreshCw } from 'lucide-react';

export default function Navbar({ systemHealth, modelInfo, onRefresh }) {
  const isHealthy = systemHealth?.overall_health === 'Healthy';
  const isWarning = systemHealth?.overall_health === 'Warning';
  const isCritical = systemHealth?.overall_health === 'Critical';

  return (
    <header className="h-16 bg-slate-900 border-b border-slate-800 px-6 flex items-center justify-between sticky top-0 z-30">
      {/* Brand */}
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
          <Activity className="w-6 h-6 text-white" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-bold text-lg text-white tracking-tight">ModelOps Hub</span>
            <span className="text-xs bg-sky-500/10 text-sky-400 font-semibold px-2 py-0.5 rounded-full border border-sky-500/20">
              v1.0.0
            </span>
          </div>
          <p className="text-xs text-slate-400">Automated Tabular ML Lifecycle Platform</p>
        </div>
      </div>

      {/* Status Bar */}
      <div className="flex items-center space-x-4">
        {/* Active Production Model */}
        <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs">
          <Cpu className="w-4 h-4 text-sky-400" />
          <span className="text-slate-400">Production:</span>
          <span className="font-semibold text-slate-200">
            {modelInfo?.algorithm || 'XGBClassifier'} (v{modelInfo?.model_version || '1'})
          </span>
        </div>

        {/* System Health Badge */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg border text-xs font-medium">
          {isHealthy && (
            <span className="flex items-center space-x-1.5 text-emerald-400 bg-emerald-500/10 border-emerald-500/20">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span>System Healthy</span>
            </span>
          )}
          {isWarning && (
            <span className="flex items-center space-x-1.5 text-amber-400 bg-amber-500/10 border-amber-500/20">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Drift Warning</span>
            </span>
          )}
          {isCritical && (
            <span className="flex items-center space-x-1.5 text-rose-400 bg-rose-500/10 border-rose-500/20 animate-pulse">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Critical Drift</span>
            </span>
          )}
        </div>

        {/* Global Refresh */}
        <button
          onClick={onRefresh}
          className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
          title="Refresh Platform State"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}
