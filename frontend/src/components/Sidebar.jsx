import React from 'react';
import { 
  LayoutDashboard, 
  Layers, 
  LineChart, 
  GitCompare, 
  AlertOctagon, 
  RefreshCcw, 
  PlayCircle 
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'registry', label: 'Model Registry', icon: Layers },
  { id: 'drift', label: 'Data Drift Monitor', icon: LineChart },
  { id: 'concept', label: 'Performance & Concept', icon: GitCompare },
  { id: 'alerts', label: 'Active Alerts', icon: AlertOctagon },
  { id: 'retraining', label: 'Retraining Jobs', icon: RefreshCcw },
  { id: 'playground', label: 'Prediction Playground', icon: PlayCircle },
];

export default function Sidebar({ activeTab, setActiveTab, alertCount = 0 }) {
  return (
    <aside className="w-64 bg-slate-900/90 border-r border-slate-800 flex flex-col justify-between shrink-0">
      <div className="p-4 space-y-1">
        <div className="px-3 py-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Management Console
        </div>
        <nav className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30 shadow-sm shadow-sky-500/10'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <Icon className={`w-4 h-4 ${isActive ? 'text-sky-400' : 'text-slate-500'}`} />
                  <span>{item.label}</span>
                </div>
                {item.id === 'alerts' && alertCount > 0 && (
                  <span className="px-2 py-0.5 text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30 rounded-full">
                    {alertCount}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Dataset & Architecture Info Footer */}
      <div className="p-4 border-t border-slate-800 text-xs text-slate-500 space-y-1 bg-slate-950/40">
        <div className="font-medium text-slate-400">ULB Credit Card Fraud</div>
        <div>284,807 tx • 0.172% Fraud</div>
        <div className="text-[11px] text-slate-500 pt-1">
          Evidently • MLflow • Prefect • XGBoost
        </div>
      </div>
    </aside>
  );
}

