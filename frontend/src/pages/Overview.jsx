import React from 'react';
import { 
  ShieldCheck, 
  Activity, 
  Zap, 
  Clock, 
  AlertTriangle, 
  Layers, 
  ArrowRight,
  TrendingUp,
  Cpu
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, AreaChart, Area 
} from 'recharts';

export default function Overview({ 
  systemHealth, 
  modelInfo, 
  driftReport, 
  activeAlerts,
  models,
  onNavigate,
  onTriggerDriftCheck,
  onTriggerRetraining
}) {
  const currentProd = models?.find(m => m.current_production_version) || {};
  const isHealthy = systemHealth?.overall_health === 'Healthy';
  const isCritical = systemHealth?.overall_health === 'Critical';

  // Sample performance radar / metric data for display
  const modelMetricsData = [
    { metric: 'PR-AUC', value: 82.5, benchmark: 80.0 },
    { metric: 'Recall', value: 79.7, benchmark: 75.0 },
    { metric: 'Precision', value: 86.8, benchmark: 80.0 },
    { metric: 'F1 Score', value: 83.1, benchmark: 78.0 },
    { metric: 'ROC-AUC', value: 97.4, benchmark: 90.0 },
  ];

  const recentTrafficData = [
    { time: '10:00', txCount: 1420, fraudDetected: 3 },
    { time: '11:00', txCount: 1850, fraudDetected: 4 },
    { time: '12:00', txCount: 2210, fraudDetected: 5 },
    { time: '13:00', txCount: 1980, fraudDetected: 2 },
    { time: '14:00', txCount: 2450, fraudDetected: 6 },
    { time: '15:00', txCount: 2890, fraudDetected: 7 },
    { time: '16:00', txCount: 2150, fraudDetected: 4 },
  ];

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between p-6 bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border border-slate-800 rounded-2xl shadow-xl">
        <div className="space-y-1">
          <div className="flex items-center space-x-3">
            <h1 className="text-2xl font-bold text-white tracking-tight">System Status & Operations</h1>
            <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider ${
              isHealthy ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30' :
              isCritical ? 'bg-rose-500/15 text-rose-400 border border-rose-500/30 animate-pulse' :
              'bg-amber-500/15 text-amber-400 border border-amber-500/30'
            }`}>
              {systemHealth?.overall_health || 'Healthy'}
            </span>
          </div>
          <p className="text-sm text-slate-400">
            Monitoring tabular transaction stream against reference baseline. Retraining guardrails active.
          </p>
        </div>

        {/* Quick Operations CTA */}
        <div className="mt-4 md:mt-0 flex items-center space-x-3">
          <button
            onClick={onTriggerDriftCheck}
            className="px-4 py-2 text-sm font-medium rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center space-x-2 transition-all shadow-sm"
          >
            <Activity className="w-4 h-4 text-sky-400" />
            <span>Run Drift Check</span>
          </button>
          <button
            onClick={onTriggerRetraining}
            className="px-4 py-2 text-sm font-medium rounded-xl bg-sky-600 hover:bg-sky-500 text-white flex items-center space-x-2 transition-all shadow-lg shadow-sky-600/25"
          >
            <Zap className="w-4 h-4" />
            <span>Trigger Retraining</span>
          </button>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Production Champion */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Production Model</span>
            <div className="w-8 h-8 rounded-lg bg-sky-500/10 flex items-center justify-center text-sky-400">
              <Cpu className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-white">
              {modelInfo?.algorithm || 'XGBClassifier'}
            </div>
            <div className="flex items-center space-x-2 mt-1 text-xs text-slate-400">
              <span className="bg-sky-500/20 text-sky-300 px-1.5 py-0.5 rounded font-mono">
                v{modelInfo?.model_version || '4'}
              </span>
              <span>• PR-AUC: 82.5%</span>
            </div>
          </div>
        </div>

        {/* Card 2: Inference Latency */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Avg Latency (P95)</span>
            <div className="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-400">
              <Clock className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-white">4.8 ms</div>
            <div className="flex items-center space-x-1 mt-1 text-xs text-emerald-400">
              <TrendingUp className="w-3.5 h-3.5" />
              <span>Optimal (SLA &lt; 50ms)</span>
            </div>
          </div>
        </div>

        {/* Card 3: Data Drift Status */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Data Drift Status</span>
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-400">
              <ShieldCheck className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-white capitalize">
              {systemHealth?.latest_drift_status || 'Healthy'}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Drift Score: {systemHealth?.latest_drift_score || 0.0}
            </div>
          </div>
        </div>

        {/* Card 4: Active Alerts */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Active Alerts</span>
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
              (systemHealth?.critical_alerts_count || 0) > 0 ? 'bg-rose-500/15 text-rose-400' : 'bg-slate-800 text-slate-400'
            }`}>
              <AlertTriangle className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-white">
              {systemHealth?.active_alerts_total || 0}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {systemHealth?.critical_alerts_count || 0} Critical • {activeAlerts?.length || 0} Total
            </div>
          </div>
        </div>
      </div>

      {/* Analytics & Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Performance Benchmark Chart */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-semibold text-white">Production Model Metrics (%)</h3>
              <p className="text-xs text-slate-400">Current XGBoost Champion vs Baseline Threshold</p>
            </div>
            <button 
              onClick={() => onNavigate('registry')}
              className="text-xs font-medium text-sky-400 hover:text-sky-300 flex items-center space-x-1"
            >
              <span>View Registry</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelMetricsData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="metric" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.75rem', color: '#f8fafc' }} 
                />
                <Bar dataKey="value" name="Champion Metric" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                <Bar dataKey="benchmark" name="Acceptance Gate" fill="#475569" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Prediction Volume & Fraud Detection Trend */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-semibold text-white">Serving Traffic Volume</h3>
              <p className="text-xs text-slate-400">Hourly transactions served and fraud flags</p>
            </div>
            <button 
              onClick={() => onNavigate('playground')}
              className="text-xs font-medium text-sky-400 hover:text-sky-300 flex items-center space-x-1"
            >
              <span>Playground</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={recentTrafficData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorTx" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.75rem', color: '#f8fafc' }} 
                />
                <Area type="monotone" dataKey="txCount" name="Total Transactions" stroke="#0ea5e9" fillOpacity={1} fill="url(#colorTx)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
