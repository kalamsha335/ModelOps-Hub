import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Overview from './pages/Overview';
import ModelRegistry from './pages/ModelRegistry';
import DriftMonitoring from './pages/DriftMonitoring';
import ConceptDrift from './pages/ConceptDrift';
import Alerts from './pages/Alerts';
import Retraining from './pages/Retraining';
import PredictionPlayground from './pages/PredictionPlayground';
import { 
  getMonitoringStatus, 
  getModelInfo, 
  getDriftDetails, 
  getDriftAlerts,
  getRegisteredModels,
  getModelVersions,
  getRetrainingHistory,
  triggerDriftAnalysis,
  triggerRetraining
} from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [systemHealth, setSystemHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [driftReport, setDriftReport] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [models, setModels] = useState([]);
  const [versions, setVersions] = useState([]);
  const [retrainingHistory, setRetrainingHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  const refreshAll = async () => {
    try {
      const [healthRes, infoRes, driftRes, alertsRes, modelsRes, retrainRes] = await Promise.allSettled([
        getMonitoringStatus(),
        getModelInfo(),
        getDriftDetails(),
        getDriftAlerts(),
        getRegisteredModels(),
        getRetrainingHistory(20)
      ]);

      if (healthRes.status === 'fulfilled') setSystemHealth(healthRes.value.data);
      if (infoRes.status === 'fulfilled') setModelInfo(infoRes.value.data);
      if (driftRes.status === 'fulfilled') setDriftReport(driftRes.value.data);
      if (alertsRes.status === 'fulfilled') setAlerts(alertsRes.value.data || []);
      if (modelsRes.status === 'fulfilled') setModels(modelsRes.value.data || []);
      if (retrainRes.status === 'fulfilled') setRetrainingHistory(retrainRes.value.data || []);

      // Fetch versions for default model
      const vRes = await getModelVersions('fraud-detection-model').catch(() => null);
      if (vRes?.data) setVersions(vRes.data);
    } catch (err) {
      console.error('Error fetching dashboard state:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshAll();
    const interval = setInterval(refreshAll, 15000); // 15s auto-polling
    return () => clearInterval(interval);
  }, []);

  const handleQuickDriftCheck = async () => {
    try {
      await triggerDriftAnalysis({ sample_size: 2000, generate_evidently_html: true });
      refreshAll();
      setActiveTab('drift');
    } catch (err) {
      alert(`Drift evaluation failed: ${err.message}`);
    }
  };

  const handleQuickRetrain = async () => {
    try {
      await triggerRetraining({ trigger_reason: 'MANUAL', force: true, quick_run: true, optuna_trials: 2 });
      refreshAll();
      setActiveTab('retraining');
    } catch (err) {
      alert(`Retraining failed: ${err.message}`);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 overflow-hidden">
      <Navbar 
        systemHealth={systemHealth} 
        modelInfo={modelInfo} 
        onRefresh={refreshAll} 
      />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar 
          activeTab={activeTab} 
          setActiveTab={setActiveTab} 
          alertCount={alerts?.length || 0}
        />

        <main className="flex-1 overflow-y-auto p-6 md:p-8 bg-slate-950">
          <div className="max-w-7xl mx-auto">
            {activeTab === 'overview' && (
              <Overview 
                systemHealth={systemHealth}
                modelInfo={modelInfo}
                driftReport={driftReport}
                activeAlerts={alerts}
                models={models}
                onNavigate={setActiveTab}
                onTriggerDriftCheck={handleQuickDriftCheck}
                onTriggerRetraining={handleQuickRetrain}
              />
            )}

            {activeTab === 'registry' && (
              <ModelRegistry 
                versions={versions} 
                onRefresh={refreshAll} 
              />
            )}

            {activeTab === 'drift' && (
              <DriftMonitoring 
                driftReport={driftReport} 
                onRefresh={refreshAll} 
              />
            )}

            {activeTab === 'concept' && (
              <ConceptDrift 
                onRefresh={refreshAll} 
              />
            )}

            {activeTab === 'alerts' && (
              <Alerts 
                alerts={alerts} 
                onRefresh={refreshAll} 
              />
            )}

            {activeTab === 'retraining' && (
              <Retraining 
                history={retrainingHistory} 
                onRefresh={refreshAll} 
              />
            )}

            {activeTab === 'playground' && (
              <PredictionPlayground 
                modelInfo={modelInfo} 
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
