import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

export const getHealth = () => axios.get('/health');
export const getModelInfo = () => api.get('/model/info');

// Prediction
export const predictSingle = (data) => api.post('/predict', data);
export const predictBatch = (data) => api.post('/predict/batch', data);
export const submitFeedback = (data) => api.post('/feedback', data);
export const getPredictionLogs = (limit = 50, offset = 0) => api.get(`/logs?limit=${limit}&offset=${offset}`);

// Registry
export const getRegisteredModels = () => api.get('/registry/models');
export const getModelVersions = (modelName) => api.get(`/registry/models/${modelName}/versions`);
export const promoteModel = (data) => api.post('/registry/promote', data);
export const rollbackModel = (data) => api.post('/registry/rollback', data);

// Monitoring
export const getMonitoringStatus = () => api.get('/monitoring/status');
export const getDriftDetails = () => api.get('/monitoring/drift');
export const getDriftReports = (limit = 20) => api.get(`/monitoring/reports?limit=${limit}`);
export const getDriftAlerts = () => api.get('/monitoring/alerts');
export const resolveDriftAlert = (alertId) => api.post(`/monitoring/alerts/${alertId}/resolve`);
export const triggerDriftAnalysis = (data) => api.post('/monitoring/analyze', data);
export const getPerformanceHistory = (version = null) => 
  api.get(`/monitoring/performance${version ? `?model_version=${version}` : ''}`);

// Retraining & Training
export const triggerRetraining = (data) => api.post('/retraining/trigger', data);
export const getRetrainingHistory = (limit = 20) => api.get(`/retraining/history?limit=${limit}`);
export const getTrainingHistory = (limit = 20) => api.get(`/training/history?limit=${limit}`);

export default api;

