// File: frontend/src/services/api.js

import axios from 'axios';
import toast from 'react-hot-toast';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds

// Create axios instance with default config
const apiClient = axios.create({
    baseURL: API_BASE_URL,
    timeout: API_TIMEOUT,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor for logging and auth
apiClient.interceptors.request.use(
    (config) => {
        // Add timestamp to requests
        config.metadata = { startTime: new Date() };

        // Log requests in development
        if (process.env.NODE_ENV === 'development') {
            console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
        }

        return config;
    },
    (error) => {
        console.error('❌ Request Error:', error);
        return Promise.reject(error);
    }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
    (response) => {
        // Calculate response time
        const responseTime = new Date() - response.config.metadata.startTime;

        // Log successful responses in development
        if (process.env.NODE_ENV === 'development') {
            console.log(`✅ API Response: ${response.config.method?.toUpperCase()} ${response.config.url} (${responseTime}ms)`);
        }

        return response;
    },
    (error) => {
        const responseTime = error.config?.metadata ?
            new Date() - error.config.metadata.startTime : 0;

        console.error(`❌ API Error: ${error.config?.method?.toUpperCase()} ${error.config?.url} (${responseTime}ms)`, error);

        // Handle different error types
        if (error.code === 'ECONNABORTED') {
            toast.error('Request timeout. Please try again.');
        } else if (error.response?.status === 503) {
            toast.error('Service temporarily unavailable');
        } else if (error.response?.status >= 500) {
            toast.error('Server error. Please try again later.');
        } else if (error.response?.status === 404) {
            toast.error('Resource not found');
        } else if (!error.response) {
            toast.error('Network error. Check your connection.');
        }

        return Promise.reject(error);
    }
);

// API Service Class
class ApiService {
    // System Endpoints
    async getSystemStatus() {
        try {
            const response = await apiClient.get('/api/system/status');
            return response.data;
        } catch (error) {
            throw this.handleError(error, 'Failed to get system status');
        }
    }

    async getDatasets() {
        try {
            const response = await apiClient.get('/api/datasets');
            return response.data;
        } catch (error) {
            throw this.handleError(error, 'Failed to get datasets');
        }
    }

    async getCheckpoints(dataset) {
        try {
            const response = await apiClient.get(`/api/checkpoints/${dataset}`);
            return response.data;
        } catch (error) {
            throw this.handleError(error, `Failed to get checkpoints for ${dataset}`);
        }
    }

    async healthCheck() {
        try {
            const response = await apiClient.get('/health');
            return response.data;
        } catch (error) {
            throw this.handleError(error, 'Health check failed');
        }
    }

    // Training Endpoints
    async startTraining(config) {
        try {
            const response = await apiClient.post('/api/training/start', config);
            toast.success('Training started successfully!');
            return response.data;
        } catch (error) {
            throw this.handleError(error, 'Failed to start training');
        }
    }

    async getTrainingStatus(trainingId) {
        try {
            const response = await apiClient.get(`/api/training/status/${trainingId}`);
            return response.data;
        } catch (error) {
            throw this.handleError(error, 'Failed to get training status');
        }
    }

    async stopTraining(trainingId) {
        try {
            const response = await apiClient.post(`/api/training/stop/${trainingId}`);
            toast.success('Training stopped');
            return response.data;
        } catch (error) {
            throw this.handleError(error, 'Failed to stop training');
        }
    }

    // Generation Endpoints
    async generateImages(request) {
        try {
            const response = await apiClient.post('/api/generate', request);
            toast.success('Images generated successfully!');
            return response.data;
        } catch (error) {
            throw this.handleError(error, 'Failed to generate images');
        }
    }

    // Reports Endpoints
    async generateReport(request) {
        try {
            const response = await apiClient.post('/api/reports/generate', request);
            toast.success('Report generation started');
            return response.data;
        } catch (error) {
            throw this.handleError(error, 'Failed to generate report');
        }
    }

    async getReport(reportId) {
        try {
            const response = await apiClient.get(`/api/reports/${reportId}`);
            return response.data;
        } catch (error) {
            throw this.handleError(error, 'Failed to get report');
        }
    }

    // Logs Endpoints
    async getTrainingLogs(dataset) {
        try {
            const response = await apiClient.get(`/api/logs/${dataset}`);
            return response.data;
        } catch (error) {
            throw this.handleError(error, 'Failed to get training logs');
        }
    }

    // File Upload/Download
    async uploadFile(file, endpoint) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await apiClient.post(endpoint, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                onUploadProgress: (progressEvent) => {
                    const progress = Math.round(
                        (progressEvent.loaded * 100) / progressEvent.total
                    );
                    // You can emit progress events here if needed
                },
            });

            return response.data;
        } catch (error) {
            throw this.handleError(error, 'File upload failed');
        }
    }

    async downloadFile(url, filename) {
        try {
            const response = await apiClient.get(url, {
                responseType: 'blob',
            });

            // Create download link
            const blob = new Blob([response.data]);
            const downloadUrl = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(downloadUrl);

            toast.success('File downloaded successfully');
            return true;
        } catch (error) {
            throw this.handleError(error, 'File download failed');
        }
    }

    // Utility Methods
    handleError(error, defaultMessage) {
        const message = error.response?.data?.detail ||
            error.response?.data?.message ||
            error.message ||
            defaultMessage;

        return new Error(message);
    }

    // Cancel Token Management
    createCancelToken() {
        return axios.CancelToken.source();
    }

    isCancel(error) {
        return axios.isCancel(error);
    }

    // Real-time endpoints with polling
    async pollTrainingStatus(trainingId, onUpdate, interval = 2000) {
        const pollInterval = setInterval(async () => {
            try {
                const status = await this.getTrainingStatus(trainingId);
                onUpdate(status);

                // Stop polling if training is complete
                if (['completed', 'error', 'stopped'].includes(status.status)) {
                    clearInterval(pollInterval);
                }
            } catch (error) {
                console.error('Polling error:', error);
                onUpdate({ error: error.message });
            }
        }, interval);

        // Return cleanup function
        return () => clearInterval(pollInterval);
    }

    // Batch operations
    async batchRequest(requests) {
        try {
            const responses = await Promise.allSettled(
                requests.map(request => apiClient(request))
            );

            return responses.map((response, index) => ({
                success: response.status === 'fulfilled',
                data: response.status === 'fulfilled' ? response.value.data : null,
                error: response.status === 'rejected' ? response.reason : null,
                originalRequest: requests[index]
            }));
        } catch (error) {
            throw this.handleError(error, 'Batch request failed');
        }
    }

    // Configuration
    setAuthToken(token) {
        if (token) {
            apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        } else {
            delete apiClient.defaults.headers.common['Authorization'];
        }
    }

    setTimeout(timeout) {
        apiClient.defaults.timeout = timeout;
    }

    getBaseURL() {
        return API_BASE_URL;
    }
}

// Create and export singleton instance
const apiService = new ApiService();

// Export individual methods for convenience
export const {
    // System
    getSystemStatus,
    getDatasets,
    getCheckpoints,
    healthCheck,

    // Training
    startTraining,
    getTrainingStatus,
    stopTraining,
    pollTrainingStatus,

    // Generation
    generateImages,

    // Reports
    generateReport,
    getReport,

    // Logs
    getTrainingLogs,

    // Files
    uploadFile,
    downloadFile,

    // Utilities
    createCancelToken,
    isCancel,
    batchRequest,
    setAuthToken,
    setTimeout,
    getBaseURL,
} = apiService;

// Export the service instance as default
export default apiService;

// Export axios instance for advanced usage
export { apiClient };

// Enhanced WebSocket helper with robust connection management
export class WebSocketService {
    constructor(url = null) {
        this.url = url || `${API_BASE_URL.replace('http', 'ws')}/ws`;
        this.ws = null;
        this.reconnectInterval = 5000;
        this.maxReconnectAttempts = 10;
        this.reconnectAttempts = 0;
        this.listeners = new Map();
        this.connectionState = 'disconnected'; // disconnected, connecting, connected, error
        this.connectionPromise = null;
        this.reconnectTimer = null;
        this.isDestroyed = false;
        this.connectionId = 0; // Track connection instances
    }

    connect() {
        // Prevent multiple simultaneous connection attempts
        if (this.connectionState === 'connecting' && this.connectionPromise) {
            return this.connectionPromise;
        }

        if (this.connectionState === 'connected' && this.isConnected()) {
            return Promise.resolve();
        }

        // Increment connection ID to track this specific connection attempt
        const currentConnectionId = ++this.connectionId;

        this.connectionState = 'connecting';
        this.connectionPromise = new Promise((resolve, reject) => {
            try {
                // Clear any existing connection
                if (this.ws && this.ws.readyState !== WebSocket.CLOSED) {
                    this.ws.close(1000, 'New connection attempt');
                }

                this.ws = new WebSocket(this.url);

                const connectionTimeout = setTimeout(() => {
                    if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
                        this.ws.close(1000, 'Connection timeout');
                        if (currentConnectionId === this.connectionId) {
                            this.connectionState = 'error';
                            reject(new Error('WebSocket connection timeout'));
                        }
                    }
                }, 10000); // 10 second timeout

                this.ws.onopen = (event) => {
                    clearTimeout(connectionTimeout);

                    // Only proceed if this is still the current connection attempt
                    if (this.isDestroyed || currentConnectionId !== this.connectionId) {
                        this.ws.close(1000, 'Outdated connection');
                        return;
                    }

                    console.log('🔌 WebSocket connected');
                    this.connectionState = 'connected';
                    this.reconnectAttempts = 0;
                    this.emit('open', event);
                    resolve(event);
                };

                this.ws.onmessage = (event) => {
                    if (this.isDestroyed || currentConnectionId !== this.connectionId) return;

                    try {
                        const data = JSON.parse(event.data);
                        this.handleMessage(data);
                    } catch (error) {
                        console.error('WebSocket message parse error:', error);
                    }
                };

                this.ws.onclose = (event) => {
                    clearTimeout(connectionTimeout);

                    // Only handle if this is the current connection
                    if (currentConnectionId === this.connectionId) {
                        this.connectionState = 'disconnected';
                        this.connectionPromise = null;

                        console.log('🔌 WebSocket disconnected', event.code, event.reason);
                        this.emit('close', event);

                        // Only attempt reconnection if it wasn't a manual close and not destroyed
                        if (!this.isDestroyed && event.code !== 1000 && event.code !== 1001) {
                            this.handleReconnect();
                        }
                    }
                };

                this.ws.onerror = (error) => {
                    clearTimeout(connectionTimeout);

                    if (currentConnectionId === this.connectionId) {
                        console.error('🔌 WebSocket error:', error);
                        this.connectionState = 'error';
                        this.emit('error', error);

                        // Don't reject immediately if connecting, let onclose handle it
                        if (this.ws.readyState === WebSocket.CONNECTING) {
                            reject(error);
                        }
                    }
                };

            } catch (error) {
                if (currentConnectionId === this.connectionId) {
                    this.connectionState = 'error';
                    this.connectionPromise = null;
                    reject(error);
                }
            }
        });

        return this.connectionPromise;
    }

    disconnect() {
        this.isDestroyed = true;
        this.connectionState = 'disconnected';

        // Clear reconnection timer
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        // Close WebSocket connection
        if (this.ws && this.ws.readyState !== WebSocket.CLOSED) {
            try {
                // Use code 1000 (normal closure) to prevent reconnection attempts
                this.ws.close(1000, 'Manual disconnect');
            } catch (error) {
                console.warn('Error closing WebSocket:', error);
            }
        }

        this.ws = null;
        this.connectionPromise = null;

        console.log('🔌 WebSocket manually disconnected');
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                this.ws.send(JSON.stringify(data));
                return true;
            } catch (error) {
                console.error('WebSocket send error:', error);
                return false;
            }
        } else {
            console.warn('WebSocket not connected, cannot send message');
            return false;
        }
    }

    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }

    off(event, callback) {
        if (this.listeners.has(event)) {
            const callbacks = this.listeners.get(event);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }

    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`WebSocket event handler error for '${event}':`, error);
                }
            });
        }
    }

    handleMessage(data) {
        const { type } = data;

        // Emit specific event type
        if (type) {
            this.emit(type, data);
        }

        // Always emit general 'message' event
        this.emit('message', data);
    }

    handleReconnect() {
        if (this.isDestroyed || this.reconnectAttempts >= this.maxReconnectAttempts) {
            if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                console.error('Max WebSocket reconnection attempts reached');
                this.emit('maxReconnectAttemptsReached');
            }
            return;
        }

        this.reconnectAttempts++;
        console.log(`🔄 WebSocket reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);

        this.reconnectTimer = setTimeout(() => {
            if (!this.isDestroyed) {
                this.connect().catch(error => {
                    console.error('WebSocket reconnection failed:', error);
                });
            }
        }, this.reconnectInterval);
    }

    isConnected() {
        return this.ws &&
            this.ws.readyState === WebSocket.OPEN &&
            this.connectionState === 'connected' &&
            !this.isDestroyed;
    }

    getConnectionState() {
        return this.connectionState;
    }

    // Reset the service for reuse
    reset() {
        this.disconnect();
        setTimeout(() => {
            this.isDestroyed = false;
            this.reconnectAttempts = 0;
            this.connectionState = 'disconnected';
            this.connectionId = 0;
        }, 100);
    }
}

// Create and export singleton WebSocket instance
export const webSocketService = new WebSocketService();