// File: frontend/src/services/api.js

import axios from 'axios';
import toast from 'react-hot-toast';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 120000; // 2 minutes for training operations
const GENERATION_TIMEOUT = 60000; // 1 minute for image generation

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
            if (config.data) {
                console.log('📤 Request Data:', config.data);
            }
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
            console.log('📥 Response Data:', response.data);
        }

        return response;
    },
    (error) => {
        const responseTime = error.config?.metadata ?
            new Date() - error.config.metadata.startTime : 0;

        console.error(`❌ API Error: ${error.config?.method?.toUpperCase()} ${error.config?.url} (${responseTime}ms)`, error);

        // Handle different error types
        if (error.code === 'ECONNABORTED') {
            toast.error('Request timeout. Training operations may take longer - check the logs tab for progress.');
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
    constructor() {
        this.baseURL = API_BASE_URL;
        console.log('🚀 Enhanced DCGAN API Service initialized');
        console.log('📡 Base URL:', this.baseURL);
    }

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
            console.log(`📋 Checkpoints for ${dataset}:`, response.data);

            // Ensure we always return a consistent format
            // Most components expect { checkpoints: array }
            if (Array.isArray(response.data)) {
                // If response.data is directly an array, wrap it
                return {
                    checkpoints: response.data,
                    total: response.data.length
                };
            } else if (response.data && response.data.checkpoints && Array.isArray(response.data.checkpoints)) {
                // If response.data has checkpoints property, return as-is
                return response.data;
            } else if (response.data && typeof response.data === 'object') {
                // If response.data is an object but not in expected format
                return {
                    checkpoints: Object.values(response.data).filter(item =>
                        item && typeof item === 'object' && (item.epoch !== undefined || item.filename || item.name)
                    ),
                    total: Object.keys(response.data).length
                };
            } else {
                // Fallback: return empty structure
                return {
                    checkpoints: [],
                    total: 0
                };
            }
        } catch (error) {
            console.error(`❌ Failed to get checkpoints for ${dataset}:`, error);
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
            console.log('🎯 Starting training with config:', config);
            const response = await apiClient.post('/api/training/start', config, {
                timeout: 180000 // 3 minutes for training start
            });
            toast.success('Training started successfully!');
            return response.data;
        } catch (error) {
            console.error('❌ Training start failed:', error);
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

    // Enhanced Generation Endpoint with better error handling
    async generateImages(request) {
        try {
            console.log('🎨 Image generation request:', request);

            // Validate request before sending
            if (!request.dataset) {
                throw new Error('Dataset is required for image generation');
            }

            if (!request.num_samples || request.num_samples < 1) {
                throw new Error('Number of samples must be at least 1');
            }

            // Prepare the request with proper structure
            const generationRequest = {
                dataset: request.dataset,
                num_samples: parseInt(request.num_samples) || 8,
                seed: parseInt(request.seed) || 42,
                use_ema: request.use_ema !== false, // Default to true
                device: request.device || 'auto',
                ...request
            };

            // Remove undefined values
            Object.keys(generationRequest).forEach(key => {
                if (generationRequest[key] === undefined) {
                    delete generationRequest[key];
                }
            });

            console.log('📤 Sending generation request:', generationRequest);

            const response = await apiClient.post('/api/generate', generationRequest, {
                timeout: GENERATION_TIMEOUT,
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            console.log('📥 Generation response received:', response.data);

            // Validate response structure
            if (!response.data) {
                throw new Error('Empty response from generation API');
            }

            const { images, generation_id, metadata } = response.data;

            if (!images || !Array.isArray(images)) {
                console.error('❌ Invalid response structure:', response.data);
                throw new Error('Invalid response: images array not found');
            }

            if (images.length === 0) {
                throw new Error('No images generated. Check if model is properly trained.');
            }

            // Process and validate each image
            const processedImages = images.map((image, index) => {
                if (typeof image === 'string') {
                    // Handle base64 strings
                    let imageUrl = image;

                    // Add data URL prefix if missing
                    if (!image.startsWith('data:image/')) {
                        // Detect image format and add appropriate header
                        if (image.startsWith('/9j/')) {
                            imageUrl = `data:image/jpeg;base64,${image}`;
                        } else if (image.startsWith('iVBORw0KGgo')) {
                            imageUrl = `data:image/png;base64,${image}`;
                        } else {
                            // Default to PNG
                            imageUrl = `data:image/png;base64,${image}`;
                        }
                    }

                    return {
                        id: `gen_${Date.now()}_${index}`,
                        url: imageUrl,
                        generated_at: new Date().toISOString(),
                        dataset: request.dataset,
                        index: index
                    };
                } else if (image && typeof image === 'object') {
                    // Handle object format
                    return {
                        id: image.id || `gen_${Date.now()}_${index}`,
                        url: image.url || image.data || image.image,
                        generated_at: image.generated_at || new Date().toISOString(),
                        dataset: request.dataset,
                        index: index,
                        ...image
                    };
                } else {
                    console.warn('⚠️ Unexpected image format:', image);
                    return null;
                }
            }).filter(img => img !== null && img.url);

            if (processedImages.length === 0) {
                throw new Error('No valid images processed from response');
            }

            console.log('✅ Processed images:', processedImages.length);

            const result = {
                images: processedImages,
                generation_id: generation_id || Date.now().toString(),
                metadata: {
                    dataset: request.dataset,
                    num_samples: processedImages.length,
                    seed: request.seed,
                    generated_at: new Date().toISOString(),
                    ...metadata
                }
            };

            toast.success(`Successfully generated ${processedImages.length} images!`, {
                duration: 4000,
                icon: '🎨'
            });

            return result;

        } catch (error) {
            console.error('❌ Image generation failed:', error);

            // Provide specific error messages
            let errorMessage = 'Failed to generate images';

            if (error.message) {
                errorMessage = error.message;
            } else if (error.response?.data?.detail) {
                errorMessage = error.response.data.detail;
            } else if (error.response?.data?.message) {
                errorMessage = error.response.data.message;
            }

            // Add context-specific guidance
            if (errorMessage.includes('checkpoint') || errorMessage.includes('model')) {
                errorMessage += '\n\nTip: Ensure the model is trained and checkpoints exist for the selected dataset.';
            } else if (errorMessage.includes('connection') || errorMessage.includes('network')) {
                errorMessage += '\n\nTip: Check your connection to the backend server.';
            } else if (errorMessage.includes('timeout')) {
                errorMessage += '\n\nTip: Generation is taking longer than expected. The model might be loading.';
            }

            // Don't show toast here as InteractiveGeneration will handle it
            throw new Error(errorMessage);
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

    // File operations
    async uploadFile(file, endpoint) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await apiClient.post(endpoint, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                onUploadProgress: (progressEvent) => {
                    const progressPercent = Math.round(
                        (progressEvent.loaded * 100) / progressEvent.total
                    );
                    if (process.env.NODE_ENV === 'development') {
                        console.log(`Upload progress: ${progressPercent}%`);
                    }
                },
            });

            return response.data;
        } catch (error) {
            throw this.handleError(error, 'File upload failed');
        }
    }

    async downloadFile(url, filename) {
        try {
            const response = await apiClient.get(url, { responseType: 'blob' });
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
        // Extract the most relevant error message
        let message = defaultMessage;

        if (error.response?.data?.detail) {
            message = error.response.data.detail;
        } else if (error.response?.data?.message) {
            message = error.response.data.message;
        } else if (error.response?.data?.error) {
            message = error.response.data.error;
        } else if (error.message) {
            message = error.message;
        }

        // Log the full error for debugging
        console.error('🚨 API Error Details:', {
            message,
            status: error.response?.status,
            statusText: error.response?.statusText,
            data: error.response?.data,
            originalError: error
        });

        return new Error(message);
    }

    createCancelToken() {
        return axios.CancelToken.source();
    }

    isCancel(error) {
        return axios.isCancel(error);
    }

    async pollTrainingStatus(trainingId, onUpdate, interval = 2000) {
        const pollInterval = setInterval(async () => {
            try {
                const status = await this.getTrainingStatus(trainingId);
                onUpdate(status);
                if (['completed', 'error', 'stopped'].includes(status.status)) {
                    clearInterval(pollInterval);
                }
            } catch (error) {
                console.error('Polling error:', error);
                onUpdate({ error: error.message });
            }
        }, interval);
        return () => clearInterval(pollInterval);
    }

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

    // Debug method to test API connection
    async testConnection() {
        try {
            console.log('🔍 Testing API connection...');
            const response = await this.healthCheck();
            console.log('✅ API connection successful:', response);
            return { success: true, data: response };
        } catch (error) {
            console.error('❌ API connection failed:', error);
            return { success: false, error: error.message };
        }
    }
}

// Enhanced WebSocket Service with Robust Connection Management
export class WebSocketService {
    constructor(url = null) {
        this.url = url || `${API_BASE_URL.replace('http', 'ws')}/ws`;
        this.ws = null;
        this.reconnectInterval = 5000;
        this.maxReconnectAttempts = 10;
        this.reconnectAttempts = 0;
        this.listeners = new Map();
        this.connectionState = 'disconnected';
        this.connectionPromise = null;
        this.reconnectTimer = null;
        this.isDestroyed = false;
        this.connectionId = 0;
        this.isInitialized = false;

        console.log('🔌 Enhanced DCGAN WebSocketService initialized');
        console.log('🔗 WebSocket URL:', this.url);
    }

    async connect() {
        // Prevent multiple simultaneous connection attempts
        if (this.connectionState === 'connecting' && this.connectionPromise) {
            return this.connectionPromise;
        }

        if (this.connectionState === 'connected' && this.isConnected()) {
            return Promise.resolve();
        }

        // Mark as initialized on first connection attempt
        this.isInitialized = true;

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
                }, 10000);

                this.ws.onopen = (event) => {
                    clearTimeout(connectionTimeout);

                    // Only proceed if this is still the current connection attempt
                    if (this.isDestroyed || currentConnectionId !== this.connectionId) {
                        if (this.ws) this.ws.close(1000, 'Outdated connection');
                        return;
                    }

                    console.log('🔌 WebSocket connected to Enhanced DCGAN Backend');
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
                        this.emit('error', error);
                    }
                };

                this.ws.onclose = (event) => {
                    clearTimeout(connectionTimeout);

                    // Only handle if this is the current connection
                    if (currentConnectionId === this.connectionId) {
                        this.connectionState = 'disconnected';
                        this.connectionPromise = null;

                        console.log(`🔌 WebSocket disconnected (${event.code}: ${event.reason})`);
                        this.emit('close', event);

                        // Only attempt reconnection for unexpected disconnections
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

                        if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
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
                this.ws.close(1000, 'Manual disconnect');
            } catch (error) {
                console.warn('Error closing WebSocket:', error);
            }
        }

        this.ws = null;
        this.connectionPromise = null;
        console.log('🔌 Enhanced DCGAN WebSocket disconnected');
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                this.ws.send(JSON.stringify(data));
                return true;
            } catch (error) {
                console.error('WebSocket send error:', error);
                this.emit('error', error);
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

        // Emit specific event type for Enhanced DCGAN events
        if (type) {
            this.emit(type, data);
        }

        // Always emit general 'message' event
        this.emit('message', data);
    }

    handleReconnect() {
        if (this.isDestroyed || this.reconnectAttempts >= this.maxReconnectAttempts) {
            if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                console.error('Max Enhanced DCGAN WebSocket reconnection attempts reached');
                this.emit('maxReconnectAttemptsReached');
            }
            return;
        }

        this.reconnectAttempts++;
        const delay = Math.min(this.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1), 30000);

        console.log(`🔄 Enhanced DCGAN WebSocket reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);

        this.reconnectTimer = setTimeout(() => {
            if (!this.isDestroyed) {
                this.connect().catch(error => {
                    console.error('Enhanced DCGAN WebSocket reconnection failed:', error);
                });
            }
        }, delay);
    }

    isConnected() {
        return this.ws &&
            this.ws.readyState === WebSocket.OPEN &&
            this.connectionState === 'connected' &&
            !this.isDestroyed;
    }

    getConnectionState() {
        if (!this.ws) return 'disconnected';
        switch (this.ws.readyState) {
            case WebSocket.OPEN: return 'connected';
            case WebSocket.CONNECTING: return 'connecting';
            default: return 'disconnected';
        }
    }

    // Reset for reuse
    reset() {
        this.disconnect();
        setTimeout(() => {
            this.isDestroyed = false;
            this.reconnectAttempts = 0;
            this.connectionState = 'disconnected';
            this.connectionId = 0;
            this.isInitialized = false;
        }, 100);
    }
}

// Create singleton instances
const apiService = new ApiService();
export const webSocketService = new WebSocketService();

// Export individual methods for convenience
export const {
    getSystemStatus,
    getDatasets,
    getCheckpoints,
    healthCheck,
    startTraining,
    getTrainingStatus,
    stopTraining,
    pollTrainingStatus,
    generateImages,
    generateReport,
    getReport,
    getTrainingLogs,
    uploadFile,
    downloadFile,
    createCancelToken,
    isCancel,
    batchRequest,
    setAuthToken,
    setTimeout,
    getBaseURL,
    testConnection,
} = apiService;

export default apiService;
export { apiClient };