// File: frontend/src/hooks/useWebSocket.js

import { useState, useEffect, useRef, useCallback } from 'react';
import { webSocketService } from '../services/api';

export const useWebSocket = (autoConnect = true) => {
    const [isConnected, setIsConnected] = useState(false);
    const [connectionStatus, setConnectionStatus] = useState('disconnected');
    const [lastMessage, setLastMessage] = useState(null);
    const [error, setError] = useState(null);

    const mountedRef = useRef(true);
    const listenersRef = useRef(new Set());
    const hasConnectedRef = useRef(false);

    // Handle connection state changes
    const updateConnectionState = useCallback(() => {
        if (!mountedRef.current) return;

        const connected = webSocketService.isConnected();
        const state = webSocketService.getConnectionState();

        setIsConnected(connected);
        setConnectionStatus(state);

        if (connected) {
            setError(null);
            hasConnectedRef.current = true;
        }
    }, []);

    // Enhanced message handler with better processing
    const handleMessage = useCallback((data) => {
        if (!mountedRef.current) return;

        console.log('📨 useWebSocket: Raw message received:', data);

        // Try to detect training status messages in different formats
        const isTrainingMessage = data.type === 'training_status' ||
            (data.training_id && (data.current_epoch !== undefined || data.data?.current_epoch !== undefined));

        if (isTrainingMessage) {
            console.log('🎯 useWebSocket: Training status message detected:', data);

            // Handle different message structures
            let current_epoch, total_epochs, training_id, status, dataset, metrics;

            if (data.data) {
                // Structure: { type: 'training_status', training_id: '...', data: { ... } }
                current_epoch = data.data.current_epoch;
                total_epochs = data.data.total_epochs;
                training_id = data.training_id;
                status = data.data.status || 'running';
                dataset = data.data.dataset;
                metrics = data.data.metrics || {};
            } else {
                // Structure: { type: 'training_status', training_id: '...', current_epoch: ..., ... }
                current_epoch = data.current_epoch;
                total_epochs = data.total_epochs;
                training_id = data.training_id;
                status = data.status || 'running';
                dataset = data.dataset;
                metrics = data.metrics || {};
            }

            // Calculate progress
            const progress_percentage = (current_epoch && total_epochs)
                ? (current_epoch / total_epochs) * 100
                : 0;

            const processedMessage = {
                type: 'training_status',
                training_id: training_id,
                data: {
                    current_epoch: current_epoch || 0,
                    total_epochs: total_epochs || 1,
                    status: status,
                    progress_percentage: progress_percentage,
                    metrics: metrics,
                    dataset: dataset,
                    last_update: new Date().toISOString()
                }
            };

            console.log('📊 useWebSocket: Processed training message:', processedMessage);
            setLastMessage(processedMessage);
        } else {
            // Handle other message types
            console.log('📝 useWebSocket: Other message type:', data.type || 'unknown');
            setLastMessage(data);
        }
    }, []);

    // Error handler
    const handleError = useCallback((error) => {
        if (mountedRef.current) {
            console.error('🔌 WebSocket error:', error);
            setError(error);
            setConnectionStatus('error');
        }
    }, []);

    // Connection handlers
    const handleOpen = useCallback(() => {
        if (mountedRef.current) {
            console.log('🎉 WebSocket connection opened');
            setIsConnected(true);
            setConnectionStatus('connected');
            setError(null);
            hasConnectedRef.current = true;
        }
    }, []);

    const handleClose = useCallback(() => {
        if (mountedRef.current) {
            console.log('📪 WebSocket connection closed');
            setIsConnected(false);
            setConnectionStatus('disconnected');
        }
    }, []);

    // Connect function
    const connect = useCallback(async () => {
        if (!mountedRef.current) return;

        try {
            setError(null);
            setConnectionStatus('connecting');
            await webSocketService.connect();
            updateConnectionState();
        } catch (err) {
            if (mountedRef.current) {
                setError(err);
                setConnectionStatus('error');
                console.error('Enhanced DCGAN WebSocket connection failed:', err);
            }
        }
    }, [updateConnectionState]);

    // Disconnect function (for manual disconnection only)
    const disconnect = useCallback(() => {
        // Clean up listeners
        const currentListeners = listenersRef.current;
        currentListeners.forEach(cleanup => cleanup());
        currentListeners.clear();

        if (mountedRef.current) {
            setIsConnected(false);
            setConnectionStatus('disconnected');
        }
    }, []);

    // Send message function
    const sendMessage = useCallback((data) => {
        if (!webSocketService.isConnected()) {
            console.warn('Cannot send message: WebSocket not connected');
            return false;
        }
        return webSocketService.send(data);
    }, []);

    // Subscribe to events with automatic cleanup tracking
    const subscribe = useCallback((event, callback) => {
        if (!mountedRef.current) return () => {};

        webSocketService.on(event, callback);

        const unsubscribe = () => {
            webSocketService.off(event, callback);
            listenersRef.current.delete(unsubscribe);
        };

        listenersRef.current.add(unsubscribe);
        return unsubscribe;
    }, []);

    // Set up core event listeners
    useEffect(() => {
        if (!mountedRef.current) return;

        const unsubscribeMessage = subscribe('message', handleMessage);
        const unsubscribeError = subscribe('error', handleError);
        const unsubscribeOpen = subscribe('open', handleOpen);
        const unsubscribeClose = subscribe('close', handleClose);

        // Subscribe specifically to training_status events
        const unsubscribeTraining = subscribe('training_status', (data) => {
            console.log('🏃 Direct training_status event:', data);
            handleMessage(data);
        });

        return () => {
            unsubscribeMessage();
            unsubscribeError();
            unsubscribeOpen();
            unsubscribeClose();
            unsubscribeTraining();
        };
    }, [subscribe, handleMessage, handleError, handleOpen, handleClose]);

    // Auto-connect effect (only if not already initialized)
    useEffect(() => {
        if (!autoConnect || !mountedRef.current) return;

        // Delay to avoid React Strict Mode issues
        const connectTimer = setTimeout(() => {
            if (mountedRef.current) {
                // Only auto-connect if the service hasn't been initialized yet
                if (!webSocketService.isInitialized) {
                    if (webSocketService.isConnected()) {
                        updateConnectionState();
                    } else {
                        connect();
                    }
                } else {
                    // Service already initialized, just sync state
                    updateConnectionState();
                }
            }
        }, 100);

        return () => clearTimeout(connectTimer);
    }, [autoConnect, connect, updateConnectionState]);

    // Component lifecycle management
    useEffect(() => {
        mountedRef.current = true;

        return () => {
            mountedRef.current = false;

            // Clean up all listeners - copy ref to avoid stale closure
            const currentListeners = listenersRef.current;
            currentListeners.forEach(cleanup => cleanup());
            currentListeners.clear();
        };
    }, []);

    // Periodic connection state monitoring
    useEffect(() => {
        const interval = setInterval(() => {
            if (mountedRef.current) {
                updateConnectionState();
            }
        }, 5000);

        return () => clearInterval(interval);
    }, [updateConnectionState]);

    return {
        isConnected,
        connectionStatus,
        lastMessage,
        error,
        connect,
        disconnect,
        sendMessage,
        subscribe
    };
};