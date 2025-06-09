// File: frontend/src/hooks/useWebSocket.js

import { useState, useEffect, useRef, useCallback } from 'react';
import { webSocketService } from '../services/api';

export const useWebSocket = (autoConnect = true) => {
    const [isConnected, setIsConnected] = useState(false);
    const [connectionStatus, setConnectionStatus] = useState('disconnected');
    const [lastMessage, setLastMessage] = useState(null);
    const [error, setError] = useState(null);

    const mountedRef = useRef(true);
    const connectAttemptedRef = useRef(false);
    const listenersRef = useRef(new Set());

    // Handle connection state changes
    const updateConnectionState = useCallback(() => {
        if (!mountedRef.current) return;

        const connected = webSocketService.isConnected();
        const state = webSocketService.getConnectionState();

        setIsConnected(connected);
        setConnectionStatus(state);

        if (connected) {
            setError(null);
        }
    }, []);

    // Message handler
    const handleMessage = useCallback((data) => {
        if (mountedRef.current) {
            setLastMessage(data);
        }
    }, []);

    // Error handler
    const handleError = useCallback((error) => {
        if (mountedRef.current) {
            setError(error);
            setConnectionStatus('error');
        }
    }, []);

    // Connection handlers
    const handleOpen = useCallback(() => {
        if (mountedRef.current) {
            setIsConnected(true);
            setConnectionStatus('connected');
            setError(null);
        }
    }, []);

    const handleClose = useCallback(() => {
        if (mountedRef.current) {
            setIsConnected(false);
            setConnectionStatus('disconnected');
        }
    }, []);

    // Connect function
    const connect = useCallback(async () => {
        if (!mountedRef.current || connectAttemptedRef.current) {
            return;
        }

        try {
            connectAttemptedRef.current = true;
            setError(null);
            setConnectionStatus('connecting');

            await webSocketService.connect();
            updateConnectionState();
        } catch (err) {
            if (mountedRef.current) {
                setError(err);
                setConnectionStatus('error');
                console.error('WebSocket connection failed:', err);
            }
        } finally {
            if (mountedRef.current) {
                connectAttemptedRef.current = false;
            }
        }
    }, [updateConnectionState]);

    // Disconnect function
    const disconnect = useCallback(() => {
        // Remove all listeners first
        listenersRef.current.forEach(cleanup => cleanup());
        listenersRef.current.clear();

        // Don't actually disconnect the singleton service, just update local state
        if (mountedRef.current) {
            setIsConnected(false);
            setConnectionStatus('disconnected');
        }
    }, []);

    // Send message function
    const sendMessage = useCallback((data) => {
        return webSocketService.send(data);
    }, []);

    // Subscribe to events
    const subscribe = useCallback((event, callback) => {
        webSocketService.on(event, callback);

        // Return unsubscribe function and track it
        const unsubscribe = () => {
            webSocketService.off(event, callback);
        };

        listenersRef.current.add(unsubscribe);
        return unsubscribe;
    }, []);

    // Set up event listeners
    useEffect(() => {
        if (!mountedRef.current) return;

        // Set up core event listeners
        const unsubscribeMessage = subscribe('message', handleMessage);
        const unsubscribeError = subscribe('error', handleError);
        const unsubscribeOpen = subscribe('open', handleOpen);
        const unsubscribeClose = subscribe('close', handleClose);

        return () => {
            unsubscribeMessage();
            unsubscribeError();
            unsubscribeOpen();
            unsubscribeClose();
        };
    }, [subscribe, handleMessage, handleError, handleOpen, handleClose]);

    // Auto-connect effect
    useEffect(() => {
        if (!autoConnect || !mountedRef.current) return;

        // Small delay to avoid React Strict Mode issues
        const connectTimer = setTimeout(() => {
            if (mountedRef.current && !connectAttemptedRef.current) {
                // Check if already connected
                if (webSocketService.isConnected()) {
                    updateConnectionState();
                } else {
                    connect();
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

            // Clean up listeners
            listenersRef.current.forEach(cleanup => cleanup());
            listenersRef.current.clear();

            // Small delay to avoid React Strict Mode double unmount issues
            setTimeout(() => {
                if (!mountedRef.current) {
                    // Only disconnect if we're truly unmounting (not just React Strict Mode)
                    // The WebSocket service will handle its own lifecycle
                }
            }, 100);
        };
    }, []);

    // Monitor connection state periodically
    useEffect(() => {
        const interval = setInterval(() => {
            if (mountedRef.current) {
                updateConnectionState();
            }
        }, 5000); // Check every 5 seconds

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