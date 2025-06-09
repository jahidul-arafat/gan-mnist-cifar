// File: frontend/src/components/WebSocketProvider.js

import { useEffect, useRef } from 'react';
import { webSocketService } from '../services/api';

export const WebSocketProvider = ({ children }) => {
    const initializedRef = useRef(false);

    useEffect(() => {
        // Initialize WebSocket connection once at app level
        if (!initializedRef.current) {
            initializedRef.current = true;

            console.log('🔌 Initializing WebSocket connection...');

            // Connect with a delay to avoid immediate connection issues
            const timer = setTimeout(() => {
                webSocketService.connect().catch(error => {
                    console.warn('Initial WebSocket connection failed:', error);
                    // The service will handle reconnection automatically
                });
            }, 1000);

            return () => clearTimeout(timer);
        }

        // Cleanup on app unmount
        return () => {
            if (initializedRef.current) {
                console.log('🔌 Cleaning up WebSocket connection...');
                webSocketService.disconnect();
            }
        };
    }, []);

    return children;
};