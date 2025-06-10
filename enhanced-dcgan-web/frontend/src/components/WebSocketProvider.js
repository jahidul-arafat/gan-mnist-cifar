import React, { useEffect, useRef } from 'react';
import { webSocketService } from '../services/api';

export const WebSocketProvider = ({ children }) => {
    const initializedRef = useRef(false);
    const mountedRef = useRef(true);

    useEffect(() => {
        mountedRef.current = true;

        // Only initialize once per app lifecycle
        if (initializedRef.current) return;

        initializedRef.current = true;

        // Initialize WebSocket connection for Enhanced DCGAN
        const initializeWebSocket = async () => {
            try {
                console.log('🚀 WebSocketProvider: Initializing Enhanced DCGAN WebSocket connection...');

                // Check if already connected to avoid multiple connections
                if (webSocketService.isConnected()) {
                    console.log('✅ Enhanced DCGAN WebSocket already connected');
                    return;
                }

                await webSocketService.connect();
                console.log('✅ WebSocketProvider: Enhanced DCGAN WebSocket initialized successfully');

                // Add connection state logging
                webSocketService.on('open', () => {
                    console.log('🎉 WebSocketProvider: Connection established');
                });

                webSocketService.on('close', () => {
                    console.log('📪 WebSocketProvider: Connection closed');
                });

                webSocketService.on('error', (error) => {
                    console.error('🚨 WebSocketProvider: Error:', error);
                });

                // Log training updates
                webSocketService.on('training_status', (data) => {
                    console.log('📊 WebSocketProvider: Training update received:', data.data?.current_epoch, '/', data.data?.total_epochs);
                });

            } catch (error) {
                console.error('❌ WebSocketProvider: Failed to initialize Enhanced DCGAN WebSocket:', error);

                // Retry connection after delay
                setTimeout(() => {
                    if (mountedRef.current) {
                        console.log('🔄 WebSocketProvider: Retrying WebSocket connection...');
                        initializeWebSocket();
                    }
                }, 5000);
            }
        };

        // Immediate connection attempt
        initializeWebSocket();

        return () => {
            // Cleanup handled in separate effect
        };
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            mountedRef.current = false;
        };
    }, []);

    return <>{children}</>;
};
