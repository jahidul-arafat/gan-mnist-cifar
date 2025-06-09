// File: frontend/src/hooks/useSystemStatus.js

import { useState, useEffect, useCallback } from 'react';
import apiService from '../services/api';

export const useSystemStatus = () => {
    const [systemStatus, setSystemStatus] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [lastUpdate, setLastUpdate] = useState(null);

    const fetchSystemStatus = useCallback(async () => {
        try {
            setError(null);
            const status = await apiService.getSystemStatus();
            setSystemStatus(status);
            setLastUpdate(new Date());
        } catch (err) {
            setError(err);
            console.error('Failed to fetch system status:', err);
        } finally {
            setIsLoading(false);
        }
    }, []);

    const refetch = useCallback(() => {
        setIsLoading(true);
        fetchSystemStatus();
    }, [fetchSystemStatus]);

    useEffect(() => {
        fetchSystemStatus();

        // Set up periodic refresh every 30 seconds
        const interval = setInterval(fetchSystemStatus, 30000);

        return () => clearInterval(interval);
    }, [fetchSystemStatus]);

    return {
        systemStatus,
        isLoading,
        error,
        lastUpdate,
        refetch
    };
};