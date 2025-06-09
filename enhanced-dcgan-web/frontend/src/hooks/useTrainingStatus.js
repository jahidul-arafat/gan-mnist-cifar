// File: frontend/src/hooks/useTrainingStatus.js

import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import apiService from '../services/api';

export const useTrainingStatus = () => {
    const [activeTrainings, setActiveTrainings] = useState({});
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const { lastMessage } = useWebSocket();

    // Handle WebSocket messages for real-time updates
    useEffect(() => {
        if (lastMessage && lastMessage.type === 'training_status') {
            const { training_id, status } = lastMessage;
            updateTraining(training_id, status);
        }
    }, [lastMessage]);

    const addTraining = useCallback((trainingId, trainingData) => {
        setActiveTrainings(prev => ({
            ...prev,
            [trainingId]: trainingData
        }));
    }, []);

    const updateTraining = useCallback((trainingId, updates) => {
        setActiveTrainings(prev => ({
            ...prev,
            [trainingId]: prev[trainingId] ? { ...prev[trainingId], ...updates } : updates
        }));
    }, []);

    const removeTraining = useCallback((trainingId) => {
        setActiveTrainings(prev => {
            const newState = { ...prev };
            delete newState[trainingId];
            return newState;
        });
    }, []);

    const getTrainingStatus = useCallback(async (trainingId) => {
        try {
            setIsLoading(true);
            const status = await apiService.getTrainingStatus(trainingId);
            updateTraining(trainingId, status);
            return status;
        } catch (err) {
            setError(err);
            throw err;
        } finally {
            setIsLoading(false);
        }
    }, [updateTraining]);

    return {
        activeTrainings,
        isLoading,
        error,
        addTraining,
        updateTraining,
        removeTraining,
        getTrainingStatus
    };
};