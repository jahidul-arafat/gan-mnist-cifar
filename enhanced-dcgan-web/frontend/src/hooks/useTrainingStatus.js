// File: frontend/src/hooks/useTrainingStatus.js

import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import apiService from '../services/api';

// Persistent storage keys
const STORAGE_KEYS = {
    ACTIVE_TRAININGS: 'dcgan_active_trainings',
    LAST_UPDATE: 'dcgan_last_training_update'
};

// Helper functions for persistent storage
const saveToStorage = (key, data) => {
    try {
        localStorage.setItem(key, JSON.stringify(data));
        console.log('💾 Saved to localStorage:', key, Object.keys(data).length, 'trainings');
    } catch (error) {
        console.warn('⚠️ Failed to save to localStorage:', error);
    }
};

const loadFromStorage = (key, defaultValue = {}) => {
    try {
        const stored = localStorage.getItem(key);
        if (stored) {
            const parsed = JSON.parse(stored);
            console.log('📂 Loaded from localStorage:', key, Object.keys(parsed).length, 'trainings');
            return parsed;
        }
    } catch (error) {
        console.warn('⚠️ Failed to load from localStorage:', error);
    }
    return defaultValue;
};

const clearStorage = (key) => {
    try {
        localStorage.removeItem(key);
        console.log('🗑️ Cleared localStorage:', key);
    } catch (error) {
        console.warn('⚠️ Failed to clear localStorage:', error);
    }
};

// Clean old completed trainings from storage
const cleanOldTrainings = (trainings, maxAge = 24 * 60 * 60 * 1000) => { // 24 hours
    const now = Date.now();
    const cleaned = {};
    let removedCount = 0;

    Object.entries(trainings).forEach(([id, training]) => {
        const lastUpdate = new Date(training.last_update || training.start_time).getTime();
        const age = now - lastUpdate;

        // Keep running/starting trainings regardless of age
        if (['running', 'starting'].includes(training.status)) {
            cleaned[id] = training;
        }
        // Keep recent trainings
        else if (age < maxAge) {
            cleaned[id] = training;
        }
        // Remove old completed/error trainings
        else {
            removedCount++;
            console.log('🧹 Removing old training:', id, training.status, `${Math.round(age / (60 * 60 * 1000))}h old`);
        }
    });

    if (removedCount > 0) {
        console.log(`🧹 Cleaned ${removedCount} old trainings from storage`);
    }

    return cleaned;
};

export const useTrainingStatus = () => {
    // Load initial state from localStorage
    const [activeTrainings, setActiveTrainings] = useState(() => {
        const stored = loadFromStorage(STORAGE_KEYS.ACTIVE_TRAININGS);
        return cleanOldTrainings(stored);
    });

    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const { lastMessage, isConnected } = useWebSocket();

    // Save to localStorage whenever activeTrainings changes
    useEffect(() => {
        // Clean and save
        const cleaned = cleanOldTrainings(activeTrainings);
        saveToStorage(STORAGE_KEYS.ACTIVE_TRAININGS, cleaned);
        saveToStorage(STORAGE_KEYS.LAST_UPDATE, new Date().toISOString());

        // Update state if cleaning removed items
        if (Object.keys(cleaned).length !== Object.keys(activeTrainings).length) {
            setActiveTrainings(cleaned);
        }
    }, [activeTrainings]);

    // Define all functions FIRST before using them in useEffect
    const addTraining = useCallback((trainingId, trainingData) => {
        console.log('➕ useTrainingStatus: Adding training:', trainingId, trainingData);

        setActiveTrainings(prev => {
            const newState = {
                ...prev,
                [trainingId]: {
                    training_id: trainingId,
                    status: 'starting',
                    dataset: 'unknown',
                    current_epoch: 0,
                    total_epochs: 50,
                    progress_percentage: 0,
                    metrics: {},
                    start_time: new Date().toISOString(),
                    end_time: null,
                    error_message: null,
                    added_at: new Date().toISOString(),
                    source: 'manual', // Track how this training was added
                    ...trainingData
                }
            };

            console.log('💾 Persisting new training to storage:', trainingId);
            return newState;
        });
    }, []);

    const updateTraining = useCallback((trainingId, updates) => {
        console.log('🔄 useTrainingStatus: Updating training:', trainingId, updates);

        setActiveTrainings(prev => {
            const currentTraining = prev[trainingId];

            if (currentTraining) {
                // Calculate progress percentage if epochs are provided
                let progressPercentage = updates.progress_percentage;
                if (updates.current_epoch !== undefined && updates.total_epochs !== undefined) {
                    progressPercentage = (updates.current_epoch / updates.total_epochs) * 100;
                }

                const updatedTraining = {
                    ...currentTraining,
                    ...updates,
                    progress_percentage: progressPercentage !== undefined ? progressPercentage : currentTraining.progress_percentage,
                    last_update: new Date().toISOString()
                };

                console.log('✅ useTrainingStatus: Training updated:', trainingId, updatedTraining);

                return {
                    ...prev,
                    [trainingId]: updatedTraining
                };
            } else {
                // Create new training if it doesn't exist (from WebSocket)
                const newTraining = {
                    training_id: trainingId,
                    status: 'running',
                    dataset: 'unknown',
                    current_epoch: 0,
                    total_epochs: 50,
                    progress_percentage: 0,
                    metrics: {},
                    start_time: new Date().toISOString(),
                    end_time: null,
                    error_message: null,
                    added_at: new Date().toISOString(),
                    source: 'websocket', // Track that this came from WebSocket
                    ...updates,
                    last_update: new Date().toISOString()
                };

                // Calculate progress for new training
                if (newTraining.current_epoch !== undefined && newTraining.total_epochs !== undefined) {
                    newTraining.progress_percentage = (newTraining.current_epoch / newTraining.total_epochs) * 100;
                }

                console.log('🆕 useTrainingStatus: Creating new training from WebSocket:', trainingId, newTraining);

                return {
                    ...prev,
                    [trainingId]: newTraining
                };
            }
        });
    }, []);

    const removeTraining = useCallback((trainingId) => {
        console.log('🗑️ useTrainingStatus: Removing training:', trainingId);

        setActiveTrainings(prev => {
            const newState = { ...prev };
            delete newState[trainingId];
            return newState;
        });
    }, []);

    const clearAllTrainings = useCallback(() => {
        console.log('🧹 useTrainingStatus: Clearing all trainings');
        setActiveTrainings({});
        clearStorage(STORAGE_KEYS.ACTIVE_TRAININGS);
        clearStorage(STORAGE_KEYS.LAST_UPDATE);
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

    // Check if any training is currently active (running or starting)
    const hasActiveTraining = useCallback(() => {
        return Object.values(activeTrainings).some(training =>
            ['running', 'starting'].includes(training.status)
        );
    }, [activeTrainings]);

    // Get all running trainings
    const getRunningTrainings = useCallback(() => {
        return Object.values(activeTrainings).filter(training =>
            ['running', 'starting'].includes(training.status)
        );
    }, [activeTrainings]);

    // Sync with backend on page load/refresh
    const syncWithBackend = useCallback(async () => {
        console.log('🔄 useTrainingStatus: Syncing with backend...');

        try {
            // Try to get active training sessions from backend
            const response = await apiService.get('/api/training/active');
            const activeTrainingSessions = response.data;

            if (Array.isArray(activeTrainingSessions) && activeTrainingSessions.length > 0) {
                console.log('✅ useTrainingStatus: Found active trainings on backend:', activeTrainingSessions);

                // Update each training from backend
                activeTrainingSessions.forEach(training => {
                    updateTraining(training.training_id, {
                        ...training,
                        source: 'backend_sync'
                    });
                });
            } else {
                console.log('📝 useTrainingStatus: No active trainings found on backend');

                // Check if we have running trainings in localStorage that might be stale
                const runningTrainings = getRunningTrainings();
                if (runningTrainings.length > 0) {
                    console.log('⚠️ Found running trainings in localStorage but none on backend, checking status...');

                    // Check each training individually
                    for (const training of runningTrainings) {
                        try {
                            await getTrainingStatus(training.training_id);
                        } catch (error) {
                            console.log(`❌ Training ${training.training_id} not found on backend, marking as error`);
                            updateTraining(training.training_id, {
                                status: 'error',
                                error_message: 'Training session not found on server',
                                end_time: new Date().toISOString()
                            });
                        }
                    }
                }
            }
        } catch (error) {
            console.log('💡 useTrainingStatus: Backend sync not available, relying on WebSocket updates');
            // This is normal and expected if the endpoint doesn't exist
        }
    }, [updateTraining, getRunningTrainings, getTrainingStatus]);

    // Handle WebSocket messages for real-time updates
    useEffect(() => {
        console.log('🔄 useTrainingStatus: Checking lastMessage:', lastMessage);

        if (lastMessage && lastMessage.type === 'training_status') {
            console.log('🔄 useTrainingStatus: Processing training status update:', lastMessage);

            const { training_id, data } = lastMessage;

            if (training_id && data) {
                console.log('📊 useTrainingStatus: Updating training:', training_id, data);
                updateTraining(training_id, {
                    ...data,
                    source: 'websocket_update'
                });
            } else {
                console.warn('⚠️ useTrainingStatus: Invalid training status message format:', lastMessage);
            }
        }
    }, [lastMessage, updateTraining]);

    // Sync with backend on mount and when connection is restored
    useEffect(() => {
        if (isConnected) {
            // Small delay to ensure WebSocket is fully ready
            const syncTimer = setTimeout(() => {
                syncWithBackend();
            }, 1000);

            return () => clearTimeout(syncTimer);
        }
    }, [isConnected, syncWithBackend]);

    // Auto-cleanup completed trainings after 5 minutes (but keep in localStorage for longer)
    useEffect(() => {
        const cleanup = setInterval(() => {
            const now = new Date();
            setActiveTrainings(prev => {
                const updated = { ...prev };
                let hasChanges = false;

                Object.entries(updated).forEach(([id, training]) => {
                    if (training.status === 'completed' && training.end_time) {
                        const endTime = new Date(training.end_time);
                        const timeDiff = (now - endTime) / (1000 * 60); // minutes
                        if (timeDiff > 5) {
                            console.log('🧹 useTrainingStatus: Auto-removing completed training from UI:', id);
                            delete updated[id];
                            hasChanges = true;
                        }
                    }
                });

                return hasChanges ? updated : prev;
            });
        }, 60000); // Check every minute

        return () => clearInterval(cleanup);
    }, []);

    // Periodic storage cleanup (every 30 minutes)
    useEffect(() => {
        const cleanup = setInterval(() => {
            setActiveTrainings(prev => cleanOldTrainings(prev));
        }, 30 * 60 * 1000);

        return () => clearInterval(cleanup);
    }, []);

    // Debug logging
    useEffect(() => {
        console.log('📊 useTrainingStatus: Active trainings state:', activeTrainings);
        console.log('🔧 useTrainingStatus: Has active training:', hasActiveTraining());
        console.log('🏃 useTrainingStatus: Running trainings:', getRunningTrainings().length);
    }, [activeTrainings, hasActiveTraining, getRunningTrainings]);

    // Listen for page unload to ensure final save
    useEffect(() => {
        const handleBeforeUnload = () => {
            saveToStorage(STORAGE_KEYS.ACTIVE_TRAININGS, activeTrainings);
        };

        window.addEventListener('beforeunload', handleBeforeUnload);
        return () => window.removeEventListener('beforeunload', handleBeforeUnload);
    }, [activeTrainings]);

    return {
        activeTrainings,
        isLoading,
        error,
        isConnected,
        hasActiveTraining,
        getRunningTrainings,
        addTraining,
        updateTraining,
        removeTraining,
        clearAllTrainings,
        getTrainingStatus,
        syncWithBackend
    };
};