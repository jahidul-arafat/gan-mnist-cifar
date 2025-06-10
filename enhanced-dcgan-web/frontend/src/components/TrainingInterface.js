// File: frontend/src/components/TrainingInterface.js

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Play,
    Square,
    Settings,
    Database,
    Clock,
    TrendingUp,
    Brain,
    Zap,
    CheckCircle,
    AlertTriangle,
    RefreshCw,
    Download
} from 'lucide-react';
import toast from 'react-hot-toast';

import { useSystemStatus } from '../hooks/useSystemStatus';
import { useTrainingStatus } from '../hooks/useTrainingStatus';
import { useWebSocket } from '../hooks/useWebSocket';
import apiService from '../services/api';

const TrainingInterface = () => {
    const { systemStatus } = useSystemStatus();
    const {
        activeTrainings,
        addTraining,
        updateTraining,
        hasActiveTraining,
        getRunningTrainings,
        syncWithBackend
    } = useTrainingStatus();

    // Add WebSocket connection to listen for training updates
    const { lastMessage, isConnected } = useWebSocket();

    const [datasets, setDatasets] = useState({});
    const [checkpoints, setCheckpoints] = useState({});
    const [selectedDataset, setSelectedDataset] = useState('mnist');
    const [trainingConfig, setTrainingConfig] = useState({
        epochs: 50,
        resume_mode: 'interactive',
        experiment_name: ''
    });
    const [isStarting, setIsStarting] = useState(false);
    const [showAdvanced, setShowAdvanced] = useState(false);

    // Check for active trainings on component mount
    useEffect(() => {
        console.log('🔍 TrainingInterface: Checking for active trainings on mount');
        const runningTrainings = getRunningTrainings();
        if (runningTrainings.length > 0) {
            console.log('✅ Found persisted active trainings:', runningTrainings.length);

            // Show notification about resumed training monitoring
            toast.success(
                `Resumed monitoring ${runningTrainings.length} active training session${runningTrainings.length > 1 ? 's' : ''}`,
                {
                    duration: 6000,
                    icon: '🔄'
                }
            );

            // Sync with backend to verify status
            syncWithBackend();
        }
    }, [getRunningTrainings, syncWithBackend]);

    // Handle WebSocket messages for training progress - Enhanced for backend epoch correction
    useEffect(() => {
        if (lastMessage && lastMessage.type === 'training_status') {
            console.log('🔄 TrainingInterface: Processing training status update:', lastMessage);

            const { training_id, data } = lastMessage;

            if (training_id && data) {
                // Get existing training to check if it's a resumed session
                const existingTraining = activeTrainings[training_id];

                // NEW: Handle corrected backend data
                let actualCurrentEpoch = data.current_epoch || 0;
                let actualTotalEpochs = data.total_epochs || 50;
                let actualProgressPercentage = data.progress_percentage || 0;

                // If backend sends resumed training info, use it directly
                if (data.resumed_from_epoch !== undefined && data.is_resumed) {
                    console.log('✅ Backend provided resume info:', {
                        current: data.current_epoch,
                        resumed_from: data.resumed_from_epoch,
                        is_resumed: data.is_resumed
                    });

                    // Use backend's calculated values directly
                    actualCurrentEpoch = data.current_epoch;
                    actualProgressPercentage = data.progress_percentage;
                }
                // FALLBACK: Legacy inference for older backend versions
                else if (existingTraining && existingTraining.resumed_from_epoch) {
                    actualCurrentEpoch = (data.current_epoch || 0) + existingTraining.resumed_from_epoch;
                    actualProgressPercentage = (actualCurrentEpoch / actualTotalEpochs) * 100;
                }
                // If this looks like a resumed session (low epoch but high progress %), try to infer
                else if (!existingTraining?.resumed_from_epoch &&
                    data.current_epoch < 10 &&
                    data.progress_percentage > 50) {

                    const inferredResumedFrom = Math.floor((data.progress_percentage / 100) * actualTotalEpochs) - data.current_epoch;
                    if (inferredResumedFrom > 0) {
                        actualCurrentEpoch = data.current_epoch + inferredResumedFrom;
                        console.log(`🔄 Inferred resumed session: epoch ${data.current_epoch} + ${inferredResumedFrom} = ${actualCurrentEpoch}`);
                    }
                }

                // Update training status in the UI
                updateTraining(training_id, {
                    current_epoch: actualCurrentEpoch,
                    total_epochs: actualTotalEpochs,
                    progress_percentage: actualProgressPercentage,
                    status: data.status || 'running',
                    metrics: data.metrics || {},
                    dataset: data.dataset,
                    last_update: new Date().toISOString(),
                    // Store backend resume info if provided
                    resumed_from_epoch: data.resumed_from_epoch,
                    is_resumed: data.is_resumed,
                    // Store original backend values for debugging
                    backend_current_epoch: data.current_epoch,
                    backend_progress_percentage: data.progress_percentage,
                    backend_relative_epoch: data.relative_epoch
                });

                console.log('✅ TrainingInterface: Updated training progress:', training_id, {
                    backend: `${data.current_epoch}/${data.total_epochs} (${data.progress_percentage?.toFixed(1)}%)`,
                    display: `${actualCurrentEpoch}/${actualTotalEpochs} (${actualProgressPercentage.toFixed(1)}%)`,
                    resumed_info: data.is_resumed ? `resumed from ${data.resumed_from_epoch}` : 'fresh training'
                });
            }
        }
    }, [lastMessage, updateTraining, activeTrainings]);

    useEffect(() => {
        loadTrainingData();
    }, []);

    useEffect(() => {
        if (selectedDataset) {
            loadCheckpoints(selectedDataset);
        }
    }, [selectedDataset]);

    const loadTrainingData = async () => {
        try {
            const datasetsData = await apiService.getDatasets();
            setDatasets(datasetsData);

            if (Object.keys(datasetsData).length > 0 && !selectedDataset) {
                setSelectedDataset(Object.keys(datasetsData)[0]);
            }
        } catch (error) {
            toast.error('Failed to load datasets');
        }
    };

    const loadCheckpoints = async (dataset) => {
        try {
            const checkpointsData = await apiService.getCheckpoints(dataset);
            console.log('Raw checkpoints response:', checkpointsData);

            // Extract the actual array from the response
            let checkpointsArray = [];
            if (checkpointsData && checkpointsData.checkpoints && Array.isArray(checkpointsData.checkpoints)) {
                checkpointsArray = checkpointsData.checkpoints;
            } else if (Array.isArray(checkpointsData)) {
                checkpointsArray = checkpointsData;
            }

            console.log('Processed checkpoints array:', checkpointsArray);

            setCheckpoints(prev => ({
                ...prev,
                [dataset]: checkpointsArray
            }));
        } catch (error) {
            console.error('Failed to load checkpoints:', error);
            setCheckpoints(prev => ({
                ...prev,
                [dataset]: []
            }));
        }
    };

    const handleStartTraining = async () => {
        if (!systemStatus?.dcgan_available) {
            toast.error('DCGAN modules not available');
            return;
        }

        setIsStarting(true);
        try {
            const config = {
                dataset: selectedDataset,
                epochs: trainingConfig.epochs,
                resume_mode: trainingConfig.resume_mode,
                experiment_name: trainingConfig.experiment_name || undefined
            };

            const result = await apiService.startTraining(config);

            // Initialize training in the state
            addTraining(result.training_id, {
                training_id: result.training_id,
                status: 'starting',
                dataset: selectedDataset,
                current_epoch: 0,
                total_epochs: trainingConfig.epochs,
                progress_percentage: 0,
                metrics: {},
                start_time: new Date().toISOString(),
                end_time: null,
                error_message: null,
                // Track if this is a resumed session
                resumed_from_epoch: trainingConfig.resume_mode !== 'fresh' ? null : 0,
                resume_mode: trainingConfig.resume_mode
            });

            toast.success('Training started successfully!');
            console.log('🎯 Training started with ID:', result.training_id);
        } catch (error) {
            toast.error(error.message || 'Failed to start training');
        } finally {
            setIsStarting(false);
        }
    };

    const handleStopTraining = async (trainingId) => {
        try {
            await apiService.stopTraining(trainingId);
            updateTraining(trainingId, { status: 'stopped' });
        } catch (error) {
            toast.error('Failed to stop training');
        }
    };

    const canStartTraining = () => {
        const hasActive = hasActiveTraining();
        const systemReady = systemStatus?.dcgan_available;
        const configValid = selectedDataset && trainingConfig.epochs > 0;

        console.log('🔍 canStartTraining check:', {
            hasActive,
            systemReady,
            configValid,
            isStarting,
            result: systemReady && configValid && !isStarting && !hasActive
        });

        return systemReady && configValid && !isStarting && !hasActive;
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                        Training Interface
                    </h1>
                    <p className="text-gray-600 dark:text-gray-400">
                        Configure and monitor GAN training sessions
                    </p>
                </div>
                <SystemStatusIndicator status={systemStatus} isConnected={isConnected} />
            </div>

            {/* Main Content */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Configuration Panel */}
                <div className="lg:col-span-1">
                    <TrainingConfigPanel
                        datasets={datasets}
                        selectedDataset={selectedDataset}
                        setSelectedDataset={setSelectedDataset}
                        trainingConfig={trainingConfig}
                        setTrainingConfig={setTrainingConfig}
                        checkpoints={checkpoints[selectedDataset] || []}
                        onStartTraining={handleStartTraining}
                        canStartTraining={canStartTraining()}
                        hasActiveTraining={hasActiveTraining()}
                        getRunningTrainings={getRunningTrainings}
                        isStarting={isStarting}
                        showAdvanced={showAdvanced}
                        setShowAdvanced={setShowAdvanced}
                        systemStatus={systemStatus}
                    />
                </div>

                {/* Training Monitoring */}
                <div className="lg:col-span-2">
                    <TrainingMonitoringPanel
                        activeTrainings={activeTrainings}
                        onStopTraining={handleStopTraining}
                    />
                </div>
            </div>

            {/* Training History */}
            <TrainingHistoryPanel
                checkpoints={checkpoints}
                datasets={datasets}
                onLoadCheckpoint={loadCheckpoints}
            />

            {/* WebSocket Debug Panel - Enhanced */}
            {process.env.NODE_ENV === 'development' && (
                <DebugWebSocketMessages
                    lastMessage={lastMessage}
                    isConnected={isConnected}
                    activeTrainings={activeTrainings}
                />
            )}
        </div>
    );
};

// Training Configuration Panel
const TrainingConfigPanel = ({
                                 datasets,
                                 selectedDataset,
                                 setSelectedDataset,
                                 trainingConfig,
                                 setTrainingConfig,
                                 checkpoints,
                                 onStartTraining,
                                 canStartTraining,
                                 hasActiveTraining,
                                 getRunningTrainings,
                                 isStarting,
                                 showAdvanced,
                                 setShowAdvanced,
                                 systemStatus
                             }) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700"
        >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Training Configuration
            </h3>

            <div className="space-y-4">
                {/* Dataset Selection */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Dataset
                    </label>
                    <select
                        value={selectedDataset}
                        onChange={(e) => setSelectedDataset(e.target.value)}
                        className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                        {Object.entries(datasets).map(([key, dataset]) => (
                            <option key={key} value={key}>
                                {dataset.name} ({dataset.available_checkpoints} checkpoints)
                            </option>
                        ))}
                    </select>
                </div>

                {/* Epochs */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Training Epochs
                    </label>
                    <input
                        type="number"
                        min="1"
                        max="1000"
                        value={trainingConfig.epochs}
                        onChange={(e) => setTrainingConfig(prev => ({
                            ...prev,
                            epochs: parseInt(e.target.value) || 1
                        }))}
                        className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                </div>

                {/* Resume Mode */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Resume Mode
                    </label>
                    <select
                        value={trainingConfig.resume_mode}
                        onChange={(e) => setTrainingConfig(prev => ({
                            ...prev,
                            resume_mode: e.target.value
                        }))}
                        className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                        <option value="fresh">Fresh Training</option>
                        <option value="latest">Resume Latest</option>
                        <option value="interactive">Interactive Resume</option>
                    </select>
                </div>

                {/* Advanced Options */}
                <div>
                    <button
                        onClick={() => setShowAdvanced(!showAdvanced)}
                        className="flex items-center space-x-2 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
                    >
                        <Settings className="w-4 h-4" />
                        <span>Advanced Options</span>
                    </button>
                </div>

                <AnimatePresence>
                    {showAdvanced && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="space-y-4"
                        >
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Experiment Name (Optional)
                                </label>
                                <input
                                    type="text"
                                    value={trainingConfig.experiment_name}
                                    onChange={(e) => setTrainingConfig(prev => ({
                                        ...prev,
                                        experiment_name: e.target.value
                                    }))}
                                    placeholder="e.g., enhanced_training_v2"
                                    className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                />
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Available Checkpoints Info */}
                {Array.isArray(checkpoints) && checkpoints.length > 0 && (
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                        <div className="flex items-center space-x-2 text-blue-600 dark:text-blue-400 mb-2">
                            <Database className="w-4 h-4" />
                            <span className="text-sm font-medium">Available Checkpoints</span>
                        </div>
                        <p className="text-sm text-blue-600 dark:text-blue-400">
                            {checkpoints.length} saved model{checkpoints.length !== 1 ? 's' : ''} found
                        </p>
                    </div>
                )}

                {/* Start Training Button */}
                <motion.button
                    whileHover={{ scale: canStartTraining ? 1.02 : 1 }}
                    whileTap={{ scale: canStartTraining ? 0.98 : 1 }}
                    onClick={onStartTraining}
                    disabled={!canStartTraining}
                    className={`w-full flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-colors ${
                        canStartTraining
                            ? 'bg-blue-600 hover:bg-blue-700 text-white'
                            : 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                    }`}
                >
                    {isStarting ? (
                        <>
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                            >
                                <RefreshCw className="w-4 h-4" />
                            </motion.div>
                            <span>Starting Training...</span>
                        </>
                    ) : (
                        <>
                            <Play className="w-4 h-4" />
                            <span>Start Training</span>
                        </>
                    )}
                </motion.button>

                {/* Training Status Info */}
                {!canStartTraining && !isStarting && (
                    <div className="mt-3 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                        <div className="text-sm text-yellow-700 dark:text-yellow-300">
                            {!systemStatus?.dcgan_available ? (
                                <>⚠️ DCGAN system not available</>
                            ) : hasActiveTraining ? (
                                <>🏃 Training already in progress ({getRunningTrainings().length} active)</>
                            ) : (
                                <>⚙️ Check configuration</>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </motion.div>
    );
};

// Training Monitoring Panel
const TrainingMonitoringPanel = ({ activeTrainings, onStopTraining }) => {
    const trainingArray = Object.values(activeTrainings);

    // Debug logging
    console.log('🎯 TrainingMonitoringPanel: Rendering with trainings:', trainingArray);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700"
        >
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Training Monitoring
                </h3>
                <div className="text-sm text-gray-500">
                    {trainingArray.length} session{trainingArray.length !== 1 ? 's' : ''}
                </div>
            </div>

            {trainingArray.length === 0 ? (
                <EmptyTrainingState />
            ) : (
                <div className="space-y-4">
                    {trainingArray.map(training => (
                        <TrainingCard
                            key={training.training_id}
                            training={training}
                            onStop={() => onStopTraining(training.training_id)}
                        />
                    ))}
                </div>
            )}
        </motion.div>
    );
};

// Enhanced Training Card Component with better progress display
const TrainingCard = ({ training, onStop }) => {
    const getStatusIcon = (status) => {
        switch (status) {
            case 'running': return <Zap className="w-4 h-4 text-blue-500 animate-pulse" />;
            case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
            case 'error': return <AlertTriangle className="w-4 h-4 text-red-500" />;
            case 'starting': return <RefreshCw className="w-4 h-4 text-yellow-500 animate-spin" />;
            default: return <Clock className="w-4 h-4 text-gray-500" />;
        }
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'running': return 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-900/20';
            case 'completed': return 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20';
            case 'error': return 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20';
            case 'starting': return 'border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20';
            default: return 'border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-900/20';
        }
    };

    // Ensure safe number conversion
    const currentEpoch = parseInt(training.current_epoch) || 0;
    const totalEpochs = parseInt(training.total_epochs) || 1;
    const progressPercentage = parseFloat(training.progress_percentage) || 0;

    // Show resumed training information
    const isResumedSession = training.resume_mode && training.resume_mode !== 'fresh';
    const backendEpoch = training.backend_current_epoch;

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`border rounded-lg p-4 ${getStatusColor(training.status)}`}
        >
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                    {getStatusIcon(training.status)}
                    <span className="font-medium text-gray-900 dark:text-white">
                        {training.dataset?.toUpperCase() || 'Unknown'} Training
                    </span>
                    <span className="text-sm text-gray-500">
                        ({training.training_id?.slice(0, 8) || 'Unknown'})
                    </span>
                    {isResumedSession && (
                        <span className="text-xs bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400 px-2 py-1 rounded">
                            Resumed
                        </span>
                    )}
                </div>

                {(training.status === 'running' || training.status === 'starting') && (
                    <button
                        onClick={onStop}
                        className="flex items-center space-x-1 px-3 py-1 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
                    >
                        <Square className="w-3 h-3" />
                        <span className="text-sm">Stop</span>
                    </button>
                )}
            </div>

            {/* Progress Bar */}
            <div className="mb-3">
                <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-1">
                    <span>
                        Epoch {currentEpoch}/{totalEpochs}
                        {backendEpoch !== undefined && backendEpoch !== currentEpoch && (
                            <span className="text-xs text-gray-500 ml-1">
                                (backend: {backendEpoch})
                            </span>
                        )}
                    </span>
                    <span>{progressPercentage.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.min(progressPercentage, 100)}%` }}
                        transition={{ duration: 0.5 }}
                        className="bg-blue-600 h-2 rounded-full"
                    />
                </div>
            </div>

            {/* Resume Information */}
            {isResumedSession && (
                <div className="mb-3 p-2 bg-blue-50 dark:bg-blue-900/10 rounded text-sm">
                    <div className="text-blue-700 dark:text-blue-400">
                        📄 Resume Mode: {training.resume_mode}
                        {training.resumed_from_epoch > 0 && (
                            <span className="ml-2">• Started from epoch {training.resumed_from_epoch}</span>
                        )}
                    </div>
                </div>
            )}

            {/* Metrics */}
            {training.metrics && Object.keys(training.metrics).length > 0 && (
                <div className="grid grid-cols-2 gap-4 text-sm mb-2">
                    {Object.entries(training.metrics).map(([key, value]) => (
                        <div key={key}>
                            <span className="text-gray-500 dark:text-gray-400">{key}:</span>
                            <span className="ml-1 font-medium text-gray-900 dark:text-white">
                                {typeof value === 'number' ? value.toFixed(4) : value}
                            </span>
                        </div>
                    ))}
                </div>
            )}

            {/* Last Update Time */}
            {training.last_update && (
                <div className="text-xs text-gray-500">
                    Last update: {new Date(training.last_update).toLocaleTimeString()}
                </div>
            )}

            {/* Error Message */}
            {training.error_message && (
                <div className="mt-3 p-2 bg-red-100 dark:bg-red-900/20 rounded text-sm text-red-700 dark:text-red-400">
                    {training.error_message}
                </div>
            )}
        </motion.div>
    );
};

// Empty Training State
const EmptyTrainingState = () => {
    return (
        <div className="text-center py-12">
            <Brain className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                No Active Training Sessions
            </h4>
            <p className="text-gray-600 dark:text-gray-400">
                Configure and start a training session to see real-time monitoring here
            </p>
        </div>
    );
};

// Training History Panel
const TrainingHistoryPanel = ({ checkpoints, datasets, onLoadCheckpoint }) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700"
        >
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Training History
                </h3>
                <button
                    onClick={() => Object.keys(datasets).forEach(onLoadCheckpoint)}
                    className="flex items-center space-x-2 text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
                >
                    <RefreshCw className="w-4 h-4" />
                    <span>Refresh</span>
                </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(datasets).map(([datasetKey, dataset]) => (
                    <CheckpointList
                        key={datasetKey}
                        datasetKey={datasetKey}
                        dataset={dataset}
                        checkpoints={checkpoints[datasetKey] || []}
                    />
                ))}
            </div>
        </motion.div>
    );
};

// Fixed Checkpoint List Component
const CheckpointList = ({ datasetKey, dataset, checkpoints }) => {
    // Ensure checkpoints is always an array
    const checkpointsArray = Array.isArray(checkpoints) ? checkpoints : [];

    console.log('CheckpointList received:', { datasetKey, dataset, checkpoints, checkpointsArray });

    return (
        <div className="border border-gray-200 dark:border-gray-600 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
                <h4 className="font-medium text-gray-900 dark:text-white">
                    {dataset.name}
                </h4>
                <span className="text-sm text-gray-500">
                    {checkpointsArray.length} checkpoints
                </span>
            </div>

            {checkpointsArray.length === 0 ? (
                <div className="text-center py-4">
                    <Database className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                    <p className="text-sm text-gray-500">No checkpoints available</p>
                </div>
            ) : (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                    {checkpointsArray.slice(0, 5).map((checkpoint, index) => (
                        <CheckpointItem
                            key={checkpoint.filename || checkpoint.name || index}
                            checkpoint={checkpoint}
                            datasetKey={datasetKey}
                        />
                    ))}
                    {checkpointsArray.length > 5 && (
                        <div className="text-center py-2">
                            <span className="text-sm text-gray-500">
                                +{checkpointsArray.length - 5} more checkpoints
                            </span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

// Checkpoint Item Component
const CheckpointItem = ({ checkpoint, datasetKey }) => {
    const handleDownload = async () => {
        try {
            await apiService.downloadFile(
                `/api/checkpoints/${datasetKey}/${checkpoint.filename}`,
                checkpoint.filename
            );
        } catch (error) {
            toast.error('Failed to download checkpoint');
        }
    };

    return (
        <div className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700/50 rounded">
            <div className="flex-1">
                <div className="text-sm font-medium text-gray-900 dark:text-white">
                    Epoch {checkpoint.epoch}
                </div>
                <div className="text-xs text-gray-500">
                    {checkpoint.file_size_mb?.toFixed(1) || '0.0'} MB • {new Date(checkpoint.timestamp).toLocaleDateString()}
                </div>
            </div>

            <button
                onClick={handleDownload}
                className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                title="Download checkpoint"
            >
                <Download className="w-4 h-4" />
            </button>
        </div>
    );
};

// Enhanced System Status Indicator
const SystemStatusIndicator = ({ status, isConnected }) => {
    if (!status) return null;

    const isReady = status.status === 'online' && status.dcgan_available;

    return (
        <div className="flex items-center space-x-4">
            {/* WebSocket Status */}
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
                isConnected
                    ? 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-400'
                    : 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-400'
            }`}>
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="font-medium">
                    {isConnected ? 'Connected' : 'Disconnected'}
                </span>
            </div>

            {/* System Status */}
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
                isReady
                    ? 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-400'
                    : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-400'
            }`}>
                <div className={`w-2 h-2 rounded-full ${isReady ? 'bg-green-500' : 'bg-yellow-500'}`} />
                <span className="font-medium">
                    {isReady ? 'Ready for Training' : 'System Not Ready'}
                </span>
            </div>
        </div>
    );
};

// Enhanced Debug WebSocket Messages Component
const DebugWebSocketMessages = ({ lastMessage, isConnected, activeTrainings }) => {
    const [messages, setMessages] = useState([]);

    useEffect(() => {
        if (lastMessage) {
            setMessages(prev => [
                { ...lastMessage, timestamp: new Date().toISOString() },
                ...prev.slice(0, 9) // Keep last 10 messages
            ]);
        }
    }, [lastMessage]);

    if (process.env.NODE_ENV !== 'development') {
        return null;
    }

    return (
        <div className="fixed bottom-4 left-4 bg-gray-900 text-white p-4 rounded-lg max-w-md max-h-96 overflow-y-auto text-xs z-50">
            <h4 className="font-bold mb-2 text-yellow-400">
                WebSocket Debug ({isConnected ? 'Connected' : 'Disconnected'})
            </h4>

            {/* Active Trainings Count */}
            <div className="mb-2 p-2 bg-gray-800 rounded">
                <div className="text-green-400">Active Trainings: {Object.keys(activeTrainings).length}</div>
                {Object.entries(activeTrainings).map(([id, training]) => (
                    <div key={id} className="text-blue-400 text-xs">
                        {id.slice(0, 8)}: {training.status} - Epoch {training.current_epoch}/{training.total_epochs} ({training.progress_percentage?.toFixed(1)}%)
                        {training.backend_current_epoch !== undefined && (
                            <span className="text-yellow-400"> [Backend: {training.backend_current_epoch}]</span>
                        )}
                    </div>
                ))}
            </div>

            {messages.length === 0 ? (
                <div className="text-gray-400">No messages received yet...</div>
            ) : (
                messages.map((msg, index) => (
                    <div key={index} className="mb-2 p-2 bg-gray-800 rounded">
                        <div className="text-yellow-400">Type: {msg.type || 'unknown'}</div>
                        <div className="text-green-400">Time: {new Date(msg.timestamp).toLocaleTimeString()}</div>
                        {msg.training_id && (
                            <div className="text-purple-400">Training ID: {msg.training_id.slice(0, 8)}</div>
                        )}
                        {msg.data && (
                            <div className="text-blue-400">
                                Epoch: {msg.data.current_epoch}/{msg.data.total_epochs} ({msg.data.progress_percentage?.toFixed(1)}%)
                            </div>
                        )}
                        <div className="text-gray-400 mt-1">
                            Raw: {JSON.stringify(msg, null, 2).slice(0, 200)}
                            {JSON.stringify(msg, null, 2).length > 200 && '...'}
                        </div>
                    </div>
                ))
            )}
        </div>
    );
};

export default TrainingInterface;