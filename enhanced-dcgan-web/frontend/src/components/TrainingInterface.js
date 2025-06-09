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
import apiService from '../services/api';

const TrainingInterface = () => {
    const { systemStatus } = useSystemStatus();
    const { activeTrainings, addTraining, updateTraining } = useTrainingStatus();

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
            setCheckpoints(prev => ({
                ...prev,
                [dataset]: checkpointsData
            }));
        } catch (error) {
            console.error('Failed to load checkpoints:', error);
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
                error_message: null
            });

            toast.success('Training started successfully!');
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
        return systemStatus?.dcgan_available &&
            selectedDataset &&
            trainingConfig.epochs > 0 &&
            !isStarting &&
            Object.values(activeTrainings).filter(t => t.status === 'running').length === 0;
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
                <SystemStatusIndicator status={systemStatus} />
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
                        isStarting={isStarting}
                        showAdvanced={showAdvanced}
                        setShowAdvanced={setShowAdvanced}
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
                                 isStarting,
                                 showAdvanced,
                                 setShowAdvanced
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
                {checkpoints.length > 0 && (
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
            </div>
        </motion.div>
    );
};

// Training Monitoring Panel
const TrainingMonitoringPanel = ({ activeTrainings, onStopTraining }) => {
    const trainingArray = Object.values(activeTrainings);
    const runningTrainings = trainingArray.filter(t => t.status === 'running');

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700"
        >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Training Monitoring
            </h3>

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

// Training Card Component
const TrainingCard = ({ training, onStop }) => {
    const getStatusIcon = (status) => {
        switch (status) {
            case 'running': return <Zap className="w-4 h-4 text-blue-500" />;
            case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
            case 'error': return <AlertTriangle className="w-4 h-4 text-red-500" />;
            default: return <Clock className="w-4 h-4 text-gray-500" />;
        }
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'running': return 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-900/20';
            case 'completed': return 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20';
            case 'error': return 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20';
            default: return 'border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-900/20';
        }
    };

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
            {training.dataset.toUpperCase()} Training
          </span>
                    <span className="text-sm text-gray-500">
            ({training.training_id.slice(0, 8)})
          </span>
                </div>

                {training.status === 'running' && (
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
                    <span>Epoch {training.current_epoch}/{training.total_epochs}</span>
                    <span>{training.progress_percentage.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${training.progress_percentage}%` }}
                        className="bg-blue-600 h-2 rounded-full"
                    />
                </div>
            </div>

            {/* Metrics */}
            {Object.keys(training.metrics).length > 0 && (
                <div className="grid grid-cols-2 gap-4 text-sm">
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

// Checkpoint List Component
const CheckpointList = ({ datasetKey, dataset, checkpoints }) => {
    return (
        <div className="border border-gray-200 dark:border-gray-600 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
                <h4 className="font-medium text-gray-900 dark:text-white">
                    {dataset.name}
                </h4>
                <span className="text-sm text-gray-500">
          {checkpoints.length} checkpoints
        </span>
            </div>

            {checkpoints.length === 0 ? (
                <div className="text-center py-4">
                    <Database className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                    <p className="text-sm text-gray-500">No checkpoints available</p>
                </div>
            ) : (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                    {checkpoints.slice(0, 5).map((checkpoint, index) => (
                        <CheckpointItem
                            key={index}
                            checkpoint={checkpoint}
                            datasetKey={datasetKey}
                        />
                    ))}
                    {checkpoints.length > 5 && (
                        <div className="text-center py-2">
              <span className="text-sm text-gray-500">
                +{checkpoints.length - 5} more checkpoints
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
                    {checkpoint.file_size_mb.toFixed(1)} MB • {new Date(checkpoint.timestamp).toLocaleDateString()}
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

// System Status Indicator
const SystemStatusIndicator = ({ status }) => {
    if (!status) return null;

    const isReady = status.status === 'online' && status.dcgan_available;

    return (
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
    );
};

export default TrainingInterface;