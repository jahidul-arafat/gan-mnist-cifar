// File: frontend/src/components/Dashboard.js

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Activity,
    Brain,
    Database,
    Cpu,
    HardDrive,
    Zap,
    TrendingUp,
    CheckCircle,
    AlertTriangle,
    Clock,
    BarChart3,
    FileText
} from 'lucide-react';

import { useSystemStatus } from '../hooks/useSystemStatus';
import { useTrainingStatus } from '../hooks/useTrainingStatus';
import apiService from '../services/api';

const Dashboard = () => {
    const { systemStatus, isLoading, error } = useSystemStatus();
    const { activeTrainings } = useTrainingStatus();
    const [datasets, setDatasets] = useState({});
    const [recentActivity, setRecentActivity] = useState([]);

    useEffect(() => {
        loadDashboardData();
    }, []);

    const loadDashboardData = async () => {
        try {
            const [datasetsData] = await Promise.all([
                apiService.getDatasets()
            ]);
            setDatasets(datasetsData);
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        }
    };

    if (isLoading) {
        return <DashboardSkeleton />;
    }

    if (error) {
        return <ErrorState error={error} onRetry={loadDashboardData} />;
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                        Dashboard
                    </h1>
                    <p className="text-gray-600 dark:text-gray-400">
                        Enhanced DCGAN System Overview
                    </p>
                </div>
                <div className="flex items-center space-x-4">
                    <SystemHealthBadge status={systemStatus?.status} />
                    <RefreshButton onClick={loadDashboardData} />
                </div>
            </div>

            {/* System Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <SystemStatusCard
                    title="System Status"
                    value={systemStatus?.status || 'Unknown'}
                    icon={Activity}
                    color="blue"
                    subtitle={`${systemStatus?.device_type || 'Unknown'} Device`}
                />

                <SystemStatusCard
                    title="Available Datasets"
                    value={systemStatus?.available_datasets?.length || 0}
                    icon={Database}
                    color="green"
                    subtitle="Ready for training"
                />

                <SystemStatusCard
                    title="Model Checkpoints"
                    value={systemStatus?.total_checkpoints || 0}
                    icon={Brain}
                    color="purple"
                    subtitle="Saved models"
                />

                <SystemStatusCard
                    title="Active Training"
                    value={Object.keys(activeTrainings).length}
                    icon={Zap}
                    color="orange"
                    subtitle="Running sessions"
                />
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* System Information */}
                <SystemInfoPanel systemStatus={systemStatus} />

                {/* Training Overview */}
                <TrainingOverviewPanel activeTrainings={activeTrainings} />
            </div>

            {/* Dataset Status */}
            <DatasetStatusPanel datasets={datasets} />

            {/* Recent Activity */}
            <RecentActivityPanel activities={recentActivity} />
        </div>
    );
};

// System Status Card Component
const SystemStatusCard = ({ title, value, icon: Icon, color, subtitle }) => {
    const colorClasses = {
        blue: 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400',
        green: 'bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400',
        purple: 'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
        orange: 'bg-orange-50 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400',
        red: 'bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400'
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700"
        >
            <div className="flex items-center">
                <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
                    <Icon className="w-6 h-6" />
                </div>
                <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                        {title}
                    </p>
                    <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                        {value}
                    </p>
                    {subtitle && (
                        <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                            {subtitle}
                        </p>
                    )}
                </div>
            </div>
        </motion.div>
    );
};

// System Information Panel
const SystemInfoPanel = ({ systemStatus }) => {
    const deviceInfo = [
        {
            label: 'Device Type',
            value: systemStatus?.device_type?.toUpperCase() || 'Unknown',
            icon: Cpu
        },
        {
            label: 'Device Name',
            value: systemStatus?.device_name || 'Unknown',
            icon: HardDrive
        },
        {
            label: 'DCGAN Available',
            value: systemStatus?.dcgan_available ? 'Yes' : 'No',
            icon: Brain,
            status: systemStatus?.dcgan_available ? 'success' : 'warning'
        },
        {
            label: 'Last Updated',
            value: systemStatus?.timestamp ?
                new Date(systemStatus.timestamp).toLocaleTimeString() : 'Unknown',
            icon: Clock
        }
    ];

    return (
        <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700"
        >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                System Information
            </h3>

            <div className="space-y-4">
                {deviceInfo.map((info, index) => (
                    <div key={index} className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                            <info.icon className="w-5 h-5 text-gray-400" />
                            <span className="text-sm text-gray-600 dark:text-gray-400">
                {info.label}
              </span>
                        </div>
                        <div className="flex items-center space-x-2">
              <span className="text-sm font-medium text-gray-900 dark:text-white">
                {info.value}
              </span>
                            {info.status && (
                                <div className={`w-2 h-2 rounded-full ${
                                    info.status === 'success' ? 'bg-green-500' : 'bg-yellow-500'
                                }`} />
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </motion.div>
    );
};

// Training Overview Panel
const TrainingOverviewPanel = ({ activeTrainings }) => {
    const trainingArray = Object.values(activeTrainings);
    const completedTrainings = trainingArray.filter(t => t.status === 'completed').length;
    const runningTrainings = trainingArray.filter(t => t.status === 'running').length;
    const errorTrainings = trainingArray.filter(t => t.status === 'error').length;

    return (
        <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700"
        >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Training Overview
            </h3>

            {trainingArray.length === 0 ? (
                <div className="text-center py-8">
                    <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-500 dark:text-gray-400">
                        No training sessions yet
                    </p>
                    <p className="text-sm text-gray-400 dark:text-gray-500">
                        Start your first training to see progress here
                    </p>
                </div>
            ) : (
                <div className="space-y-4">
                    <div className="grid grid-cols-3 gap-4">
                        <div className="text-center">
                            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                                {completedTrainings}
                            </div>
                            <div className="text-xs text-gray-500">Completed</div>
                        </div>
                        <div className="text-center">
                            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                                {runningTrainings}
                            </div>
                            <div className="text-xs text-gray-500">Running</div>
                        </div>
                        <div className="text-center">
                            <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                                {errorTrainings}
                            </div>
                            <div className="text-xs text-gray-500">Failed</div>
                        </div>
                    </div>

                    <div className="space-y-2">
                        {trainingArray.slice(0, 3).map((training, index) => (
                            <TrainingSessionCard key={training.training_id} training={training} />
                        ))}
                    </div>
                </div>
            )}
        </motion.div>
    );
};

// Training Session Card
const TrainingSessionCard = ({ training }) => {
    const getStatusColor = (status) => {
        switch (status) {
            case 'completed': return 'text-green-600 bg-green-100 dark:bg-green-900/20';
            case 'running': return 'text-blue-600 bg-blue-100 dark:bg-blue-900/20';
            case 'error': return 'text-red-600 bg-red-100 dark:bg-red-900/20';
            default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20';
        }
    };

    return (
        <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
            <div className="flex items-center space-x-3">
                <div className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(training.status)}`}>
                    {training.status}
                </div>
                <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                        {training.dataset.toUpperCase()}
                    </div>
                    <div className="text-xs text-gray-500">
                        Epoch {training.current_epoch}/{training.total_epochs}
                    </div>
                </div>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
                {training.progress_percentage.toFixed(1)}%
            </div>
        </div>
    );
};

// Dataset Status Panel
const DatasetStatusPanel = ({ datasets }) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700"
        >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Dataset Status
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(datasets).map(([key, dataset]) => (
                    <DatasetCard key={key} datasetKey={key} dataset={dataset} />
                ))}
            </div>
        </motion.div>
    );
};

// Dataset Card
const DatasetCard = ({ datasetKey, dataset }) => {
    return (
        <div className="border border-gray-200 dark:border-gray-600 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium text-gray-900 dark:text-white">
                    {dataset.name}
                </h4>
                <div className="flex items-center space-x-1">
                    <Database className="w-4 h-4 text-gray-400" />
                    <span className="text-sm text-gray-500">
            {dataset.available_checkpoints}
          </span>
                </div>
            </div>

            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                {dataset.description}
            </p>

            <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                    <span className="text-gray-500">Size:</span>
                    <span className="ml-1 text-gray-900 dark:text-white">
            {dataset.image_size}×{dataset.image_size}
          </span>
                </div>
                <div>
                    <span className="text-gray-500">Classes:</span>
                    <span className="ml-1 text-gray-900 dark:text-white">
            {dataset.num_classes}
          </span>
                </div>
            </div>
        </div>
    );
};

// Recent Activity Panel
const RecentActivityPanel = ({ activities }) => {
    // Mock data for demonstration
    const mockActivities = [
        {
            id: 1,
            type: 'training_started',
            message: 'Training started for MNIST dataset',
            timestamp: new Date().toISOString(),
            icon: Brain
        },
        {
            id: 2,
            type: 'checkpoint_saved',
            message: 'Checkpoint saved at epoch 25',
            timestamp: new Date(Date.now() - 300000).toISOString(),
            icon: CheckCircle
        },
        {
            id: 3,
            type: 'report_generated',
            message: 'Academic report generated for CIFAR-10',
            timestamp: new Date(Date.now() - 600000).toISOString(),
            icon: FileText
        },
        {
            id: 4,
            type: 'generation_completed',
            message: 'Image generation completed',
            timestamp: new Date(Date.now() - 900000).toISOString(),
            icon: BarChart3
        }
    ];

    const displayActivities = activities.length > 0 ? activities : mockActivities;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700"
        >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Recent Activity
            </h3>

            <div className="space-y-4">
                {displayActivities.map((activity) => (
                    <ActivityItem key={activity.id} activity={activity} />
                ))}
            </div>
        </motion.div>
    );
};

// Activity Item Component
const ActivityItem = ({ activity }) => {
    const Icon = activity.icon;
    const timeAgo = getTimeAgo(activity.timestamp);

    return (
        <div className="flex items-center space-x-3">
            <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/50 rounded-lg flex items-center justify-center">
                    <Icon className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                </div>
            </div>
            <div className="flex-1 min-w-0">
                <p className="text-sm text-gray-900 dark:text-white">
                    {activity.message}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                    {timeAgo}
                </p>
            </div>
        </div>
    );
};

// System Health Badge Component
const SystemHealthBadge = ({ status }) => {
    const getStatusInfo = (status) => {
        switch (status) {
            case 'online':
                return { color: 'green', text: 'System Healthy', icon: CheckCircle };
            case 'degraded':
                return { color: 'yellow', text: 'Degraded Performance', icon: AlertTriangle };
            case 'offline':
                return { color: 'red', text: 'System Offline', icon: AlertTriangle };
            default:
                return { color: 'gray', text: 'Unknown Status', icon: Activity };
        }
    };

    const statusInfo = getStatusInfo(status);
    const Icon = statusInfo.icon;

    return (
        <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
            statusInfo.color === 'green' ? 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-400' :
                statusInfo.color === 'yellow' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-400' :
                    statusInfo.color === 'red' ? 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-400' :
                        'bg-gray-100 text-gray-800 dark:bg-gray-900/50 dark:text-gray-400'
        }`}>
            <Icon className="w-4 h-4" />
            <span className="font-medium">{statusInfo.text}</span>
        </div>
    );
};

// Refresh Button Component
const RefreshButton = ({ onClick }) => {
    const [isRefreshing, setIsRefreshing] = useState(false);

    const handleRefresh = async () => {
        setIsRefreshing(true);
        await onClick();
        setTimeout(() => setIsRefreshing(false), 1000);
    };

    return (
        <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
            <motion.div
                animate={isRefreshing ? { rotate: 360 } : { rotate: 0 }}
                transition={{ duration: 1, repeat: isRefreshing ? Infinity : 0, ease: "linear" }}
            >
                <Activity className="w-4 h-4" />
            </motion.div>
            <span className="text-sm font-medium">
        {isRefreshing ? 'Refreshing...' : 'Refresh'}
      </span>
        </motion.button>
    );
};

// Dashboard Skeleton Loader
const DashboardSkeleton = () => {
    return (
        <div className="space-y-6 animate-pulse">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-48 mb-2"></div>
                    <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-64"></div>
                </div>
                <div className="flex space-x-4">
                    <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-24"></div>
                    <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-20"></div>
                </div>
            </div>

            {/* Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {[...Array(4)].map((_, index) => (
                    <div key={index} className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                        <div className="flex items-center">
                            <div className="w-12 h-12 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
                            <div className="ml-4 space-y-2">
                                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-20"></div>
                                <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-16"></div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Main Panels */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {[...Array(2)].map((_, index) => (
                    <div key={index} className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                        <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-32 mb-4"></div>
                        <div className="space-y-4">
                            {[...Array(3)].map((_, i) => (
                                <div key={i} className="h-4 bg-gray-200 dark:bg-gray-700 rounded"></div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

// Error State Component
const ErrorState = ({ error, onRetry }) => {
    return (
        <div className="text-center py-12">
            <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                Failed to Load Dashboard
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
                {error.message || 'An unexpected error occurred'}
            </p>
            <button
                onClick={onRetry}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            >
                Try Again
            </button>
        </div>
    );
};

// Utility function for time ago
const getTimeAgo = (timestamp) => {
    const now = new Date();
    const time = new Date(timestamp);
    const diffInMinutes = Math.floor((now - time) / (1000 * 60));

    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
    return `${Math.floor(diffInMinutes / 1440)}d ago`;
};

export default Dashboard;