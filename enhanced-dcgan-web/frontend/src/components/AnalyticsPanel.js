import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    BarChart3,
    TrendingUp,
    Activity,
    Brain,
    Zap,
    Clock
} from 'lucide-react';

const AnalyticsPanel = () => {
    const [analyticsData, setAnalyticsData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        // Mock data for now
        setTimeout(() => {
            setAnalyticsData({
                trainingMetrics: {
                    totalTrainings: 15,
                    completedTrainings: 12,
                    averageEpochs: 45,
                    totalComputeTime: 120.5
                }
            });
            setIsLoading(false);
        }, 1000);
    }, []);

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                    Analytics
                </h1>
                <p className="text-gray-600 dark:text-gray-400">
                    Training metrics and performance analysis
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <AnalyticsCard
                    title="Total Trainings"
                    value={analyticsData?.trainingMetrics?.totalTrainings || 0}
                    icon={Brain}
                    color="blue"
                />
                <AnalyticsCard
                    title="Completed"
                    value={analyticsData?.trainingMetrics?.completedTrainings || 0}
                    icon={TrendingUp}
                    color="green"
                />
                <AnalyticsCard
                    title="Avg Epochs"
                    value={analyticsData?.trainingMetrics?.averageEpochs || 0}
                    icon={Activity}
                    color="purple"
                />
                <AnalyticsCard
                    title="Compute Hours"
                    value={`${analyticsData?.trainingMetrics?.totalComputeTime || 0}h`}
                    icon={Clock}
                    color="orange"
                />
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Performance Trends
                </h3>
                <div className="text-center py-12 text-gray-500">
                    Analytics charts will be implemented here
                </div>
            </div>
        </div>
    );
};

const AnalyticsCard = ({ title, value, icon: Icon, color }) => {
    const colorClasses = {
        blue: 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400',
        green: 'bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400',
        purple: 'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
        orange: 'bg-orange-50 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400'
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
                </div>
            </div>
        </motion.div>
    );
};

export default AnalyticsPanel;