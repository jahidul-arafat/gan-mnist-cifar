import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Terminal,
    Download,
    RefreshCw,
    Search,
    Filter,
    Calendar
} from 'lucide-react';

const LogsPanel = () => {
    const [logs, setLogs] = useState([]);
    const [filteredLogs, setFilteredLogs] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [logLevel, setLogLevel] = useState('all');

    useEffect(() => {
        // Mock data for now
        setTimeout(() => {
            const mockLogs = [
                {
                    id: 1,
                    timestamp: new Date().toISOString(),
                    level: 'info',
                    message: 'Training started for MNIST dataset',
                    dataset: 'mnist',
                    source: 'training'
                },
                {
                    id: 2,
                    timestamp: new Date(Date.now() - 300000).toISOString(),
                    level: 'debug',
                    message: 'Generator loss: 0.4521, Discriminator loss: 0.7892',
                    dataset: 'mnist',
                    source: 'training'
                },
                {
                    id: 3,
                    timestamp: new Date(Date.now() - 600000).toISOString(),
                    level: 'warning',
                    message: 'High memory usage detected: 85%',
                    dataset: 'cifar10',
                    source: 'system'
                },
                {
                    id: 4,
                    timestamp: new Date(Date.now() - 900000).toISOString(),
                    level: 'error',
                    message: 'Failed to load checkpoint: file not found',
                    dataset: 'cifar10',
                    source: 'checkpoint'
                }
            ];
            setLogs(mockLogs);
            setFilteredLogs(mockLogs);
            setIsLoading(false);
        }, 1000);
    }, []);

    useEffect(() => {
        let filtered = logs;

        if (searchTerm) {
            filtered = filtered.filter(log =>
                log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
                log.dataset.toLowerCase().includes(searchTerm.toLowerCase())
            );
        }

        if (logLevel !== 'all') {
            filtered = filtered.filter(log => log.level === logLevel);
        }

        setFilteredLogs(filtered);
    }, [logs, searchTerm, logLevel]);

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                        System Logs
                    </h1>
                    <p className="text-gray-600 dark:text-gray-400">
                        Training and system activity logs
                    </p>
                </div>
                <div className="flex items-center space-x-2">
                    <button className="flex items-center space-x-2 px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600">
                        <Download className="w-4 h-4" />
                        <span>Export</span>
                    </button>
                    <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                        <RefreshCw className="w-4 h-4" />
                        <span>Refresh</span>
                    </button>
                </div>
            </div>

            {/* Filters */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center space-x-4">
                    <div className="flex-1">
                        <div className="relative">
                            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                            <input
                                type="text"
                                placeholder="Search logs..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            />
                        </div>
                    </div>
                    <div>
                        <select
                            value={logLevel}
                            onChange={(e) => setLogLevel(e.target.value)}
                            className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        >
                            <option value="all">All Levels</option>
                            <option value="error">Error</option>
                            <option value="warning">Warning</option>
                            <option value="info">Info</option>
                            <option value="debug">Debug</option>
                        </select>
                    </div>
                </div>
            </div>

            {/* Logs Display */}
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                        Recent Logs ({filteredLogs.length})
                    </h3>
                </div>
                <div className="max-h-96 overflow-y-auto">
                    {filteredLogs.length === 0 ? (
                        <div className="p-8 text-center text-gray-500">
                            No logs found matching your criteria
                        </div>
                    ) : (
                        filteredLogs.map(log => (
                            <LogEntry key={log.id} log={log} />
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

const LogEntry = ({ log }) => {
    const getLevelColor = (level) => {
        switch (level) {
            case 'error':
                return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
            case 'warning':
                return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20 dark:text-yellow-400';
            case 'info':
                return 'text-blue-600 bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400';
            case 'debug':
                return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20 dark:text-gray-400';
            default:
                return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20 dark:text-gray-400';
        }
    };

    return (
        <div className="p-4 border-b border-gray-100 dark:border-gray-700 last:border-b-0 hover:bg-gray-50 dark:hover:bg-gray-700/50">
            <div className="flex items-start justify-between">
                <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getLevelColor(log.level)}`}>
                            {log.level.toUpperCase()}
                        </span>
                        <span className="text-sm text-gray-500">
                            {log.dataset} • {log.source}
                        </span>
                        <span className="text-sm text-gray-400">
                            {new Date(log.timestamp).toLocaleTimeString()}
                        </span>
                    </div>
                    <p className="text-sm text-gray-900 dark:text-white font-mono">
                        {log.message}
                    </p>
                </div>
            </div>
        </div>
    );
};

export default LogsPanel;
