import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    FileText,
    Download,
    Eye,
    Plus,
    Calendar,
    User,
    Clock
} from 'lucide-react';

const ReportsPanel = () => {
    const [reports, setReports] = useState([]);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        // Mock data for now
        setTimeout(() => {
            setReports([
                {
                    id: 1,
                    title: 'MNIST Training Analysis',
                    description: 'Comprehensive analysis of MNIST training performance',
                    status: 'completed',
                    createdAt: new Date().toISOString(),
                    author: 'Enhanced DCGAN System',
                    type: 'academic'
                },
                {
                    id: 2,
                    title: 'CIFAR-10 Performance Report',
                    description: 'Performance metrics and convergence analysis',
                    status: 'generating',
                    createdAt: new Date().toISOString(),
                    author: 'Enhanced DCGAN System',
                    type: 'performance'
                }
            ]);
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
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                        Reports
                    </h1>
                    <p className="text-gray-600 dark:text-gray-400">
                        Academic and performance reports
                    </p>
                </div>
                <button className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
                    <Plus className="w-4 h-4" />
                    <span>Generate Report</span>
                </button>
            </div>

            <div className="grid grid-cols-1 gap-6">
                {reports.map(report => (
                    <ReportCard key={report.id} report={report} />
                ))}
            </div>
        </div>
    );
};

const ReportCard = ({ report }) => {
    const getStatusColor = (status) => {
        switch (status) {
            case 'completed':
                return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
            case 'generating':
                return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400';
            case 'failed':
                return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
            default:
                return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700"
        >
            <div className="flex items-start justify-between">
                <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                        <FileText className="w-5 h-5 text-gray-400" />
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                            {report.title}
                        </h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(report.status)}`}>
                            {report.status}
                        </span>
                    </div>
                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                        {report.description}
                    </p>
                    <div className="flex items-center space-x-4 text-sm text-gray-500">
                        <div className="flex items-center space-x-1">
                            <User className="w-4 h-4" />
                            <span>{report.author}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                            <Calendar className="w-4 h-4" />
                            <span>{new Date(report.createdAt).toLocaleDateString()}</span>
                        </div>
                    </div>
                </div>
                <div className="flex items-center space-x-2">
                    {report.status === 'completed' && (
                        <>
                            <button className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
                                <Eye className="w-4 h-4" />
                            </button>
                            <button className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
                                <Download className="w-4 h-4" />
                            </button>
                        </>
                    )}
                    {report.status === 'generating' && (
                        <div className="flex items-center space-x-2 text-blue-600">
                            <Clock className="w-4 h-4 animate-spin" />
                            <span className="text-sm">Generating...</span>
                        </div>
                    )}
                </div>
            </div>
        </motion.div>
    );
};

export default ReportsPanel;