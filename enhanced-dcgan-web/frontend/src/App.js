// File: frontend/src/App.js

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { motion } from 'framer-motion';

// Components
import Dashboard from './components/Dashboard';
import TrainingInterface from './components/TrainingInterface';
import InteractiveGeneration from './components/InteractiveGeneration';
import AnalyticsPanel from './components/AnalyticsPanel';
import ReportsPanel from './components/ReportsPanel';
import LogsPanel from './components/LogsPanel';
import { WebSocketProvider } from './components/WebSocketProvider';

// Services and Hooks
import { useSystemStatus } from './hooks/useSystemStatus';
import { useWebSocket } from './hooks/useWebSocket';

// Icons
import {
    Activity,
    Brain,
    BarChart3,
    FileText,
    Terminal,
    Settings,
    Wifi,
    WifiOff,
    AlertTriangle,
    Zap
} from 'lucide-react';

function App() {
    const [activeTab, setActiveTab] = useState('dashboard');
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [darkMode, setDarkMode] = useState(false);

    // System status and WebSocket connection
    const { systemStatus, isLoading: systemLoading, error: systemError } = useSystemStatus();
    const { isConnected } = useWebSocket();

    // Initialize theme
    useEffect(() => {
        const savedTheme = localStorage.getItem('dcgan-theme');
        if (savedTheme) {
            setDarkMode(savedTheme === 'dark');
        } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            setDarkMode(true);
        }
    }, []);

    // Apply theme changes
    useEffect(() => {
        document.documentElement.classList.toggle('dark', darkMode);
        localStorage.setItem('dcgan-theme', darkMode ? 'dark' : 'light');
    }, [darkMode]);

    // Navigation items
    const navigationItems = [
        {
            id: 'dashboard',
            label: 'Dashboard',
            icon: Activity,
            component: Dashboard,
            description: 'System overview and real-time status'
        },
        {
            id: 'training',
            label: 'Training',
            icon: Brain,
            component: TrainingInterface,
            description: 'Start and monitor GAN training'
        },
        {
            id: 'generation',
            label: 'Generate',
            icon: Zap,
            component: InteractiveGeneration,
            description: 'Interactive image generation'
        },
        {
            id: 'analytics',
            label: 'Analytics',
            icon: BarChart3,
            component: AnalyticsPanel,
            description: 'Training metrics and analysis'
        },
        {
            id: 'reports',
            label: 'Reports',
            icon: FileText,
            component: ReportsPanel,
            description: 'Academic research reports'
        },
        {
            id: 'logs',
            label: 'Logs',
            icon: Terminal,
            component: LogsPanel,
            description: 'System and training logs'
        }
    ];

    // Loading state
    if (systemLoading) {
        return <LoadingScreen />;
    }

    // Error state
    if (systemError) {
        return <ErrorScreen error={systemError} />;
    }

    return (
        <Router>
            <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200`}>
                {/* Sidebar */}
                <Sidebar
                    navigationItems={navigationItems}
                    activeTab={activeTab}
                    setActiveTab={setActiveTab}
                    sidebarOpen={sidebarOpen}
                    setSidebarOpen={setSidebarOpen}
                    systemStatus={systemStatus}
                    isConnected={isConnected}
                />

                {/* Main Content */}
                <div className={`transition-all duration-300 ${sidebarOpen ? 'ml-64' : 'ml-16'}`}>
                    {/* Header */}
                    <Header
                        darkMode={darkMode}
                        setDarkMode={setDarkMode}
                        sidebarOpen={sidebarOpen}
                        setSidebarOpen={setSidebarOpen}
                        activeTab={activeTab}
                        navigationItems={navigationItems}
                        isConnected={isConnected}
                        systemStatus={systemStatus}
                    />

                    {/* Page Content */}
                    <main className="p-6">
                        <Routes>
                            <Route path="/" element={<Navigate to="/dashboard" replace />} />
                            {navigationItems.map(item => {
                                const Component = item.component;
                                return (
                                    <Route
                                        key={item.id}
                                        path={`/${item.id}`}
                                        element={
                                            <PageWrapper>
                                                <Component />
                                            </PageWrapper>
                                        }
                                    />
                                );
                            })}
                        </Routes>
                    </main>
                </div>

                {/* Toast notifications */}
                <Toaster
                    position="top-right"
                    toastOptions={{
                        duration: 4000,
                        style: {
                            background: darkMode ? '#374151' : '#ffffff',
                            color: darkMode ? '#f9fafb' : '#111827',
                            border: darkMode ? '1px solid #4b5563' : '1px solid #e5e7eb'
                        }
                    }}
                />

                {/* Connection Status Indicator */}
                <ConnectionStatus isConnected={isConnected} />
            </div>
        </Router>
    );
}

// Sidebar Component
const Sidebar = ({
                     navigationItems,
                     activeTab,
                     setActiveTab,
                     sidebarOpen,
                     setSidebarOpen,
                     systemStatus,
                     isConnected
                 }) => {
    return (
        <motion.div
            initial={false}
            animate={{ width: sidebarOpen ? 256 : 64 }}
            className="fixed left-0 top-0 h-full bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 z-30"
        >
            <div className="flex flex-col h-full">
                {/* Logo */}
                <div className="flex items-center p-4 border-b border-gray-200 dark:border-gray-700">
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                        <Brain className="w-5 h-5 text-white" />
                    </div>
                    {sidebarOpen && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="ml-3"
                        >
                            <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
                                Enhanced DCGAN
                            </h1>
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                                v1.0.0
                            </p>
                        </motion.div>
                    )}
                </div>

                {/* Navigation */}
                <nav className="flex-1 p-4 space-y-2">
                    {navigationItems.map(item => (
                        <NavItem
                            key={item.id}
                            item={item}
                            isActive={activeTab === item.id}
                            onClick={() => setActiveTab(item.id)}
                            sidebarOpen={sidebarOpen}
                        />
                    ))}
                </nav>

                {/* System Status */}
                <div className="p-4 border-t border-gray-200 dark:border-gray-700">
                    <SystemStatusIndicator
                        systemStatus={systemStatus}
                        isConnected={isConnected}
                        sidebarOpen={sidebarOpen}
                    />
                </div>
            </div>
        </motion.div>
    );
};

// Navigation Item Component
const NavItem = ({ item, isActive, onClick, sidebarOpen }) => {
    const Icon = item.icon;

    return (
        <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onClick}
            className={`w-full flex items-center p-3 rounded-lg transition-colors ${
                isActive
                    ? 'bg-blue-50 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
        >
            <Icon className="w-5 h-5 flex-shrink-0" />
            {sidebarOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="ml-3 text-left"
                >
                    <div className="text-sm font-medium">{item.label}</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                        {item.description}
                    </div>
                </motion.div>
            )}
        </motion.button>
    );
};

// Header Component
const Header = ({
                    darkMode,
                    setDarkMode,
                    sidebarOpen,
                    setSidebarOpen,
                    activeTab,
                    navigationItems,
                    isConnected,
                    systemStatus
                }) => {
    const currentItem = navigationItems.find(item => item.id === activeTab);

    return (
        <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4">
            <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                    <button
                        onClick={() => setSidebarOpen(!sidebarOpen)}
                        className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
                    >
                        <Settings className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                    </button>

                    <div>
                        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                            {currentItem?.label || 'Enhanced DCGAN'}
                        </h2>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                            {currentItem?.description || 'Advanced GAN Training Platform'}
                        </p>
                    </div>
                </div>

                <div className="flex items-center space-x-4">
                    {/* System Health */}
                    <SystemHealthBadge systemStatus={systemStatus} />

                    {/* Connection Status */}
                    <div className="flex items-center space-x-2">
                        {isConnected ? (
                            <Wifi className="w-4 h-4 text-green-500" />
                        ) : (
                            <WifiOff className="w-4 h-4 text-red-500" />
                        )}
                        <span className="text-sm text-gray-600 dark:text-gray-400">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
                    </div>

                    {/* Theme Toggle */}
                    <button
                        onClick={() => setDarkMode(!darkMode)}
                        className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
                    >
                        {darkMode ? '🌙' : '☀️'}
                    </button>
                </div>
            </div>
        </header>
    );
};

// System Status Indicator
const SystemStatusIndicator = ({ systemStatus, isConnected, sidebarOpen }) => {
    const getStatusColor = () => {
        if (!isConnected) return 'text-red-500';
        if (!systemStatus?.dcgan_available) return 'text-yellow-500';
        return 'text-green-500';
    };

    const getStatusText = () => {
        if (!isConnected) return 'Disconnected';
        if (!systemStatus?.dcgan_available) return 'Limited';
        return 'Ready';
    };

    return (
        <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full ${getStatusColor()} bg-current`} />
            {sidebarOpen && (
                <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                        System {getStatusText()}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                        {systemStatus?.device_type || 'Unknown'} • {systemStatus?.total_checkpoints || 0} models
                    </div>
                </div>
            )}
        </div>
    );
};

// System Health Badge
const SystemHealthBadge = ({ systemStatus }) => {
    if (!systemStatus) return null;

    const isHealthy = systemStatus.status === 'online' && systemStatus.dcgan_available;

    return (
        <div className={`px-3 py-1 rounded-full text-xs font-medium ${
            isHealthy
                ? 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-400'
                : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-400'
        }`}>
            {isHealthy ? 'System Healthy' : 'System Issues'}
        </div>
    );
};

// Connection Status Indicator
const ConnectionStatus = ({ isConnected }) => {
    if (isConnected) return null;

    return (
        <div className="fixed bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4" />
            <span className="text-sm">Connection Lost</span>
        </div>
    );
};

// Page Wrapper with animations
const PageWrapper = ({ children }) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
        >
            {children}
        </motion.div>
    );
};

// Loading Screen
const LoadingScreen = () => {
    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <div className="text-center">
                <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    className="w-16 h-16 border-4 border-white/20 border-t-white rounded-full mx-auto mb-4"
                />
                <h2 className="text-xl font-semibold text-white mb-2">Enhanced DCGAN</h2>
                <p className="text-white/80">Loading neural networks...</p>
            </div>
        </div>
    );
};

// Error Screen
const ErrorScreen = ({ error }) => {
    return (
        <div className="min-h-screen bg-red-50 flex items-center justify-center">
            <div className="text-center max-w-md">
                <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                <h2 className="text-xl font-semibold text-red-800 mb-2">System Error</h2>
                <p className="text-red-600 mb-4">{error.message || 'Failed to connect to backend'}</p>
                <button
                    onClick={() => window.location.reload()}
                    className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600"
                >
                    Reload Application
                </button>
            </div>
        </div>
    );
};

export default App;