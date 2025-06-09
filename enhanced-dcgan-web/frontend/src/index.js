// File: frontend/src/index.js

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/index.css';

// Performance monitoring - updated for web-vitals v3+
import { onCLS, onINP, onFCP, onLCP, onTTFB } from 'web-vitals';

// Error boundary
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        this.setState({
            error: error,
            errorInfo: errorInfo
        });

        // Log error to console in development
        if (process.env.NODE_ENV === 'development') {
            console.error('Enhanced DCGAN Error:', error, errorInfo);
        }

        // In production, you might want to send errors to a logging service
        // logErrorToService(error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="min-h-screen bg-red-50 flex items-center justify-center">
                    <div className="text-center max-w-2xl p-8">
                        <div className="text-6xl mb-6">⚠️</div>
                        <h1 className="text-2xl font-bold text-red-800 mb-4">
                            Enhanced DCGAN Application Error
                        </h1>
                        <p className="text-red-600 mb-6">
                            Something went wrong with the application. This error has been logged.
                        </p>

                        {process.env.NODE_ENV === 'development' && (
                            <div className="bg-red-100 border border-red-300 rounded-lg p-4 mb-6 text-left">
                                <h3 className="text-red-800 font-semibold mb-2">Error Details:</h3>
                                <pre className="text-sm text-red-700 overflow-auto">
                  {this.state.error && this.state.error.toString()}
                </pre>
                                {this.state.errorInfo && (
                                    <details className="mt-4">
                                        <summary className="text-red-800 font-semibold cursor-pointer">
                                            Component Stack Trace
                                        </summary>
                                        <pre className="text-sm text-red-700 mt-2 overflow-auto">
                      {this.state.errorInfo.componentStack}
                    </pre>
                                    </details>
                                )}
                            </div>
                        )}

                        <div className="space-x-4">
                            <button
                                onClick={() => window.location.reload()}
                                className="bg-red-500 text-white px-6 py-2 rounded-lg hover:bg-red-600 transition-colors"
                            >
                                Reload Application
                            </button>

                            <button
                                onClick={() => {
                                    this.setState({ hasError: false, error: null, errorInfo: null });
                                }}
                                className="bg-gray-500 text-white px-6 py-2 rounded-lg hover:bg-gray-600 transition-colors"
                            >
                                Try Again
                            </button>
                        </div>

                        <div className="mt-8 text-sm text-gray-600">
                            <p>If this problem persists, please contact support.</p>
                            <p className="mt-2">
                                <a
                                    href="https://github.com/jahidul-arafat/gan-mnist-cifar"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-blue-600 hover:text-blue-800 underline"
                                >
                                    View Documentation
                                </a>
                            </p>
                        </div>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}

// Performance monitoring function
function sendToAnalytics(metric) {
    // In production, you might want to send metrics to an analytics service
    if (process.env.NODE_ENV === 'development') {
        console.log('Performance Metric:', metric);
    }

    // Example: send to analytics service
    // analytics.track('performance', metric);
}

// Performance observer - updated for web-vitals v3+
const observePerformance = () => {
    onCLS(sendToAnalytics);
    onINP(sendToAnalytics); // Replaces FID in newer versions
    onFCP(sendToAnalytics);
    onLCP(sendToAnalytics);
    onTTFB(sendToAnalytics);
};

// Initialize React application
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render application with error boundary
root.render(
    <React.StrictMode>
        <ErrorBoundary>
            <App />
        </ErrorBoundary>
    </React.StrictMode>
);

// Start performance monitoring
observePerformance();

// Service Worker registration - Only register if file exists
if ('serviceWorker' in navigator && process.env.NODE_ENV === 'production') {
    fetch('/sw.js')
        .then(() => {
            navigator.serviceWorker.register('/sw.js')
                .then((registration) => {
                    console.log('Service Worker registered successfully:', registration);
                })
                .catch((error) => {
                    console.log('Service Worker registration failed:', error);
                });
        })
        .catch(() => {
            console.log('Service Worker file not found, skipping registration');
        });
}

// Hot module replacement for development
if (process.env.NODE_ENV === 'development' && module.hot) {
    module.hot.accept('./App', () => {
        root.render(
            <React.StrictMode>
                <ErrorBoundary>
                    <App />
                </ErrorBoundary>
            </React.StrictMode>
        );
    });
}

// Global error handler for unhandled promises
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);

    // Prevent the default handling (which would show the error in console)
    event.preventDefault();

    // In production, you might want to send this to a logging service
    if (process.env.NODE_ENV === 'production') {
        // logErrorToService(event.reason);
    }
});

// Global error handler for uncaught exceptions
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);

    // In production, you might want to send this to a logging service
    if (process.env.NODE_ENV === 'production') {
        // logErrorToService(event.error);
    }
});

// Development helpers
if (process.env.NODE_ENV === 'development') {
    // Add development tools to window for debugging
    window.dcganDebug = {
        clearCache: () => {
            localStorage.clear();
            sessionStorage.clear();
            window.location.reload();
        },
        getStorageInfo: () => {
            return {
                localStorage: Object.keys(localStorage).reduce((acc, key) => {
                    acc[key] = localStorage.getItem(key);
                    return acc;
                }, {}),
                sessionStorage: Object.keys(sessionStorage).reduce((acc, key) => {
                    acc[key] = sessionStorage.getItem(key);
                    return acc;
                }, {})
            };
        },
        testApiConnection: async () => {
            try {
                const response = await fetch('/api/system/status');
                return await response.json();
            } catch (error) {
                return { error: error.message };
            }
        }
    };

    console.log('🔧 Development mode active');
    console.log('🛠️ Debug tools available at window.dcganDebug');
    console.log('📊 Performance monitoring enabled');
}