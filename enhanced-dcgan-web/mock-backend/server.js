// File: mock-backend/server.js
// Simple mock backend for development

const express = require('express');
const cors = require('cors');
const app = express();
const PORT = 8000;

// Middleware
app.use(cors());
app.use(express.json());

// Mock data
const mockSystemStatus = {
    status: 'online',
    dcgan_available: true,
    device_type: 'cpu',
    device_name: 'Development Environment',
    available_datasets: ['mnist', 'cifar10'],
    total_checkpoints: 5,
    timestamp: new Date().toISOString()
};

const mockDatasets = {
    mnist: {
        name: 'MNIST',
        description: 'Handwritten digits dataset',
        image_size: 28,
        num_classes: 10,
        available_checkpoints: 3
    },
    cifar10: {
        name: 'CIFAR-10',
        description: 'Natural images dataset',
        image_size: 32,
        num_classes: 10,
        available_checkpoints: 2
    }
};

// Routes
app.get('/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

app.get('/api/system/status', (req, res) => {
    res.json(mockSystemStatus);
});

app.get('/api/datasets', (req, res) => {
    res.json(mockDatasets);
});

app.get('/api/checkpoints/:dataset', (req, res) => {
    const { dataset } = req.params;
    const mockCheckpoints = [
        {
            epoch: 50,
            filename: `${dataset}_epoch_50.pth`,
            timestamp: new Date().toISOString(),
            file_size_mb: 25.3
        },
        {
            epoch: 25,
            filename: `${dataset}_epoch_25.pth`,
            timestamp: new Date(Date.now() - 86400000).toISOString(),
            file_size_mb: 25.1
        }
    ];
    res.json(mockCheckpoints);
});

app.post('/api/training/start', (req, res) => {
    const trainingId = `training_${Date.now()}`;
    res.json({
        training_id: trainingId,
        status: 'starting',
        message: 'Training initiated successfully'
    });
});

app.get('/api/training/status/:id', (req, res) => {
    res.json({
        training_id: req.params.id,
        status: 'running',
        current_epoch: 15,
        total_epochs: 50,
        progress_percentage: 30,
        metrics: {
            generator_loss: 0.4521,
            discriminator_loss: 0.7892
        }
    });
});

app.post('/api/training/stop/:id', (req, res) => {
    res.json({
        training_id: req.params.id,
        status: 'stopped',
        message: 'Training stopped successfully'
    });
});

// Error handling
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something went wrong!' });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({ error: 'Not found' });
});

app.listen(PORT, () => {
    console.log(`Mock backend running on http://localhost:${PORT}`);
    console.log('Available endpoints:');
    console.log('  GET  /health');
    console.log('  GET  /api/system/status');
    console.log('  GET  /api/datasets');
    console.log('  GET  /api/checkpoints/:dataset');
    console.log('  POST /api/training/start');
    console.log('  GET  /api/training/status/:id');
    console.log('  POST /api/training/stop/:id');
});

module.exports = app;