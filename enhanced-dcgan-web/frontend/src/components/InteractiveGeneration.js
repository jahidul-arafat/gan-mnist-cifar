// File: frontend/src/components/InteractiveGeneration.js

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Zap,
    Settings,
    Download,
    RefreshCw,
    Image as ImageIcon,
    Sliders,
    AlertCircle,
    CheckCircle,
    Info
} from 'lucide-react';
import toast from 'react-hot-toast';
import apiService from '../services/api';

const InteractiveGeneration = () => {
    const [generationConfig, setGenerationConfig] = useState({
        dataset: 'mnist',
        numSamples: 8,
        noiseVector: 'random',
        seed: 42,
        prompt: ''
    });
    const [isGenerating, setIsGenerating] = useState(false);
    const [generatedImages, setGeneratedImages] = useState([]);
    const [lastGenerationId, setLastGenerationId] = useState(null);
    const [systemStatus, setSystemStatus] = useState(null);
    const [availableCheckpoints, setAvailableCheckpoints] = useState([]);
    const [selectedCheckpoint, setSelectedCheckpoint] = useState('latest');
    const [loadingCheckpoints, setLoadingCheckpoints] = useState(false);

    // Load system status and checkpoints on mount
    useEffect(() => {
        loadSystemInfo();
    }, []);

    // Load checkpoints when dataset changes
    useEffect(() => {
        if (generationConfig.dataset) {
            loadCheckpoints(generationConfig.dataset);
        }
    }, [generationConfig.dataset]);

    const loadSystemInfo = async () => {
        try {
            const status = await apiService.getSystemStatus();
            setSystemStatus(status);
            console.log('System status loaded:', status);
        } catch (error) {
            console.error('Failed to load system status:', error);
            toast.error('Failed to load system information');
        }
    };

    const loadCheckpoints = async (dataset) => {
        try {
            setLoadingCheckpoints(true);
            console.log(`Loading checkpoints for ${dataset}...`);

            const response = await apiService.getCheckpoints(dataset);
            console.log(`Checkpoints response for ${dataset}:`, response);

            // Handle different response formats - now should always get { checkpoints: array }
            let checkpoints = [];
            if (response && response.checkpoints && Array.isArray(response.checkpoints)) {
                checkpoints = response.checkpoints;
            } else if (Array.isArray(response)) {
                checkpoints = response;
            } else {
                console.warn('Unexpected checkpoint response format:', response);
                checkpoints = [];
            }

            console.log(`Processed checkpoints for ${dataset}:`, checkpoints);
            setAvailableCheckpoints(checkpoints);

            if (checkpoints.length > 0) {
                // Select the most recent checkpoint by default
                const latest = checkpoints.reduce((prev, current) => {
                    const prevEpoch = parseInt(prev.epoch) || 0;
                    const currentEpoch = parseInt(current.epoch) || 0;
                    return (prevEpoch > currentEpoch) ? prev : current;
                });

                console.log('Selected latest checkpoint:', latest);
                setSelectedCheckpoint(latest.filename || latest.name || 'latest');
            } else {
                setSelectedCheckpoint('latest');
                console.warn(`No checkpoints found for ${dataset}`);
            }
        } catch (error) {
            console.error(`Failed to load checkpoints for ${dataset}:`, error);
            setAvailableCheckpoints([]);
            setSelectedCheckpoint('latest');
            toast.error(`Failed to load checkpoints for ${dataset}`);
        } finally {
            setLoadingCheckpoints(false);
        }
    };

    const handleGenerate = async () => {
        console.log('Generate button clicked');
        console.log('System status:', systemStatus);
        console.log('Available checkpoints:', availableCheckpoints);

        if (!systemStatus?.dcgan_available) {
            toast.error('DCGAN system not available. Please check system status.');
            return;
        }

        if (availableCheckpoints.length === 0) {
            toast.error(`No trained checkpoints found for ${generationConfig.dataset}. Please train the model first.`);
            return;
        }

        setIsGenerating(true);
        setGeneratedImages([]); // Clear previous images

        try {
            console.log('Starting image generation with config:', {
                dataset: generationConfig.dataset,
                num_samples: generationConfig.numSamples,
                checkpoint: selectedCheckpoint,
                seed: generationConfig.seed
            });

            const generationRequest = {
                dataset: generationConfig.dataset,
                num_samples: generationConfig.numSamples,
                checkpoint_path: selectedCheckpoint !== 'latest' ? selectedCheckpoint : undefined,
                seed: generationConfig.seed,
                use_ema: true,
                device: systemStatus?.device_type?.toLowerCase() || 'auto'
            };

            // Add prompt if provided
            if (generationConfig.prompt && generationConfig.prompt.trim()) {
                generationRequest.prompt = generationConfig.prompt.trim();
            }

            console.log('Sending generation request:', generationRequest);

            const response = await apiService.generateImages(generationRequest);
            console.log('Generation response:', response);

            if (response && response.images && Array.isArray(response.images) && response.images.length > 0) {
                // Process and validate images
                const processedImages = response.images.map((image, index) => {
                    // Handle different response formats
                    let imageData = image;

                    if (typeof image === 'string') {
                        // If it's a base64 string, ensure proper format
                        if (image.startsWith('data:image/')) {
                            imageData = { url: image, id: `gen_${Date.now()}_${index}` };
                        } else if (image.startsWith('/9j/') || image.startsWith('iVBORw0KGgo')) {
                            // Raw base64 without header
                            imageData = {
                                url: `data:image/png;base64,${image}`,
                                id: `gen_${Date.now()}_${index}`
                            };
                        } else {
                            // Could be a URL or path
                            imageData = { url: image, id: `gen_${Date.now()}_${index}` };
                        }
                    } else if (image && typeof image === 'object') {
                        // Already an object, ensure it has required fields
                        imageData = {
                            ...image,
                            id: image.id || `gen_${Date.now()}_${index}`,
                            generated_at: image.generated_at || new Date().toISOString()
                        };
                    }

                    return imageData;
                });

                console.log('Processed images:', processedImages.length);
                setGeneratedImages(processedImages);
                setLastGenerationId(response.generation_id || Date.now().toString());

                toast.success(`Successfully generated ${processedImages.length} images!`, {
                    duration: 4000,
                    icon: '🎨'
                });
            } else {
                console.error('Invalid response format:', response);
                toast.error('No images were generated. Check if the model is properly trained.');
            }
        } catch (error) {
            console.error('Generation failed:', error);

            let errorMessage = 'Failed to generate images';
            if (error.message) {
                errorMessage += ': ' + error.message;
            }

            // Provide specific error guidance
            if (error.message?.includes('checkpoint') || error.message?.includes('model')) {
                errorMessage += '. Please ensure the model is trained and checkpoints exist.';
            } else if (error.message?.includes('connection') || error.message?.includes('network')) {
                errorMessage += '. Please check your connection to the backend server.';
            }

            toast.error(errorMessage, { duration: 6000 });
        } finally {
            setIsGenerating(false);
        }
    };

    const downloadImage = async (image, index) => {
        try {
            if (!image.url) {
                toast.error('Image data not available for download');
                return;
            }

            // Handle base64 images
            if (image.url.startsWith('data:image/')) {
                const link = document.createElement('a');
                link.href = image.url;
                link.download = `generated_${generationConfig.dataset}_${index + 1}.png`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                toast.success('Image downloaded successfully!');
            } else {
                // Handle URL-based images
                const response = await fetch(image.url);
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `generated_${generationConfig.dataset}_${index + 1}.png`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                window.URL.revokeObjectURL(url);
                toast.success('Image downloaded successfully!');
            }
        } catch (error) {
            console.error('Download failed:', error);
            toast.error('Failed to download image');
        }
    };

    const downloadAllImages = async () => {
        if (generatedImages.length === 0) {
            toast.error('No images to download');
            return;
        }

        try {
            toast.loading('Downloading all images...', { id: 'download-all' });

            for (let i = 0; i < generatedImages.length; i++) {
                await downloadImage(generatedImages[i], i);
                // Small delay between downloads to prevent overwhelming the browser
                await new Promise(resolve => setTimeout(resolve, 200));
            }

            toast.success(`Downloaded all ${generatedImages.length} images!`, { id: 'download-all' });
        } catch (error) {
            console.error('Batch download failed:', error);
            toast.error('Failed to download all images', { id: 'download-all' });
        }
    };

    const getPromptPlaceholder = () => {
        if (generationConfig.dataset === 'mnist') {
            return 'e.g., "Generate a digit 7" or "Show me a 3"';
        } else if (generationConfig.dataset === 'cifar10') {
            return 'e.g., "Generate a car" or "Show me an airplane"';
        }
        return 'Describe what you want to generate';
    };

    const getDatasetInfo = () => {
        if (generationConfig.dataset === 'mnist') {
            return 'Generates handwritten digits (0-9)';
        } else if (generationConfig.dataset === 'cifar10') {
            return 'Generates objects: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck';
        }
        return 'Select a dataset to see information';
    };

    // Check if generation should be enabled
    const canGenerate = systemStatus?.dcgan_available && availableCheckpoints.length > 0 && !loadingCheckpoints;

    console.log('Render state:', {
        systemStatus: systemStatus?.dcgan_available,
        checkpointsCount: availableCheckpoints.length,
        loadingCheckpoints,
        canGenerate,
        selectedCheckpoint
    });

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                    Interactive Generation
                </h1>
                <p className="text-gray-600 dark:text-gray-400">
                    Generate new images using trained DCGAN models
                </p>
            </div>

            {/* System Status Alert */}
            {systemStatus && (
                <div className={`p-4 rounded-lg border ${
                    systemStatus.dcgan_available
                        ? 'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800'
                        : 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800'
                }`}>
                    <div className="flex items-center space-x-2">
                        {systemStatus.dcgan_available ? (
                            <CheckCircle className="w-5 h-5 text-green-600" />
                        ) : (
                            <AlertCircle className="w-5 h-5 text-yellow-600" />
                        )}
                        <span className="font-medium">
                            {systemStatus.dcgan_available
                                ? `DCGAN Ready (${systemStatus.device_type})`
                                : 'DCGAN Not Available'}
                        </span>
                    </div>
                    {!systemStatus.dcgan_available && (
                        <p className="mt-1 text-sm text-yellow-700 dark:text-yellow-300">
                            Please ensure PyTorch and required dependencies are installed.
                        </p>
                    )}
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Generation Controls */}
                <div className="lg:col-span-1">
                    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                            Generation Settings
                        </h3>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Dataset
                                </label>
                                <select
                                    value={generationConfig.dataset}
                                    onChange={(e) => setGenerationConfig(prev => ({ ...prev, dataset: e.target.value }))}
                                    className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                                >
                                    <option value="mnist">MNIST</option>
                                    <option value="cifar10">CIFAR-10</option>
                                </select>
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                    {getDatasetInfo()}
                                </p>
                            </div>

                            {/* Checkpoint Selection */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Model Checkpoint
                                </label>
                                <select
                                    value={selectedCheckpoint}
                                    onChange={(e) => setSelectedCheckpoint(e.target.value)}
                                    className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                                    disabled={availableCheckpoints.length === 0 || loadingCheckpoints}
                                >
                                    <option value="latest">Latest Checkpoint</option>
                                    {availableCheckpoints.map((checkpoint) => (
                                        <option key={checkpoint.filename || checkpoint.name} value={checkpoint.filename || checkpoint.name}>
                                            Epoch {checkpoint.epoch} ({checkpoint.filename || checkpoint.name})
                                        </option>
                                    ))}
                                </select>
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                    {loadingCheckpoints ? (
                                        'Loading checkpoints...'
                                    ) : availableCheckpoints.length > 0 ? (
                                        `${availableCheckpoints.length} checkpoints available`
                                    ) : (
                                        'No checkpoints found - train the model first'
                                    )}
                                </p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Prompt (Optional)
                                </label>
                                <input
                                    type="text"
                                    value={generationConfig.prompt}
                                    onChange={(e) => setGenerationConfig(prev => ({ ...prev, prompt: e.target.value }))}
                                    placeholder={getPromptPlaceholder()}
                                    className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                    Describe what you want to generate
                                </p>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Number of Samples
                                </label>
                                <input
                                    type="range"
                                    min="1"
                                    max="16"
                                    value={generationConfig.numSamples}
                                    onChange={(e) => setGenerationConfig(prev => ({ ...prev, numSamples: parseInt(e.target.value) }))}
                                    className="w-full"
                                />
                                <div className="text-center text-sm text-gray-600 dark:text-gray-400 mt-1">
                                    {generationConfig.numSamples} images
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Random Seed
                                </label>
                                <input
                                    type="number"
                                    value={generationConfig.seed}
                                    onChange={(e) => setGenerationConfig(prev => ({ ...prev, seed: parseInt(e.target.value) || 42 }))}
                                    className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                                />
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                    Use same seed for reproducible results
                                </p>
                            </div>

                            <button
                                onClick={handleGenerate}
                                disabled={isGenerating || !canGenerate}
                                className={`w-full flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-colors ${
                                    isGenerating || !canGenerate
                                        ? 'bg-gray-400 cursor-not-allowed text-white'
                                        : 'bg-blue-600 hover:bg-blue-700 text-white'
                                }`}
                            >
                                {isGenerating ? (
                                    <>
                                        <RefreshCw className="w-4 h-4 animate-spin" />
                                        <span>Generating...</span>
                                    </>
                                ) : (
                                    <>
                                        <Zap className="w-4 h-4" />
                                        <span>Generate Images</span>
                                    </>
                                )}
                            </button>

                            {/* Debug Info */}
                            {process.env.NODE_ENV === 'development' && (
                                <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded text-xs">
                                    <p>Debug: System OK: {systemStatus?.dcgan_available ? 'Yes' : 'No'}</p>
                                    <p>Checkpoints: {availableCheckpoints.length}</p>
                                    <p>Loading: {loadingCheckpoints ? 'Yes' : 'No'}</p>
                                    <p>Can Generate: {canGenerate ? 'Yes' : 'No'}</p>
                                </div>
                            )}

                            {!canGenerate && (
                                <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                                    <div className="flex items-center space-x-2">
                                        <Info className="w-4 h-4 text-yellow-600" />
                                        <span className="text-sm text-yellow-700 dark:text-yellow-300">
                                            {!systemStatus?.dcgan_available
                                                ? 'DCGAN system not available'
                                                : loadingCheckpoints
                                                    ? 'Loading checkpoints...'
                                                    : 'No trained models found. Please train a model first.'
                                            }
                                        </span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Generated Images */}
                <div className="lg:col-span-2">
                    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                                Generated Images
                            </h3>
                            {generatedImages.length > 0 && (
                                <button
                                    onClick={downloadAllImages}
                                    className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                                >
                                    <Download className="w-4 h-4" />
                                    <span>Download All</span>
                                </button>
                            )}
                        </div>

                        {generatedImages.length === 0 ? (
                            <div className="text-center py-12">
                                <ImageIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                                <p className="text-gray-500 dark:text-gray-400 mb-2">
                                    No images generated yet
                                </p>
                                <p className="text-sm text-gray-400 dark:text-gray-500">
                                    Configure settings and click "Generate Images"
                                </p>
                            </div>
                        ) : (
                            <>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    {generatedImages.map((image, index) => (
                                        <GeneratedImageCard
                                            key={image.id || index}
                                            image={image}
                                            index={index}
                                            onDownload={() => downloadImage(image, index)}
                                        />
                                    ))}
                                </div>

                                {/* Generation Info */}
                                <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                                    <div className="text-sm text-gray-600 dark:text-gray-400">
                                        <p><strong>Dataset:</strong> {generationConfig.dataset.toUpperCase()}</p>
                                        <p><strong>Checkpoint:</strong> {selectedCheckpoint}</p>
                                        <p><strong>Prompt:</strong> {generationConfig.prompt || 'No prompt provided'}</p>
                                        <p><strong>Count:</strong> {generatedImages.length} images</p>
                                        <p><strong>Seed:</strong> {generationConfig.seed}</p>
                                        {lastGenerationId && (
                                            <p><strong>Generation ID:</strong> {lastGenerationId}</p>
                                        )}
                                    </div>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

// Generated Image Card Component
const GeneratedImageCard = ({ image, index, onDownload }) => {
    const [imageError, setImageError] = useState(false);
    const [imageLoading, setImageLoading] = useState(true);

    const handleImageLoad = () => {
        setImageLoading(false);
    };

    const handleImageError = (e) => {
        console.error('Image load error:', e.target.src);
        setImageError(true);
        setImageLoading(false);
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="relative group bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden"
        >
            {/* Image */}
            <div className="aspect-square relative">
                {imageLoading && (
                    <div className="absolute inset-0 flex items-center justify-center">
                        <RefreshCw className="w-6 h-6 text-gray-400 animate-spin" />
                    </div>
                )}

                {imageError ? (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400">
                        <AlertCircle className="w-8 h-8 mb-2" />
                        <span className="text-xs">Load Error</span>
                        <span className="text-xs mt-1">Check console</span>
                    </div>
                ) : (
                    <img
                        src={image.url}
                        alt={`Generated ${index + 1}`}
                        className="w-full h-full object-cover"
                        onLoad={handleImageLoad}
                        onError={handleImageError}
                        crossOrigin="anonymous"
                    />
                )}

                {/* Hover Overlay */}
                <div className="absolute inset-0 bg-black bg-opacity-50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <button
                        onClick={onDownload}
                        className="p-2 bg-white rounded-lg hover:bg-gray-100 transition-colors"
                        title="Download Image"
                    >
                        <Download className="w-4 h-4 text-gray-700" />
                    </button>
                </div>
            </div>

            {/* Image Info */}
            <div className="p-2">
                <div className="text-xs text-gray-600 dark:text-gray-400">
                    Image {index + 1}
                </div>
                {image.generated_at && (
                    <div className="text-xs text-gray-500 dark:text-gray-500">
                        {new Date(image.generated_at).toLocaleTimeString()}
                    </div>
                )}
            </div>
        </motion.div>
    );
};

export default InteractiveGeneration;