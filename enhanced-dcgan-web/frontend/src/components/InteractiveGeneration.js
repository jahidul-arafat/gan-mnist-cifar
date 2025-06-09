import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
    Zap,
    Settings,
    Download,
    RefreshCw,
    Image as ImageIcon,
    Sliders
} from 'lucide-react';

const InteractiveGeneration = () => {
    const [generationConfig, setGenerationConfig] = useState({
        dataset: 'mnist',
        numSamples: 8,
        noiseVector: 'random',
        seed: 42
    });
    const [isGenerating, setIsGenerating] = useState(false);
    const [generatedImages, setGeneratedImages] = useState([]);

    const handleGenerate = async () => {
        setIsGenerating(true);
        // Mock generation process
        setTimeout(() => {
            const mockImages = Array.from({ length: generationConfig.numSamples }, (_, i) => ({
                id: i,
                url: `data:image/svg+xml,${encodeURIComponent(`
                    <svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
                        <rect width="64" height="64" fill="#${Math.floor(Math.random()*16777215).toString(16)}"/>
                        <text x="32" y="35" text-anchor="middle" fill="white" font-size="12">${i + 1}</text>
                    </svg>
                `)}`,
                generated_at: new Date().toISOString()
            }));
            setGeneratedImages(mockImages);
            setIsGenerating(false);
        }, 2000);
    };

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                    Interactive Generation
                </h1>
                <p className="text-gray-600 dark:text-gray-400">
                    Generate new images using trained models
                </p>
            </div>

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
                            </div>

                            <button
                                onClick={handleGenerate}
                                disabled={isGenerating}
                                className={`w-full flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium ${
                                    isGenerating
                                        ? 'bg-gray-400 cursor-not-allowed'
                                        : 'bg-blue-600 hover:bg-blue-700'
                                } text-white transition-colors`}
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
                                <button className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">
                                    <Download className="w-4 h-4" />
                                    <span>Download All</span>
                                </button>
                            )}
                        </div>

                        {generatedImages.length === 0 ? (
                            <div className="text-center py-12">
                                <ImageIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                                <p className="text-gray-500 dark:text-gray-400">
                                    No images generated yet
                                </p>
                                <p className="text-sm text-gray-400 dark:text-gray-500">
                                    Configure settings and click "Generate Images"
                                </p>
                            </div>
                        ) : (
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {generatedImages.map(image => (
                                    <motion.div
                                        key={image.id}
                                        initial={{ opacity: 0, scale: 0.8 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        className="relative group"
                                    >
                                        <img
                                            src={image.url}
                                            alt={`Generated ${image.id + 1}`}
                                            className="w-full h-24 object-cover rounded-lg border border-gray-200 dark:border-gray-600"
                                        />
                                        <div className="absolute inset-0 bg-black bg-opacity-50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
                                            <button className="p-2 bg-white rounded-lg hover:bg-gray-100">
                                                <Download className="w-4 h-4 text-gray-700" />
                                            </button>
                                        </div>
                                    </motion.div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default InteractiveGeneration;