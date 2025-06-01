# Enhanced DCGAN Research Framework

A comprehensive framework for Enhanced Deep Convolutional Generative Adversarial Networks with academic reporting, advanced checkpointing, and production-ready features.

## Features

- **WGAN-GP Loss** with Gradient Penalty for stable training
- **Exponential Moving Average (EMA)** for improved sample quality
- **Enhanced Generator/Critic Architecture** with spectral normalization
- **Advanced Checkpointing** with auto-save every 5 epochs
- **Graceful Interrupt Handling** (Ctrl+C support)
- **Emergency Error Recovery** with crash-proof training
- **Academic Report Generation** with comprehensive analysis
- **Live Progress Tracking** with real-time monitoring
- **Multi-Device Support** (CUDA, Apple Metal MPS, CPU)
- **Interactive Image Generation** with prompts like "Draw me a 7"

## Installation

### From TestPyPI

```bash
pip install -i https://test.pypi.org/simple/ enhanced-dcgan-research
```

### Development Installation

```bash
git clone https://github.com/yourusername/enhanced-dcgan-research.git
cd enhanced-dcgan-research
pip install -e .
```

### With Optional Dependencies

```bash
pip install enhanced-dcgan-research[tensorboard,psutil]
```

## Quick Start

### Python API

```python
import enhanced_dcgan_research as edr

# Train Enhanced DCGAN
ema_generator, critic = edr.train_enhanced_gan('mnist', num_epochs=50)

# Generate academic report
reporter, report_path = edr.create_academic_report('mnist')

# Check package info
info = edr.get_info()
print(f"Device: {info['device_name']} ({info['device_type']})")
```

### Command Line Interface

```bash
# Train with interactive mode
enhanced-dcgan

# Train specific dataset
enhanced-dcgan --dataset mnist --epochs 50 --resume fresh

# Generate report
enhanced-dcgan --dataset cifar10 --report-only
```

## Supported Datasets

- **MNIST**: Handwritten digits (28x28 grayscale → 32x32)
- **CIFAR-10**: Natural images (32x32 RGB)

## Advanced Features

### Checkpoint Management

- Auto-save every 5 epochs
- Resume from any checkpoint
- Emergency saves on interrupts
- Graceful error recovery

### Academic Reporting

- Comprehensive training analysis
- Generated image documentation
- Statistical convergence analysis
- Publication-ready visualizations

### Device Optimization

- **Apple Silicon**: Metal Performance Shaders (MPS)
- **NVIDIA GPUs**: CUDA acceleration with cuDNN
- **CPU**: Multi-threaded optimization

## Examples

### Basic Training

```python
from enhanced_dcgan_research import train_enhanced_gan

# Train on MNIST for 25 epochs
ema_gen, critic = train_enhanced_gan('mnist', num_epochs=25, resume_mode='fresh')
```

### Academic Study

```python
from enhanced_dcgan_research import run_fixed_fully_integrated_academic_study

# Run complete academic study with report generation
reporter, report_path = run_fixed_fully_integrated_academic_study(
    dataset_choice='cifar10',
    num_epochs=100,
    resume_mode='interactive'
)
```

### Interactive Generation

```python
from enhanced_dcgan_research import InteractiveDigitGenerator

# Create interactive generator (after training)
interactive_gen = InteractiveDigitGenerator(ema_gen, 'mnist', config, device)
interactive_gen.start_interactive_session()
# Now you can type: "Draw me a 7", "Generate 3", etc.
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- CUDA-capable GPU (optional, but recommended)
- Apple Silicon Mac (optional, for MPS acceleration)

## Output Structure

```
reports/dataset/experiment_id/
├── comprehensive_academic_report.md    # Main report with embedded images
├── executive_summary.md                # Summary with statistics
├── generated_samples/                  # All generated images by epoch
│   ├── epoch_001/
│   ├── epoch_010/
│   └── ...
├── figures/                           # Training analysis plots
├── data/                             # CSV data and JSON metadata
└── interactive_generations/          # Interactive generation results
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{enhanced_dcgan_research,
  title={Enhanced DCGAN Research Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/enhanced-dcgan-research}
}
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **MPS not available**: Ensure you have Apple Silicon Mac with macOS 12.3+
3. **Import errors**: Install with `pip install -e .` for development

### Performance Tips

- Use CUDA/MPS for faster training
- Enable tensorboard for monitoring: `pip install tensorboard`
- Use `psutil` for memory monitoring: `pip install psutil`

## Changelog

### v0.1.0
- Initial release
- Complete DCGAN implementation with enhancements
- Academic reporting framework
- Multi-device support
- Advanced checkpointing system