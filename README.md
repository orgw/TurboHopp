# TurboHopp: Accelerated Molecule Scaffold Hopping with Consistency Models

Official implementation of ["TurboHopp: Accelerated Molecule Scaffold Hopping with Consistency Models"](https://arxiv.org/abs/2410.20660).

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{turbohopp2024,
  title={TurboHopp: Accelerated Molecule Scaffold Hopping with Consistency Models},
  author={Yoo, Kiwoong and Oertell, Owen and Lee, Junhyun and Lee, Sanghoon and Kang, Jaewoo}, # Add authors
  journal={arXiv preprint arXiv:2410.20660},
  year={2024}
}
```
## Repository Structure
```
├── configs/           # Configuration files for training and evaluation
├── consistency/       # Consistency model implementation
├── diffusion_hopping/ # Core modules adapted from DiffHopp
├── utils/            # Helper functions
├── train_consistency.py    # Train consistency model
├── train_rlcm.py          # Train with RL for optimization
└── evaluate_consistency.py # Evaluation script
```

## Environment Setup

This code was developed and tested with:
- CUDA 11.8
- Python 3.9
- PyTorch 2.0.1

To install the environment:
```bash
conda env create -f environment.yml
conda activate turbohopp
```

## Usage

### Training Consistency Model
```bash
python train_consistency.py --config configs/train_config.yaml
```

### Training with RL
For a pretrained consistency model checkpoint:
```bash
python train_rlcm.py  --config configs/config_rlcm_docking.yaml
```

### Evaluation
Evaluate a trained model:
```bash
python evaluate_consistency.py \
    --checkpoint_path path/to/model.ckpt \
    --cuda_device 0 \
    --molecules_per_pocket 10 \
    --find_best \
    --batch_size 512 \
    --mode train \
    --dataset pdbbind_filtered
```


## Acknowledgments

This codebase builds upon [DiffHopp](https://github.com/jostorge/diffusion-hopping). We thank the authors for making their code available.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
