# AttachedModuleForTransfuser

This project extends the TransFuser model with additional modules for adversarial attacking and waypoint prediction. It incorporates Vmamba and Feature Pyramid Network (FPN) architectures, as well as Deformable Attention mechanisms.

## Overview

The main model is based on the TransFuser architecture from [autonomousvision/transfuser](https://github.com/autonomousvision/transfuser). This project adds:

1. Adversarial attack methods
2. Vmamba integration
3. FPN (Feature Pyramid Network) implementation
4. Deformable Attention for improved waypoint prediction

## Key Features

- **Adversarial Attacking**: Implementation of attack methods on the main TransFuser model and other related models.
- **Vmamba**: Integration of the Vmamba architecture for enhanced performance.
- **FPN**: Utilization of Feature Pyramid Network for multi-scale feature representation.
- **Deformable Attention**: Application of deformable attention mechanisms for more flexible and adaptive waypoint prediction.

## References

This project draws inspiration and techniques from the following papers:

1. [Adversarial Attack Method](https://arxiv.org/abs/2402.11120)
2. [Vmamba: Visual State Space Models](https://arxiv.org/abs/2401.10166)
3. [Deformable Attention Network](https://arxiv.org/abs/2201.00520)

## Installation
Please consider install Vmamba locally as in its repository

## Next improvement 

Consider Jax for inner attention calculation to accelerate training and evaluation process.

## Acknowledgements

- Original TransFuser model by [autonomousvision](https://github.com/autonomousvision/transfuser)
- Authors of the referenced papers for their innovative techniques and approaches

- @article{Chitta2022PAMI,
  author = {Chitta, Kashyap and
            Prakash, Aditya and
            Jaeger, Bernhard and
            Yu, Zehao and
            Renz, Katrin and
            Geiger, Andreas},
  title = {TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving},
  journal = {Pattern Analysis and Machine Intelligence (PAMI)},
  year = {2022},
}

## License ##
This project is licensed under the Apache License 2.0 

>>>>>>> origin/main
