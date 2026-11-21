| model | AUROC | AUPRC | sens. @ 95% spec. | folds |
|---|---|---|---|---|
| ResNet-18 (ImageNet, fine-tuned) | 0.901 ± 0.041 | 0.674 ± 0.130 | 0.613 ± 0.166 | 5 |
| EfficientNet-B0 (ImageNet, fine-tuned) | 0.918 ± 0.035 | 0.667 ± 0.175 | 0.598 ± 0.240 | 5 |
| ResNet-18 (random init) | 0.865 ± 0.017 | 0.527 ± 0.138 | 0.428 ± 0.121 | 5 |
| Frozen ImageNet ResNet-18 + LR | 0.805 ± 0.047 | 0.454 ± 0.134 | 0.358 ± 0.186 | 5 |