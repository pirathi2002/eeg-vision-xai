| model | chb01 | chb02 | chb03 | chb05 | chb08 | chb10 | chb20 | mean AUROC | mean AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-18, ImageNet init, fine-tuned | 0.979 | 0.962 | 0.719 | 0.978 | 0.863 | 0.960 | 0.644 | **0.872** | 0.758 |
| ResNet-18, random init | 0.978 | 0.905 | 0.609 | 0.888 | 0.882 | 0.886 | 0.783 | **0.847** | 0.694 |
| Frozen ImageNet features + logistic reg. | 0.856 | 0.836 | 0.659 | 0.821 | 0.670 | 0.769 | 0.604 | **0.745** | 0.519 |
