# Fine-Tuning Comparison: LoRA vs MeZO+LoRA

## 1. LoRA Fine-Tuning Performance

### Final Accuracy
- Achieved **81.86%** validation accuracy on MRPC
- Trained only **0.62% of parameters** (2.2M of 357.6M total)

### Training Dynamics
- Accuracy progression: **68.38%** (step 100) → **81.86%** (step 1000)
- Loss decreased from **0.675** → **0.438** (stable convergence)

### Resource Usage
- Training time: **93 minutes** (1000 steps @ ~3.98 sec/step)
- GPU memory: **12-15GB** (batch_size=4)

## 2. MeZO + LoRA Observations

### Loss Behavior (First 100 Steps)
| Step | Loss+ (perturbed) | Loss- (original) |
|------|-------------------|------------------|
| 0    | 0.5665            | 0.5673           |
| 60   | 0.9543            | 0.9394           |
| 90   | 0.5652            | 0.5662           |

### Preliminary Findings
- Requires **more steps for stabilization** vs standard LoRA
- Full accuracy metrics pending (training stopped at 100 steps)

## 3. Comparative Analysis

| Metric               | LoRA               | MeZO+LoRA         |
|----------------------|--------------------|-------------------|
| **Accuracy**         | 81.86%             | Pending           |
| **Training Time**    | 93 minutes         | Expected longer   |
| **GPU Memory**       | 12-15GB            | Potentially lower* |
| **Stability**        | High               | Initial fluctuations |

_*MeZO uses forward passes only (no backpropagation)_

## 4. Recommendations

### For MeZO+LoRA:
1. Extend training to **2000-3000 steps**
2. Measure final accuracy
3. Hyperparameter tuning:
   - Perturbation size (ε)
   - Learning rate

### Key Tradeoffs
| Advantage          | Disadvantage               |
|--------------------|----------------------------|
| ✅ Memory efficient | ❌ Slower convergence      |
|                    | ❌ Hyperparameter sensitive |

## Conclusion
- LoRA demonstrated **reliable convergence**
- MeZO+LoRA shows potential but requires:
  - Extended training
  - Stability improvements
  - Full metric evaluation

## Terminology
| Term            | Description                          |
|-----------------|--------------------------------------|
| `Loss+/Loss-`   | Loss with ± parameter perturbations  |
| `grad_norm`     | Gradient magnitude                  |
| `eval_accuracy` | Validation set accuracy              |

**Next Steps:** Complete MeZO+LoRA training (1000+ steps) for direct comparison.
