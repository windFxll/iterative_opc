from dataclasses import dataclass
import time

import torch
from tqdm import tqdm

from ..config import OptimizationConfig
from ..resist.models import hard_resist, soft_resist
from ..utils.metrics import binary_iou
from .losses import boundary_weight_map, ilt_loss
from .parameterization import FilteredMask


@dataclass
class ILTResult:
    optimized_mask: torch.Tensor
    optimized_aerial: torch.Tensor
    optimized_printed: torch.Tensor
    history: list
    elapsed_seconds: float
    best_step: int
    steps_completed: int
    stop_reason: str


def _lerp(start: float, end: float, fraction: float) -> float:
    return start + (end - start) * fraction


class PixelILT:
    def __init__(self, imaging, optimization: OptimizationConfig):
        self.imaging = imaging
        self.config = optimization

    def optimize(self, target: torch.Tensor) -> ILTResult:
        cfg = self.config
        model = FilteredMask(target, cfg.mask_filter_sigma_px,
                             cfg.border_margin_px).to(target.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.iterations, eta_min=cfg.minimum_learning_rate)
        weights = boundary_weight_map(target, cfg.edge_weight,
                                      cfg.boundary_radius_px)
        history, best_iou, best_mask, best_step = [], -1.0, None, 0
        previous_hard_mask = None
        convergence_hits = 0
        stop_reason = "maximum iterations reached"
        steps_completed = 0
        start_time = time.perf_counter()

        progress = tqdm(range(cfg.iterations), desc="GPU ILT", unit="step", dynamic_ncols=True)
        for step in progress:
            hardening_steps = max(1, int(round(cfg.iterations * cfg.hardening_fraction)))
            fraction = min(step / hardening_steps, 1.0)
            mask_beta = _lerp(cfg.mask_beta_start, cfg.mask_beta_end, fraction)
            resist_beta = _lerp(cfg.resist_beta_start, cfg.resist_beta_end, fraction)

            optimizer.zero_grad(set_to_none=True)
            mask = model(mask_beta)
            aerial = self.imaging(mask)
            printed_soft = soft_resist(aerial, self.imaging.config.resist_threshold,
                                       resist_beta)
            loss, parts = ilt_loss(mask, printed_soft, target, weights,
                                   cfg.tv_weight, cfg.binary_weight, cfg.area_weight)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.gradient_clip))
            optimizer.step()
            scheduler.step()
            steps_completed = step + 1

            should_validate = (step % cfg.log_every == 0 or step == cfg.iterations - 1)
            if should_validate:
                with torch.no_grad():
                    hard_mask = (model(mask_beta) >= 0.5).float()
                    hard_aerial = self.imaging(hard_mask)
                    hard_printed = hard_resist(
                        hard_aerial, self.imaging.config.resist_threshold)
                    iou = binary_iou(hard_printed, target)
                    if previous_hard_mask is None:
                        mask_change = 1.0
                    else:
                        mask_change = float((hard_mask != previous_hard_mask).float().mean())
                    previous_hard_mask = hard_mask.detach().clone()
                    if iou > best_iou:
                        best_iou, best_step = iou, step
                        best_mask = hard_mask.detach().clone()
                row = {
                    "step": step, "loss": float(loss.detach()), "iou": iou,
                    "mask_beta": mask_beta, "resist_beta": resist_beta,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": grad_norm, "mask_change": mask_change,
                    **{name: float(value) for name, value in parts.items()},
                }
                history.append(row)
                progress.set_postfix(loss=f"{row['loss']:.5f}", iou=f"{iou:.5f}",
                                     change=f"{mask_change:.1e}")

                # 只在 continuation 完成、beta 固定后判断收敛。
                if (cfg.early_stop and step >= max(cfg.early_stop_min_step,
                                                    hardening_steps)
                        and len(history) >= cfg.early_stop_window):
                    recent = history[-cfg.early_stop_window:]
                    first_loss, last_loss = recent[0]["loss"], recent[-1]["loss"]
                    relative_loss = abs(first_loss - last_loss) / max(abs(first_loss), 1e-12)
                    ious = [item["iou"] for item in recent]
                    changes = [item["mask_change"] for item in recent]
                    converged = (
                        relative_loss < cfg.early_stop_relative_loss
                        and max(ious) - min(ious) < cfg.early_stop_iou_span
                        and max(changes) < cfg.early_stop_mask_change
                    )
                    convergence_hits = convergence_hits + 1 if converged else 0
                    if convergence_hits >= cfg.early_stop_patience:
                        stop_reason = (
                            "early convergence: fixed-beta loss/IoU/mask-change plateau"
                        )
                        progress.write(
                            f"Early stop at step {step}: rel_loss={relative_loss:.2e}, "
                            f"iou_span={max(ious)-min(ious):.2e}, "
                            f"mask_change_max={max(changes):.2e}")
                        break

        elapsed = time.perf_counter() - start_time
        with torch.no_grad():
            if best_mask is None:
                best_mask = (model(cfg.mask_beta_end) >= 0.5).float()
            aerial = self.imaging(best_mask)
            printed = hard_resist(aerial, self.imaging.config.resist_threshold)
        return ILTResult(best_mask, aerial, printed, history, elapsed, best_step,
                         steps_completed, stop_reason)

