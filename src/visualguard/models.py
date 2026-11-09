from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import Sample, require_normal
from .errors import VisualGuardError
from .images import preprocess, read_image
from .utils import resolve_inside


@dataclass(frozen=True)
class ModelConfig:
    method: str = "patchcore"
    backbone: str = "resnet18.tv_in1k"
    input_size: int = 256
    coreset_ratio: float = 0.05
    max_train_patches: int = 20000
    max_embedding_mb: int = 1024
    batch_size: int = 4
    seed: int = 42
    num_neighbors: int = 9
    quantile: float = 0.99
    threads: int = 4
    pretrained: bool = True

    def __post_init__(self):
        if self.method not in {"global_knn", "patchcore"} or self.backbone != "resnet18.tv_in1k":
            raise VisualGuardError(
                "Supported methods: global_knn, patchcore; backbone: resnet18.tv_in1k."
            )
        if self.input_size not in {32, 64, 128, 224, 256}:
            raise VisualGuardError("Supported square input sizes: 32, 64, 128, 224, 256.")
        if not 0 < self.coreset_ratio <= 1 or not 0 < self.quantile < 1:
            raise VisualGuardError("Invalid coreset ratio or calibration quantile.")
        if not 1 <= self.batch_size <= 64 or not 1 <= self.threads <= 32:
            raise VisualGuardError("Batch size or CPU thread limit out of range.")
        if not 1 <= self.num_neighbors <= 32 or not 100 <= self.max_train_patches <= 1000000:
            raise VisualGuardError("Memory bank configuration out of range.")
        if not 16 <= self.max_embedding_mb <= 8192 or not 0 <= self.seed < 2**32:
            raise VisualGuardError("Resource limit or seed out of range.")

    def to_dict(self):
        return asdict(self)


def configure_torch(config: ModelConfig):
    import torch

    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True)


def nearest_distances(queries: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Small transparent reference implementation, used in correctness tests."""
    queries, bank = np.asarray(queries), np.asarray(bank)
    if queries.ndim != 2 or bank.ndim != 2 or not len(bank) or queries.shape[1] != bank.shape[1]:
        raise VisualGuardError("Incompatible embedding matrices or empty bank.")
    if not np.isfinite(queries).all() or not np.isfinite(bank).all():
        raise VisualGuardError("Embeddings must be finite.")
    return np.sqrt(((queries[:, None, :] - bank[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)


class Detector:
    def __init__(self, config: ModelConfig, *, initialize_pretrained: bool = False):
        import timm
        import torch

        self.config = config
        self.fit_details = {}
        self.bank = torch.empty(0)
        if config.method == "global_knn":
            self.model = timm.create_model(
                config.backbone, pretrained=initialize_pretrained, num_classes=0, global_pool="avg"
            )
        else:
            from anomalib.models.image.patchcore.torch_model import PatchcoreModel

            self.model = PatchcoreModel(
                layers=["layer2", "layer3"],
                backbone=config.backbone,
                pre_trained=initialize_pretrained,
                num_neighbors=config.num_neighbors,
            )
        self.model.requires_grad_(False).eval().cpu()

    def _tensor(self, images):
        import torch

        return torch.stack([preprocess(image, self.config.input_size) for image in images])

    def fit(self, samples: list[Sample], root: Path) -> None:
        import torch

        require_normal(samples, "train")
        unique = {s.pixels_sha256: s for s in samples}
        rows = sorted(unique.values(), key=lambda s: s.path)
        estimated_patches = len(rows) * (self.config.input_size // 8) ** 2
        estimated_mb = estimated_patches * 384 * 4 / 1024**2
        if self.config.method == "patchcore" and estimated_mb > self.config.max_embedding_mb:
            raise VisualGuardError(
                f"Embedding estimate {estimated_mb:.0f} MiB exceeds configured limit; reduce input size or use a larger memory budget."
            )
        collected = []
        if self.config.method == "patchcore":
            self.model.train()
            self.model.feature_extractor.eval()
        with torch.inference_mode():
            for start in range(0, len(rows), self.config.batch_size):
                batch = rows[start : start + self.config.batch_size]
                tensor = self._tensor([read_image(resolve_inside(root, s.path)) for s in batch])
                embeddings = self.model(tensor)
                if self.config.method == "global_knn":
                    collected.append(embeddings)
        if self.config.method == "global_knn":
            self.bank = torch.cat(collected).contiguous()
            candidates = len(self.bank)
        else:
            # Keep Anomalib's coreset and full image weighting; bound CPU candidate cost explicitly.
            embeddings = torch.cat(self.model.embedding_store)
            self.model.embedding_store.clear()
            candidates = len(embeddings)
            if candidates > self.config.max_train_patches:
                generator = torch.Generator().manual_seed(self.config.seed)
                indices = torch.randperm(candidates, generator=generator)[
                    : self.config.max_train_patches
                ]
                embeddings = embeddings[indices].contiguous()
            selected_count = int(len(embeddings) * self.config.coreset_ratio)
            if selected_count < self.config.num_neighbors:
                raise VisualGuardError(
                    "Coreset too small for PatchCore neighbors; increase training samples or coreset ratio."
                )
            self.model.embedding_store.append(embeddings)
            with torch.no_grad():
                self.model.subsample_embedding(sampling_ratio=self.config.coreset_ratio)
            self.bank = self.model.memory_bank
        self.model.eval()
        self.fit_details = {
            "unique_train_images": len(rows),
            "duplicate_train_images_removed": len(samples) - len(rows),
            "raw_candidates": candidates,
            "retained_candidates": min(candidates, self.config.max_train_patches)
            if self.config.method == "patchcore"
            else candidates,
            "bank_rows": len(self.bank),
            "bank_dimensions": self.bank.shape[1],
            "bank_bytes": self.bank.numel() * self.bank.element_size(),
        }

    def predict(self, images) -> tuple[np.ndarray, np.ndarray | None]:
        import torch

        if not images or len(images) > self.config.batch_size:
            raise VisualGuardError("Inference batch is empty or exceeds configured batch size.")
        with torch.inference_mode():
            result = self.model(self._tensor(images))
            if self.config.method == "global_knn":
                if self.bank.ndim != 2 or not len(self.bank):
                    raise VisualGuardError("Model has no fitted feature bank.")
                scores = torch.cdist(result, self.bank).min(dim=1).values.cpu().numpy()
                maps = None
            else:
                scores = result.pred_score.cpu().numpy().reshape(-1)
                maps = result.anomaly_map[:, 0].cpu().numpy()
        if (
            not np.isfinite(scores).all()
            or (scores < 0).any()
            or (maps is not None and not np.isfinite(maps).all())
        ):
            raise VisualGuardError("Model produced invalid scores; inspection must be reviewed.")
        return scores, maps

    def state_dict(self):
        result = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in self.model.state_dict().items()
        }
        if self.config.method == "global_knn":
            result["visualguard_bank"] = self.bank.detach().cpu().contiguous().clone()
        return result

    def load_state_dict(self, state):
        if self.config.method == "global_knn":
            self.bank = state.pop("visualguard_bank").clone()
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        if self.config.method == "patchcore":
            self.bank = self.model.memory_bank
        if self.bank.ndim != 2 or not len(self.bank):
            raise VisualGuardError("Artifact has an empty or malformed memory bank.")
