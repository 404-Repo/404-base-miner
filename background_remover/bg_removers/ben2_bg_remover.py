import gc
from PIL import Image

import torch
from background_remover.bg_removers.ben2_model.ben2 import BEN_Base
from background_remover.bg_removers.base_bg_remover import BaseBGRemover


class Ben2BGRemover(BaseBGRemover):
    def __init__(self, device, enable_optimize: bool = False):
        super().__init__()
        self._bg_remover: BEN_Base | None = None
        self._device = device
        self._enable_optimize = enable_optimize

    def load_model(self) -> None:
        self._bg_remover = BEN_Base.from_pretrained("PramaLLC/BEN2").to(self._device)
        self._bg_remover.eval().half()
        if self._enable_optimize:
            self.optimize()

    def unload_model(self) -> None:
        del self._bg_remover
        gc.collect()
        torch.cuda.empty_cache()
        self._bg_remover = None

    def remove_bg(self, image: Image) -> tuple[Image, bool]:
        result_image = self._bg_remover.inference(image, refine_foreground=False)
        # valid = self.is_image_valid(result_image)
        valid = True
        return result_image, valid
