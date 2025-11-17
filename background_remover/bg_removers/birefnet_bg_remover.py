import gc
from PIL import Image

import torch
from torchvision import transforms
from background_remover.bg_removers.base_bg_remover import BaseBGRemover
from background_remover.bg_removers.birefnet_model.birefnet import BiRefNet


class BiRefNetBGRemover(BaseBGRemover):
    def __init__(self, device, enable_optimize: bool = False):
        super().__init__()
        self._bg_remover: BiRefNet | None = None
        self._device = device
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        self._transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._enable_optimize = enable_optimize

    def load_model(self) -> None:
        self._bg_remover = BiRefNet.from_pretrained('ZhengPeng7/BiRefNet_dynamic')
        self._bg_remover.to(self._device)
        self._bg_remover.eval()
        self._bg_remover.half()

        if self._enable_optimize:
            self.optimize()

    def unload_model(self) -> None:
        del self._bg_remover
        gc.collect()
        torch.cuda.empty_cache()
        self._bg_remover = None

    def remove_bg(self, image: Image) -> tuple[Image, bool]:
        result_image = image

        input_image = self._transform_image(result_image).unsqueeze(0).to(self._device).half()
        with torch.no_grad():
            preds = self._bg_remover(input_image)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(result_image.size)
        result_image.putalpha(mask)

        # filtered_image = self.filter_semi_transparent_pixels(result_image)
        # valid = self.is_image_valid(result_image)
        valid = True

        return result_image, valid

    def optimize(self):
        # logger.info("🔦 running torch compilation for BiRefnet model...")
        # self._bg_remover = torch.compile(self._bg_remover, mode="max-autotune", fullgraph=True)
        #
        # image = self._create_dummy_image()
        # input_image = self._transform_image(image).unsqueeze(0).to(self._device).half()
        # with torch.no_grad():
        #     self._bg_remover(input_image)
        # logger.info("🔦 Finished torch compilation.")
        pass
