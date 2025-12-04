"""
Configuration file for 404-base-miner service
"""
import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Server configuration settings"""
    host: str = "0.0.0.0"
    port: int = 10006
    max_workers: int = 1
    reload: bool = False


@dataclass
class ImageConfig:
    """Image processing configuration"""
    max_file_size_mb: int = 20
    min_width: int = 64
    min_height: int = 64
    default_image_shape: tuple[int, int, int] = (512, 512, 3)


@dataclass
class ModelConfig:
    """Model configuration settings"""
    trellis_model_name: str = "microsoft/TRELLIS-image-large"
    ben2_model_name: str = "PramaLLC/BEN2"
    birefnet_model_name: str = "ZhengPeng7/BiRefNet_dynamic"
    attn_backend: str = "flash-attn"  # or "xformers"
    

@dataclass
class RayConfig:
    """Ray configuration settings"""
    ignore_reinit_error: bool = True
    num_gpus_per_worker: float = 0.05


@dataclass
class AppConfig:
    """Main application configuration"""
    server: ServerConfig = ServerConfig()
    image: ImageConfig = ImageConfig()
    model: ModelConfig = ModelConfig()
    ray: RayConfig = RayConfig()
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables"""
        config = cls()
        
        # Server config
        config.server.host = os.getenv("SERVER_HOST", config.server.host)
        config.server.port = int(os.getenv("SERVER_PORT", config.server.port))
        config.server.max_workers = int(os.getenv("MAX_WORKERS", config.server.max_workers))
        
        # Image config
        config.image.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", config.image.max_file_size_mb))
        config.image.min_width = int(os.getenv("MIN_IMAGE_WIDTH", config.image.min_width))
        config.image.min_height = int(os.getenv("MIN_IMAGE_HEIGHT", config.image.min_height))
        
        # Model config
        config.model.trellis_model_name = os.getenv("TRELLIS_MODEL", config.model.trellis_model_name)
        config.model.attn_backend = os.getenv("ATTN_BACKEND", config.model.attn_backend)
        
        return config


# Global configuration instance
config = AppConfig.from_env()
