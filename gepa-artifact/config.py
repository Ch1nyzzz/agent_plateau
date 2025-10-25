"""
环境配置管理模块
用于统一管理和读取环境变量配置
"""
import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    加载 .env 文件中的环境变量

    Args:
        env_path: .env 文件的路径，如果为 None 则使用项目根目录的 .env
    """
    if env_path is None:
        # 获取项目根目录
        root_dir = Path(__file__).parent
        env_path = root_dir / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        print(f"警告: .env 文件不存在于 {env_path}")
        print(f"请复制 .env.template 为 .env 并填入实际配置")
        return

    # 读取并设置环境变量
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue

            # 解析 KEY=VALUE 格式
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # 只在环境变量不存在时设置（命令行设置的优先级更高）
                if key and value and key not in os.environ:
                    os.environ[key] = value


class Config:
    """配置类，提供统一的配置访问接口"""

    def __init__(self, auto_load: bool = True):
        """
        初始化配置

        Args:
            auto_load: 是否自动加载 .env 文件
        """
        if auto_load:
            load_env_file()

    # OpenAI 配置
    @property
    def openai_api_key(self) -> str:
        """获取 OpenAI API Key"""
        key = os.getenv('OPENAI_API_KEY', '')
        if not key or key == 'your_openai_api_key_here':
            raise ValueError(
                "OPENAI_API_KEY 未设置或使用默认值。"
                "请在 .env 文件中设置有效的 API Key"
            )
        return key

    # vLLM 配置
    @property
    def vllm_model_path(self) -> str:
        """获取 vLLM 模型路径"""
        return os.getenv('VLLM_MODEL_PATH', '/home/yuhan/model_zoo/Qwen3-8B')

    @property
    def vllm_host(self) -> str:
        """获取 vLLM 服务主机"""
        return os.getenv('VLLM_HOST', 'localhost')

    @property
    def vllm_port(self) -> int:
        """获取 vLLM 服务端口"""
        return int(os.getenv('VLLM_PORT', '8000'))

    @property
    def vllm_api_base(self) -> str:
        """获取 vLLM API 基础 URL"""
        return f"http://{self.vllm_host}:{self.vllm_port}/v1"

    @property
    def vllm_gpu_memory_utilization(self) -> float:
        """获取 vLLM GPU 内存利用率"""
        return float(os.getenv('VLLM_GPU_MEMORY_UTILIZATION', '0.9'))

    @property
    def vllm_tensor_parallel_size(self) -> int:
        """获取 vLLM Tensor 并行度"""
        return int(os.getenv('VLLM_TENSOR_PARALLEL_SIZE', '1'))

    @property
    def vllm_max_model_len(self) -> int:
        """获取 vLLM 最大模型长度"""
        return int(os.getenv('VLLM_MAX_MODEL_LEN', '15000'))

    @property
    def vllm_max_num_seqs(self) -> int:
        """获取 vLLM 最大并发序列数"""
        return int(os.getenv('VLLM_MAX_NUM_SEQS', '256'))

    @property
    def vllm_max_num_batched_tokens(self) -> int:
        """获取 vLLM 单批次最大Token数"""
        return int(os.getenv('VLLM_MAX_NUM_BATCHED_TOKENS', '8192'))

    # WandB 配置
    @property
    def wandb_api_key(self) -> Optional[str]:
        """获取 WandB API Key（可选）"""
        return os.getenv('WANDB_API_KEY')

    def setup_openai_env(self) -> None:
        """设置 OpenAI 相关环境变量"""
        os.environ['OPENAI_API_KEY'] = self.openai_api_key

    def setup_wandb_env(self) -> None:
        """设置 WandB 相关环境变量"""
        if self.wandb_api_key:
            os.environ['WANDB_API_KEY'] = self.wandb_api_key

    def setup_all_env(self) -> None:
        """设置所有环境变量"""
        self.setup_openai_env()
        self.setup_wandb_env()

    def display_config(self) -> None:
        """显示当前配置（隐藏敏感信息）"""
        print("=" * 60)
        print("当前配置:")
        print("=" * 60)
        print(f"OpenAI API Key: {'*' * 10}{self.openai_api_key[-10:] if len(self.openai_api_key) > 10 else '***'}")
        print(f"\nvLLM 基础配置:")
        print(f"  模型路径: {self.vllm_model_path}")
        print(f"  API 地址: {self.vllm_api_base}")
        print(f"\nvLLM 性能参数:")
        print(f"  GPU 内存利用率: {self.vllm_gpu_memory_utilization}")
        print(f"  Tensor 并行度: {self.vllm_tensor_parallel_size}")
        print(f"  最大模型长度: {self.vllm_max_model_len}")
        print(f"\nvLLM 并发参数:")
        print(f"  最大并发序列数: {self.vllm_max_num_seqs}")
        print(f"  最大批次Token数: {self.vllm_max_num_batched_tokens}")
        if self.wandb_api_key:
            print(f"\nWandB API Key: {'*' * 10}{self.wandb_api_key[-10:]}")
        else:
            print("\nWandB API Key: 未设置")
        print("=" * 60)


# 创建全局配置实例
config = Config()


if __name__ == "__main__":
    # 测试配置加载
    config.display_config()
    config.setup_all_env()
    print("\n环境变量已设置完成!")
