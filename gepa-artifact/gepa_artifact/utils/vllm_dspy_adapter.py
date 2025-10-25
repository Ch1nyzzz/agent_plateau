
import dspy
from vllm import LLM, SamplingParams
from typing import List, Optional, Dict, Any
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import ray


# ============================================================
# 混合并行版本（张量并行 + 数据并行）
# ============================================================

class vLLMOfflineHybridParallel:
    
    def __init__(
        self,
        model: str = None,
        tensor_parallel_size: int = 2,
        num_model_instances: int = 2,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 32768,
        max_num_seqs: int = 128,
        trust_remote_code: bool = True,
        temperature: float = 0.6,
        max_tokens: int = 16384,
        top_p: float = 0.95,
        **kwargs,
    ):
        """
        初始化混合并行推理

        参数:
            model: 模型路径
            tensor_parallel_size: 每个模型实例使用的GPU数量（张量并行）
            num_model_instances: 同时运行的模型实例数量（数据并行）
            gpu_memory_utilization: GPU显存利用率
            max_model_len: 最大序列长度
            max_num_seqs: 最大并行处理序列数
            trust_remote_code: 是否信任远程代码
            temperature: 默认温度
            max_tokens: 默认最大token数
            top_p: 默认top_p值
        """
        self.model_path = model or os.environ.get('VLLM_MODEL_PATH', '/home/yuhan/model_zoo/Qwen3-8B')
        self.tensor_parallel_size = tensor_parallel_size
        self.num_model_instances = num_model_instances
        self.max_num_seqs = max_num_seqs
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.kwargs = kwargs

        # 计算总GPU数量
        total_gpus_needed = tensor_parallel_size * num_model_instances

        # 检查可用GPU
        available_gpus = torch.cuda.device_count()
        if available_gpus < total_gpus_needed:
            raise ValueError(
                f"需要 {total_gpus_needed} 张GPU "
                f"(tensor_parallel_size={tensor_parallel_size} × num_model_instances={num_model_instances}), "
                f"但只检测到 {available_gpus} 张GPU"
            )

        print(f"[vLLM] 正在初始化混合并行模式")
        print(f"[vLLM] 架构配置:")
        print(f"  - 张量并行大小: {tensor_parallel_size} (每个模型实例用{tensor_parallel_size}张GPU)")
        print(f"  - 模型实例数量: {num_model_instances} (数据并行)")
        print(f"  - 总GPU使用: {total_gpus_needed} 张")
        print(f"[vLLM] 模型: {self.model_path}")
        print("[vLLM] 这可能需要几分钟，请耐心等待...")

        # GPU分配方案
        print(f"\n[vLLM] GPU分配方案:")
        for i in range(num_model_instances):
            start_gpu = i * tensor_parallel_size
            end_gpu = start_gpu + tensor_parallel_size - 1
            print(f"  模型实例 {i}: GPU {start_gpu}-{end_gpu}")

        # 初始化 Ray（如果尚未初始化）
        if not ray.is_initialized():
            print("\n[Ray] 初始化 Ray...")
            ray.init(
                address=None,  # 显式指定启动新集群,不连接到现有集群
                ignore_reinit_error=True,
                num_cpus=None,  # 自动检测
                num_gpus=available_gpus,  # 使用检测到的GPU数量
                dashboard_host="0.0.0.0",  # 启用dashboard便于监控
                # 增加超时时间，因为模型加载可能需要较长时间
                _system_config={
                    "gcs_rpc_server_reconnect_timeout_s": 300,  # 5分钟
                    "gcs_server_request_timeout_seconds": 300,
                    "task_retry_delay_ms": 5000,
                },
                # 设置对象存储内存限制
                object_store_memory=10 * 1024 * 1024 * 1024,  # 10GB
            )
            print("[Ray] ✓ Ray 初始化完成")
            print(f"[Ray] GCS 超时设置: 300 秒（足够加载大模型）")

        # 检查 Ray 可用资源
        ray_resources = ray.available_resources()
        ray_gpus = ray_resources.get("GPU", 0)
        print(f"[Ray] 可用资源: {ray_gpus} GPU(s)")

        if ray_gpus < total_gpus_needed:
            raise ValueError(
                f"Ray 中可用GPU不足: 需要 {total_gpus_needed} 张GPU, "
                f"但 Ray 只有 {ray_gpus} 张GPU可用。\n"
                f"提示: 确保 Ray 初始化时能检测到所有GPU"
            )

        # 创建 Ray Actor 类来管理每个模型实例
        @ray.remote(num_gpus=tensor_parallel_size)
        class VLLMModelActor:
            """Ray Actor，每个实例管理一个 vLLM 模型"""

            def __init__(self, model_path, tensor_parallel_size, gpu_memory_utilization,
                        max_model_len, max_num_seqs, trust_remote_code, gpu_ids, instance_id):
                """初始化模型"""
                import time
                start_time = time.time()

                # 设置这个 Actor 可见的 GPU
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

                print(f"[Ray Actor #{instance_id}] 开始在 GPU {gpu_ids} 上加载模型...")
                print(f"[Ray Actor #{instance_id}] 模型路径: {model_path}")
                print(f"[Ray Actor #{instance_id}] 这可能需要 1-3 分钟，请耐心等待...")

                self.llm = LLM(
                    model=model_path,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                    trust_remote_code=trust_remote_code,
                    max_num_seqs=max_num_seqs,
                    # 减少初始化时的详细输出
                    disable_log_stats=True,
                )

                self.gpu_ids = gpu_ids
                self.instance_id = instance_id
                elapsed = time.time() - start_time
                print(f"[Ray Actor #{instance_id}] ✓ GPU {gpu_ids} 上的模型加载完成！耗时 {elapsed:.1f} 秒")

            def generate(self, prompts, sampling_params_dict):
                """执行推理"""
                # 从字典重建 SamplingParams
                sampling_params = SamplingParams(**sampling_params_dict)
                outputs = self.llm.generate(prompts, sampling_params)

                # 提取结果
                results = []
                for output in outputs:
                    if sampling_params.n == 1:
                        results.append(output.outputs[0].text)
                    else:
                        results.append([o.text for o in output.outputs])

                return results

            def get_gpu_ids(self):
                """返回此 Actor 使用的 GPU IDs"""
                return self.gpu_ids

        # 创建多个 Ray Actor 实例
        print("\n[Ray] 创建模型 Actor 实例...")
        print("[提示] 多个模型会并行加载，总耗时约等于单个模型加载时间")
        self.model_actors = []

        import time
        overall_start = time.time()

        for instance_id in range(num_model_instances):
            # 计算这个实例使用的GPU
            start_gpu = instance_id * tensor_parallel_size
            gpu_ids = list(range(start_gpu, start_gpu + tensor_parallel_size))

            print(f"\n[Ray] 提交模型实例 {instance_id} 的加载任务 (GPU {gpu_ids})...")

            # 创建 Ray Actor（异步，不会阻塞）
            actor = VLLMModelActor.remote(
                model_path=self.model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                trust_remote_code=trust_remote_code,
                gpu_ids=gpu_ids,
                instance_id=instance_id,
            )

            self.model_actors.append({
                'instance_id': instance_id,
                'gpu_ids': gpu_ids,
                'actor': actor,
            })

        # 等待所有 Actor 就绪（并行加载）
        print(f"\n[Ray] 等待所有 {num_model_instances} 个模型实例加载完成...")
        print(f"[Ray] 加载进度：并行加载中，请稍候...")

        try:
            ray.get([actor['actor'].get_gpu_ids.remote() for actor in self.model_actors])
            overall_elapsed = time.time() - overall_start
            print(f"\n[Ray] ✓ 所有模型实例加载成功！总耗时: {overall_elapsed:.1f} 秒")
        except Exception as e:
            print(f"\n[Ray] ✗ 模型加载失败: {e}")
            print(f"[提示] 如果超时，请尝试：")
            print(f"  1. 减少 num_model_instances")
            print(f"  2. 增加 Ray 超时时间")
            print(f"  3. 检查 GPU 显存是否足够")
            raise

        print(f"\n[vLLM] ✓ 所有 {num_model_instances} 个模型实例已就绪！")
        print(f"[vLLM] 配置: temp={temperature}, max_tokens={max_tokens}, top_p={top_p}")
        print(f"[vLLM] 混合并行已启用 (基于 Ray):")
        print(f"  - 张量并行：每个模型分布在{tensor_parallel_size}张GPU上（减少显存占用）")
        print(f"  - 数据并行：同时运行{num_model_instances}个模型实例（提升吞吐量）")
        print(f"  - 理论吞吐量提升：{num_model_instances}x")

        self._lock = threading.Lock()
        self.history = []
        self._request_count = 0

    def __call__(
        self,
        prompt: str = None,
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        单个推理（轮询分配到不同的模型实例）

        参数:
            prompt: 文本prompt
            messages: 对话格式消息
            **kwargs: 其他参数

        返回:
            生成的文本列表
        """
        # 处理输入
        if messages is not None:
            prompt = self._format_messages(messages)
        elif prompt is None:
            raise ValueError("必须提供 prompt 或 messages")

        # 轮询负载均衡
        with self._lock:
            instance_idx = self._request_count % self.num_model_instances
            self._request_count += 1

        # 使用选中的模型 Actor
        actor = self.model_actors[instance_idx]['actor']

        # 合并参数
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        n = kwargs.get("n", 1)

        # 创建采样参数字典（Ray 需要可序列化的参数）
        sampling_params_dict = {
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'n': n,
        }

        # 通过 Ray 执行推理
        results = ray.get(actor.generate.remote([prompt], sampling_params_dict))

        return results

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """将对话格式转换为prompt"""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        if not prompt.endswith("Assistant: "):
            prompt += "Assistant: "

        return prompt

    def batch_generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[str]:
        """
        批量推理（混合并行）

        将prompts分成num_model_instances份，分配到各个模型实例并行处理

        参数:
            prompts: prompt列表
            **kwargs: 采样参数

        返回:
            生成的文本列表（顺序与输入一致）
        """
        num_prompts = len(prompts)
        print(f"[vLLM] 批量推理 {num_prompts} 个prompt")
        print(f"[vLLM] 使用 {self.num_model_instances} 个模型实例（每个实例用{self.tensor_parallel_size}张GPU）")

        # 合并参数
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        n = kwargs.get("n", 1)

        # 创建采样参数字典
        sampling_params_dict = {
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'n': n,
        }

        # 将prompts分配到各个模型实例
        chunk_size = (num_prompts + self.num_model_instances - 1) // self.num_model_instances
        prompt_chunks = []

        for i in range(self.num_model_instances):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_prompts)
            if start_idx < num_prompts:
                prompt_chunks.append((i, prompts[start_idx:end_idx], start_idx))

        print(f"[vLLM] 数据分配:")
        for instance_id, chunk, start_idx in prompt_chunks:
            gpu_ids = self.model_actors[instance_id]['gpu_ids']
            print(f"  模型实例 {instance_id} (GPU {gpu_ids}): {len(chunk)} prompts (索引 {start_idx}-{start_idx+len(chunk)-1})")

        # 使用 Ray 并行执行推理
        print(f"[Ray] 启动并行推理任务...")
        ray_futures = []
        for instance_id, chunk, start_idx in prompt_chunks:
            actor = self.model_actors[instance_id]['actor']
            # 提交异步任务
            future = actor.generate.remote(chunk, sampling_params_dict)
            ray_futures.append((start_idx, future))

        # 收集结果
        print(f"[Ray] 等待所有任务完成...")
        results_dict = {}
        for start_idx, future in ray_futures:
            chunk_results = ray.get(future)
            results_dict[start_idx] = chunk_results

        # 按原始顺序重组结果
        results = []
        for start_idx in sorted(results_dict.keys()):
            results.extend(results_dict[start_idx])

        print(f"[vLLM] ✓ 批量推理完成，共生成 {len(results)} 个结果")

        return results

    def inspect(self) -> Dict[str, Any]:
        """查看配置信息"""
        return {
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "num_model_instances": self.num_model_instances,
            "total_gpus": self.tensor_parallel_size * self.num_model_instances,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "total_requests": self._request_count,
            "gpu_allocation": [
                {
                    "instance_id": inst['instance_id'],
                    "gpu_ids": inst['gpu_ids']
                }
                for inst in self.model_actors
            ],
            "backend": "Ray"
        }

    def shutdown(self):
        """关闭 Ray actors 并清理资源"""
        print("\n[Ray] 关闭模型实例...")
        for actor_info in self.model_actors:
            ray.kill(actor_info['actor'])
        print("[Ray] ✓ 所有模型实例已关闭")

    def __del__(self):
        """析构函数，自动清理资源"""
        try:
            self.shutdown()
        except:
            pass


# ============================================================
# 使用示例
# ============================================================

def example_basic():
    """示例1：基础推理"""
    print("=" * 60)
    print("示例1: 基础推理")
    print("=" * 60)

    # 创建 vLLM 离线推理适配器
    lm = vLLMOffline(
        model="/home/yuhan/model_zoo/Qwen3-8B",
        temperature=0.6,
        max_tokens=200,
    )

    # 配置 DSPy
    dspy.configure(lm=lm)

    print("\n单个推理:")
    result = lm("什么是深度学习？请用一句话回答。")
    print(f"回答: {result[0]}")


def example_batch():
    """示例2：批量推理"""
    print("=" * 60)
    print("示例2: 批量推理（数据并行）")
    print("=" * 60)

    lm = vLLMOffline()

    prompts = [
        "1+1=?",
        "2+2=?",
        "3+3=?",
        "Python中如何定义函数？",
        "什么是transformer？",
    ]

    import time
    start = time.time()
    results = lm.batch_generate(prompts, max_tokens=100)
    end = time.time()

    print(f"\n批量推理 {len(prompts)} 个问题，耗时: {end-start:.2f}秒")
    for prompt, result in zip(prompts, results):
        print(f"\n问题: {prompt}")
        print(f"回答: {result.strip()[:100]}...")


def example_benchmark():
    """示例3：在 benchmark 评估中使用（模拟 notebook）"""
    print("=" * 60)
    print("示例3: Benchmark 评估（模拟 notebook）")
    print("=" * 60)

    # 1. 初始化 vLLM 离线推理
    lm = vLLMOffline(
        temperature=0.6,
        max_tokens=15000,  # 与 notebook 一致
        top_p=0.95,
    )

    # 2. 配置 DSPy
    dspy.configure(lm=lm)

    # 3. 加载 benchmark（这里使用简单示例）
    print("\n加载 benchmark...")
    from gepa_artifact.benchmarks.AIME import benchmark as aime_metas

    cur_meta = aime_metas
    bench = cur_meta[0].benchmark()

    print(f"训练集大小: {len(bench.train_set)}")
    print(f"验证集大小: {len(bench.val_set)}")
    print(f"测试集大小: {len(bench.test_set)}")

    # 4. 加载 program
    print("\n加载 program...")
    program = cur_meta[0].program[0]
    print(program)

    # 5. 定义评估器
    print("\n创建评估器...")
    evaluate = dspy.Evaluate(
        devset=bench.test_set[:5],  # 只评估前5个样本作为演示
        metric=cur_meta[0].metric,
        num_threads=4,  # 多线程评估
        display_table=True,
        display_progress=True,
        max_errors=100,
    )

    # 6. 评估程序
    print("\n开始评估...")
    score = evaluate(program)
    print(f"\n最终得分: {score}")


def example_multi_gpu():
    """示例4：多GPU数据并行"""
    print("=" * 60)
    print("示例4: 多GPU数据并行推理（4张GPU）")
    print("=" * 60)

    # 初始化多GPU数据并行
    lm = vLLMOfflineMultiGPU(
        model="/home/yuhan/model_zoo/Qwen3-8B",
        num_gpus=4,  # 使用4张GPU
        temperature=0.6,
        max_tokens=500,
    )

    # 配置 DSPy
    dspy.configure(lm=lm)

    # 准备大量问题（模拟真实场景）
    print("\n生成测试数据...")
    prompts = []
    for i in range(100):
        prompts.append(f"请计算 {i} + {i+1} 等于多少？只回答数字。")

    # 批量推理 - 数据会自动分配到4张GPU
    print(f"\n开始批量推理 {len(prompts)} 个问题...")
    import time
    start = time.time()
    results = lm.batch_generate(prompts, max_tokens=100)
    end = time.time()

    print(f"\n✓ 批量推理完成！")
    print(f"总耗时: {end-start:.2f} 秒")
    print(f"平均每个问题: {(end-start)/len(prompts):.3f} 秒")
    print(f"吞吐量: {len(prompts)/(end-start):.1f} 问题/秒")

    # 显示前5个结果
    print(f"\n前5个结果:")
    for i in range(5):
        print(f"  问题: {prompts[i]}")
        print(f"  回答: {results[i].strip()[:50]}")

    # 查看配置信息
    print(f"\n配置信息:")
    print(lm.inspect())


def example_dspy_signature():
    """示例5：在 DSPy Signature 中使用"""
    print("=" * 60)
    print("示例5: 在 DSPy Signature 中使用")
    print("=" * 60)

    lm = vLLMOffline()
    dspy.configure(lm=lm)

    # 定义 Signature
    class QA(dspy.Signature):
        """回答问题"""
        question = dspy.InputField(desc="问题")
        answer = dspy.OutputField(desc="答案")

    # 使用 ChainOfThought
    qa_module = dspy.ChainOfThought(QA)

    # 推理
    questions = [
        "什么是注意力机制？",
        "解释一下什么是vLLM？",
    ]

    for question in questions:
        prediction = qa_module(question=question)
        print(f"\n问题: {question}")
        print(f"答案: {prediction.answer[:200]}...")


if __name__ == "__main__":
    import sys

    # 根据命令行参数选择运行哪个示例
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name == "basic":
            example_basic()
        elif example_name == "batch":
            example_batch()
        elif example_name == "benchmark":
            example_benchmark()
        elif example_name == "multigpu":
            example_multi_gpu()
        elif example_name == "signature":
            example_dspy_signature()
        else:
            print(f"未知示例: {example_name}")
            print("可用示例: basic, batch, benchmark, multigpu, signature")
    else:
        # 默认运行基础示例
        print("提示: 可以使用 'python vllm_dspy_adapter.py [示例名]' 运行特定示例")
        print("可用示例: basic, batch, benchmark, multigpu, signature")
        print("\n【推荐】如果你有多张GPU，使用: python vllm_dspy_adapter.py multigpu\n")
        example_basic()
