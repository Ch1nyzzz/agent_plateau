#!/usr/bin/env python3
"""
测试 vLLM 不同版本的脚本
"""

import sys
sys.path.insert(0, '/data/home/yuhan/ReAct_learning/agent_plateau/gepa-artifact')

from gepa_artifact.utils.vllm_dspy_adapter import (
    vLLMOfflineSimple,
    vLLMOfflineHybridParallel,
    create_vllm_model
)


def test_simple_version():
    """测试简单版本（不使用 Ray）"""
    print("\n" + "="*70)
    print("测试 1: vLLMOfflineSimple (不使用 Ray)")
    print("="*70)

    # 创建简单版本的模型
    model = vLLMOfflineSimple(
        tensor_parallel_size=1,
        max_tokens=100,
        temperature=0.7,
    )

    # 查看配置
    print("\n配置信息:")
    import json
    print(json.dumps(model.inspect(), indent=2, ensure_ascii=False))

    # 单个推理测试
    print("\n单个推理测试:")
    prompt = "请用一句话介绍Python编程语言。"
    result = model(prompt=prompt)
    print(f"Prompt: {prompt}")
    print(f"结果: {result[0]}")

    # 批量推理测试
    print("\n批量推理测试:")
    prompts = [
        "什么是机器学习？",
        "什么是深度学习？",
    ]
    results = model.batch_generate(prompts)
    for p, r in zip(prompts, results):
        print(f"\nPrompt: {p}")
        print(f"结果: {r}")

    print("\n✓ 简单版本测试完成")


def test_factory_function():
    """测试工厂函数"""
    print("\n" + "="*70)
    print("测试 2: create_vllm_model 工厂函数")
    print("="*70)

    # 测试自动选择（应该选择简单版本）
    print("\n2.1 自动选择 (num_model_instances=1, 应该选择简单版本):")
    model1 = create_vllm_model(
        tensor_parallel_size=1,
        num_model_instances=1,
        max_tokens=50,
    )
    print(f"选择的版本: {model1.inspect()['backend']}")

    # 测试强制使用简单版本
    print("\n2.2 强制使用简单版本 (use_ray=False):")
    model2 = create_vllm_model(
        tensor_parallel_size=1,
        use_ray=False,
        max_tokens=50,
    )
    print(f"选择的版本: {model2.inspect()['backend']}")

    print("\n✓ 工厂函数测试完成")


def test_hybrid_version():
    """测试混合并行版本（需要多个GPU和Ray）"""
    print("\n" + "="*70)
    print("测试 3: vLLMOfflineHybridParallel (使用 Ray)")
    print("="*70)
    print("注意: 此测试需要多个GPU，如果GPU不足会跳过")

    import torch
    available_gpus = torch.cuda.device_count()

    if available_gpus < 2:
        print(f"跳过: 需要至少2个GPU，当前只有 {available_gpus} 个")
        return

    try:
        # 创建混合并行版本
        model = create_vllm_model(
            tensor_parallel_size=1,
            num_model_instances=2,
            max_tokens=100,
        )

        # 查看配置
        print("\n配置信息:")
        import json
        print(json.dumps(model.inspect(), indent=2, ensure_ascii=False))

        # 批量推理测试
        print("\n批量推理测试:")
        prompts = [
            "什么是人工智能？",
            "什么是自然语言处理？",
            "什么是计算机视觉？",
            "什么是强化学习？",
        ]
        results = model.batch_generate(prompts)
        for p, r in zip(prompts, results):
            print(f"\nPrompt: {p}")
            print(f"结果: {r}")

        print("\n✓ 混合并行版本测试完成")

        # 清理资源
        model.shutdown()

    except Exception as e:
        print(f"混合并行版本测试失败: {e}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("vLLM 适配器版本测试")
    print("="*70)

    import argparse
    parser = argparse.ArgumentParser(description="测试 vLLM 不同版本")
    parser.add_argument(
        "--test",
        choices=["simple", "factory", "hybrid", "all"],
        default="simple",
        help="选择要运行的测试 (默认: simple)"
    )
    args = parser.parse_args()

    if args.test in ["simple", "all"]:
        test_simple_version()

    if args.test in ["factory", "all"]:
        test_factory_function()

    if args.test in ["hybrid", "all"]:
        test_hybrid_version()

    print("\n" + "="*70)
    print("所有测试完成")
    print("="*70)
