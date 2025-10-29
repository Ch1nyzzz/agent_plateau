
from vllm import LLM, SamplingParams

def main():
    # 1) 把这里改成你的本地模型目录，如："/home/yuhan/model_zoo/Qwen3-8B"
    model_path = "/home/yuhan/model_zoo/Qwen3-8B"

    # 2) 采样参数（尽量简洁）
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=200,
    )
    
    # 3) 初始化 LLM（默认会用到可见的 CUDA GPU）
    llm = LLM(
        model=model_path,
        trust_remote_code=True,   # 某些本地模型需要
        tensor_parallel_size=2,   # 单卡最简单；多卡时改成 >1
        dtype="bfloat16"          # 你的 GPU 不支持 bfloat16 时可改为 "float16"
    )

    # 4) 简单的输入
    prompt = "用一句中文介绍一下你自己："

    # 5) 推理
    outputs = llm.generate([prompt], sampling_params)

    # 6) 打印结果
    print("=== Prompt ===")
    print(prompt)
    print("\n=== Output ===")
    print(outputs[0].outputs[0].text.strip())

if __name__ == "__main__":
    main()