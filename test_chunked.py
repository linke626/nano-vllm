import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    # 1. 准备模型路径
    path = os.path.expanduser("~/models/Qwen3-0.6B/") # 确保路径正确
    
    # 2. 初始化引擎，关键在于【限制 batched tokens】
    # 我们把 budget 设为 32，意味着每一步最多只能处理 32 个 token
    llm = LLM(
        path, 
        enforce_eager=True, 
        tensor_parallel_size=1,
        max_num_batched_tokens=32  # <--- 核心测试点：制造瓶颈
    )

    # 3. 构造一个长 Prompt (长度肯定超过 32)
    long_prompt = "Hello, " * 50  # 大约 100-200 个 token
    
    # 4. 运行生成
    print(f"开始测试 Chunked Prefill...")
    print(f"Prompt 长度估算: {len(long_prompt.split())} 单词 (肯定 > 32 tokens)")
    print(f"预期行为: 进度条应该会缓慢移动，而不是瞬间完成 Prefill")
    
    sampling_params = SamplingParams(temperature=0.1, max_tokens=20) # 生成稍微短点
    outputs = llm.generate([long_prompt], sampling_params)

    print("\n生成结果:", outputs[0]["text"])

if __name__ == "__main__":
    main()