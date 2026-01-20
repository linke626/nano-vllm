import os
import time
from nanovllm import LLM, SamplingParams

def run_experiment(budget, prompt_len):
    print(f"\n[实验配置] Budget: {budget} tokens | Prompt Length: ~{prompt_len} tokens")
    
    # 1. 初始化引擎，人为制造瓶颈
    # 这里的关键是：Prompt 长度 > max_num_batched_tokens
    path = os.path.expanduser("~/models/Qwen3-0.6B/") 
    try:
        llm = LLM(
            path, 
            enforce_eager=True, 
            tensor_parallel_size=1,
            max_num_batched_tokens=budget # <--- 核心限制
        )
        
        # 2. 构造超长 Prompt
        # 假设 1 个单词约等于 1.3 token，构造一个肯定超过 budget 的 prompt
        dummy_prompt = "Hello world " * (prompt_len // 2) 
        
        # 3. 尝试生成
        sampling_params = SamplingParams(temperature=0.7, max_tokens=10)
        start_time = time.time()
        output = llm.generate([dummy_prompt], sampling_params, use_tqdm=True)
        end_time = time.time()
        
        print(f"✅ [成功] 耗时: {end_time - start_time:.2f}s")
        print(f"   输出: {output[0]['text']}")
        
    except Exception as e:
        print(f"❌ [失败] 系统崩溃/报错: {str(e)}")

if __name__ == "__main__":
    # Case 1: 模拟原版系统 (Budget 充足) -> 应该成功
    # run_experiment(budget=4096, prompt_len=2000)
    
    # Case 2: 模拟硬件受限场景 (Budget 不足) -> 原版会挂，你的版本应该能跑
    run_experiment(budget=512, prompt_len=2000)