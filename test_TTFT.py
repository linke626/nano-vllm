import os
import time
from nanovllm import LLM, SamplingParams

def test_ttft_mixed_workload():
    print("=== 开始长短请求混合调度测试 (首字延迟 TTFT 模拟) ===")
    
    # 设定极小的算力预算，强行触发 Chunked Prefill
    budget = 512 
    path = os.path.expanduser("~/models/Qwen3-0.6B/") 
    
    print(f"初始化引擎中... (强制硬件算力预算: {budget} tokens/step)")
    llm = LLM(
        path, 
        enforce_eager=False, 
        tensor_parallel_size=1,
        max_num_batched_tokens=budget 
    )
    
    # 构造请求数据 (假设约等于 1个词 1.2个 token)
    long_prompt_len = 4000
    short_prompt_len = 20
    
    long_prompt = "The quick brown fox jumps over the lazy dog. " * (long_prompt_len // 10)
    short_prompt = "Hello, what is your name? " * (short_prompt_len // 5)
    
    # 【修复1】：使用合法的 temperature (大于 1e-10)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=10)

    # 【修复2】：直接使用 llm 实例的方法，去掉 .engine
    print(f"\n[1] 注入超长请求 (期望长度约 {long_prompt_len} tokens)...")
    llm.add_request(long_prompt, sampling_params)
    
    print(f"[2] 注入超短请求 (期望长度约 {short_prompt_len} tokens)...")
    llm.add_request(short_prompt, sampling_params)
    
    print("\n[3] 引擎开始步进 (Step) 调度...")
    
    # 【修复3】：严格确保局部变量在 while 循环之外初始化
    step_count = 0
    start_time = time.time()
    
    while not llm.is_finished():
        # 执行一次前向传播
        step_start = time.time()
        output, num_tokens = llm.step()
        step_time = time.time() - step_start
        
        step_count += 1
        
        # 实时打印每一步的吞吐，直观感受 Chunked Prefill 的切分！
        if num_tokens > 0:
            print(f"  -> 第 {step_count:02d} 步 (Prefill 预填充): 本次吞下 {num_tokens:4d} tokens, 耗时 {step_time:.3f}s")
        else:
            print(f"  -> 第 {step_count:02d} 步 (Decode 解码)  : 本次吐出 {-num_tokens:4d} tokens, 耗时 {step_time:.3f}s")

        # 检查是否有序列彻底生成完毕
        if output:
            for seq_id, token_ids in output:
                elapsed = time.time() - start_time
                # Nano-vLLM 的 seq_id 是从0开始递增的
                if seq_id == 1: 
                    print(f"\n✅ 短请求 (ID:{seq_id}) 处理完成! \n   等待队列 + 推理总耗时: {elapsed:.2f}s (发生于第 {step_count} 步)\n")
                else: 
                    print(f"\n✅ 长请求 (ID:{seq_id}) 处理完成! \n   漫长的预填充 + 推理总耗时: {elapsed:.2f}s (发生于第 {step_count} 步)\n")

if __name__ == "__main__":
    test_ttft_mixed_workload()