import time
import torch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

def run_ablation_experiment(model_path: str, enforce_eager: bool, batch_size: int, output_len: int):
    mode_name = "传统 Eager 模式 (无 Graph，巨量调度开销)" if enforce_eager else "CUDA Graph 模式 (零调度开销)"
    print(f"\n{'='*20} 开始测试: {mode_name} {'='*20}")
    print(f"配置 -> 并发数(Batch Size): {batch_size}, 强制生成长度: {output_len}")

    # 核心开关：通过 enforce_eager 控制是否使用 CUDA Graph
    engine = LLMEngine(
        model=model_path,
        enforce_eager=enforce_eager,
        max_num_seqs=max(16, batch_size) # 确保系统能容纳当前并发
    )
    
    # 构造相同的 Dummy Prompt
    dummy_prompt = "Hello, I am a highly efficient AI inference engine. Please tell me a long story about "
    prompts = [dummy_prompt + str(i) for i in range(batch_size)]
    
    # 强制生成固定长度的词，关闭提前结束，保证对比的绝对公平
    sampling_params = SamplingParams(max_tokens=output_len, ignore_eos=True, temperature=0.01)

    # 1. 预热阶段 (Warmup)：极其重要！
    # 必须先跑一次，让 PyTorch 完成底层的 CUDA 上下文初始化和 Caching Allocator 的内存伸缩
    print("正在进行系统预热 (Warmup)...")
    engine.generate([prompts[0]], sampling_params, use_tqdm=False)
    
    # 2. 正式压测阶段
    print("预热完成，开始计时...")
    torch.cuda.synchronize() # 确保 GPU 之前的任务全清空
    start_time = time.perf_counter()
    
    # 运行你的生成引擎
    outputs = engine.generate(prompts, sampling_params, use_tqdm=True)
    
    torch.cuda.synchronize() # 确保 GPU 真实计算完毕
    end_time = time.perf_counter()
    
    # 3. 统计指标计算
    total_time = end_time - start_time
    total_decode_tokens = batch_size * output_len
    
    # 这里做了一个简化：端到端时间包含了极小部分的 Prefill 时间
    # 但由于输出长度较长，端到端吞吐量无限逼近 Decode 吞吐量
    throughput = total_decode_tokens / total_time
    # 估算单个请求的平均 字间延迟 (Inter-Token Latency)
    itl_ms = (total_time / output_len) * 1000 
    
    print(f"\n>>> 实验结果 ({mode_name}) <<<")
    print(f"端到端总耗时:   {total_time:.4f} 秒")
    print(f"系统总吞吐量:   {throughput:.2f} Tokens/秒")
    print(f"估算字间延迟:   {itl_ms:.2f} ms/Token (ITL)")
    print("=" * 70)
    
    # 4. 安全退出，彻底释放显存，防止第二次运行 OOM
    engine.exit()
    del engine
    torch.cuda.empty_cache()
    time.sleep(2) # 给操作系统一点回收显存的时间

if __name__ == "__main__":
    # TODO: 替换为你的本地或 HF 模型路径
    MODEL_PATH = "/data2/home/wanghaoyi/models/Qwen3-0.6B/" 
    
    OUTPUT_LENGTH = 128 # 强制每个请求生成 128 个词
    
    print("\n" + "#"*50)
    print("实验 1：极小并发场景 (BS=1) -> 观察 CPU 调度瓶颈的放大")
    print("#"*50)
    run_ablation_experiment(MODEL_PATH, enforce_eager=True,  batch_size=1, output_len=OUTPUT_LENGTH)
    run_ablation_experiment(MODEL_PATH, enforce_eager=False, batch_size=1, output_len=OUTPUT_LENGTH)
    
    print("\n" + "#"*50)
    print("实验 2：较大并发场景 (BS=16) -> 观察显存瓶颈掩盖 CPU 瓶颈")
    print("#"*50)
    run_ablation_experiment(MODEL_PATH, enforce_eager=True,  batch_size=16, output_len=OUTPUT_LENGTH)
    run_ablation_experiment(MODEL_PATH, enforce_eager=False, batch_size=16, output_len=OUTPUT_LENGTH)