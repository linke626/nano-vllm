from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # ==========================================
        # 阶段 1: 绝对优先保障 Decode (老用户优先)
        # ==========================================
        decode_seqs = []
        # 注意：这里必须提取当前 running 队列的固定长度，防止在循环内 append 导致死循环
        num_running = len(self.running)
        for _ in range(num_running):
            if num_seqs >= self.max_num_seqs:
                break
            seq = self.running.popleft()
            
            # 极度关键的防 OOM 逻辑：确保老用户有显存吐出下一个字
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop()) # 牺牲队尾老用户
                else:
                    self.preempt(seq) # 连自己都保不住了，直接暂停
                    break
            else:
                # 显存充足，为老用户安排 1 个 Token 的算力预算
                num_seqs += 1
                self.block_manager.may_append(seq)
                seq.this_step_token_len = 1
                decode_seqs.append(seq)
                num_batched_tokens += 1
                
        # 必须将活下来的老用户重新放回 running 队列左侧，保持正确的轮转顺序
        self.running.extendleft(reversed(decode_seqs))

        # ==========================================
        # 阶段 2: 用闲置预算进行 Chunked Prefill (混合填缝)
        # ==========================================
        prefill_seqs = []
        # 核心：计算扣除老用户消耗后，还剩下多少 Token 算力预算
        budget = self.max_num_batched_tokens - num_batched_tokens
        
        # 只有在还有剩余预算的情况下，才去接待新用户
        while self.waiting and num_seqs < self.max_num_seqs and budget > 0:
            seq = self.waiting[0]
            
            # 1. 首次分配显存处理
            if not seq.block_table:
                if not self.block_manager.can_allocate(seq):
                    break # 物理显存彻底耗尽，停止接待新客
                self.block_manager.allocate(seq)
                seq.processed_token_len = seq.num_cached_tokens

            # 2. 计算本次应该切多大的 Chunk (分块)
            num_remaining = seq.num_prompt_tokens - seq.processed_token_len
            chunk_len = min(num_remaining, budget)
            
            if chunk_len <= 0:
                # 已经被 Prefix Cache 完全命中，直接送入 Decode 阶段
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                continue
                
            # 3. 安排这个 Chunk 进入调度
            seq.this_step_token_len = chunk_len
            prefill_seqs.append(seq)
            budget -= chunk_len
            num_batched_tokens += chunk_len
            num_seqs += 1
            
            # 4. 检查这个长请求是否在这一轮被彻底切完了
            if seq.processed_token_len + chunk_len >= seq.num_prompt_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            else:
                # 还没切完，但本帧的算力预算已被榨干，跳出循环，留到下一个 Step 继续切
                break 

        # ==========================================
        # 阶段 3: 混合调度决策 (Piggybacking Decision)
        # ==========================================
        if prefill_seqs:
            # 只要有任何的 Prefill 新人上车，当前这一个 Step 必须走动态变长算子 (is_prefill=True)
            # 这就是真正的 Piggybacking！老用户的 Decode 任务搭上了新用户 Prefill 的顺风车
            scheduled_seqs = decode_seqs + prefill_seqs
            return scheduled_seqs, True
        elif decode_seqs:
            # 没有任何新任务，纯粹的老用户 Decode 车队。返回 False，享受 CUDA Graph 极限加速！
            return decode_seqs, False
        else:
            return [], False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        # [Nano-vLLM Mod] Reset progress on preempt (simple implementation)
        seq.processed_token_len = 0 
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            # [Nano-vLLM Mod] Only append tokens if we are generating (Running status)
            # During chunked prefill, we might get logits but we typically don't generate tokens 
            # until the last chunk. However, existing logic handles 'Running' status check.
            if seq.status == SequenceStatus.RUNNING:
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)