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
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # [Nano-vLLM Mod] Chunked Prefill Logic
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            
            # 1. First time allocation (if needed)
            if not seq.block_table:
                # Assuming BlockManager allocates all blocks at once for simplicity
                if not self.block_manager.can_allocate(seq):
                    break
                self.block_manager.allocate(seq)
                # Initialize progress (skip cached tokens)
                seq.processed_token_len = seq.num_cached_tokens

            # 2. Check budget
            budget = self.max_num_batched_tokens - num_batched_tokens
            if budget <= 0:
                break

            # 3. Calculate chunk size
            num_remaining = seq.num_prompt_tokens - seq.processed_token_len
            chunk_len = min(num_remaining, budget)
            
            if chunk_len <= 0:
                # Should not happen unless logic error or fully cached, move to next
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                num_seqs += 1
                scheduled_seqs.append(seq)
                continue

            # 4. Schedule this chunk
            seq.this_step_token_len = chunk_len
            num_batched_tokens += chunk_len
            scheduled_seqs.append(seq)

            # 5. Check if prefill is finished
            if seq.processed_token_len + chunk_len >= seq.num_prompt_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                num_seqs += 1
            else:
                # Keep in waiting queue (at head) for next step
                # Break to prevent starvation of this large request by smaller ones?
                # Or continue to fill batch? Usually break if we hit budget.
                break

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                # [Nano-vLLM Mod] Decode always processes 1 token
                seq.this_step_token_len = 1
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

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