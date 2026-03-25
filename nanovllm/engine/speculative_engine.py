from dataclasses import dataclass
from tqdm.auto import tqdm

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


@dataclass
class _SpecPair:
	draft: Sequence
	target: Sequence


class SpeculativeEngine:

	def __init__(self, draft_model: str, target_model: str, k: int = 1, **kwargs):
		assert k >= 1
		self.k = k
		self.draft_engine = LLMEngine(draft_model, **kwargs)
		self.target_engine = LLMEngine(target_model, **kwargs)

	def exit(self):
		self.draft_engine.exit()
		self.target_engine.exit()

	def _init_sequence(self, prompt: str | list[int], sampling_params: SamplingParams) -> _SpecPair:
		tokenizer = self.target_engine.tokenizer
		token_ids = tokenizer.encode(prompt) if isinstance(prompt, str) else prompt

		draft = Sequence(token_ids, sampling_params)
		target = Sequence(token_ids, sampling_params)
		draft.status = target.status = SequenceStatus.RUNNING

		draft_bm = self.draft_engine.scheduler.block_manager
		target_bm = self.target_engine.scheduler.block_manager
		assert draft_bm.can_allocate(draft) and target_bm.can_allocate(target), "insufficient kv cache blocks"
		draft_bm.allocate(draft)
		target_bm.allocate(target)

		draft.processed_token_len = draft.num_cached_tokens
		target.processed_token_len = target.num_cached_tokens

		if draft.processed_token_len < draft.num_prompt_tokens:
			draft.this_step_token_len = draft.num_prompt_tokens - draft.processed_token_len
			self.draft_engine.model_runner.call("run", [draft], True)
			draft.processed_token_len = draft.num_prompt_tokens

		if target.processed_token_len < target.num_prompt_tokens:
			target.this_step_token_len = target.num_prompt_tokens - target.processed_token_len
			self.target_engine.model_runner.call("run", [target], True)
			target.processed_token_len = target.num_prompt_tokens

		return _SpecPair(draft, target)

	@staticmethod
	def _prepare_slot_for_next_token(seq: Sequence, block_manager):
		assert block_manager.can_append(seq), "insufficient kv cache blocks while appending"
		block_manager.may_append(seq)

	def _draft_decode_k(self, pair: _SpecPair, k: int) -> list[int]:
		sp = pair.target
		eos = self.target_engine.scheduler.eos
		drafted = []
		bm = self.draft_engine.scheduler.block_manager
		for _ in range(k):
			self._prepare_slot_for_next_token(pair.draft, bm)
			token = self.draft_engine.model_runner.call("run", [pair.draft], False)[0]
			pair.draft.append_token(token)
			drafted.append(token)
			if (not sp.ignore_eos) and token == eos:
				break
		return drafted

	def _sync_draft_to_target(self, pair: _SpecPair, old_len: int, committed_tokens: list[int]):
		draft_bm = self.draft_engine.scheduler.block_manager
		draft_bm.rewind(pair.draft, old_len)
		for token in committed_tokens:
			self._prepare_slot_for_next_token(pair.draft, draft_bm)
			pair.draft.append_token(token)

	def _verify_and_commit(self, pair: _SpecPair, drafted: list[int]) -> list[int]:
		if not drafted:
			return []

		target_bm = self.target_engine.scheduler.block_manager
		old_len = len(pair.target)

		self._prepare_slot_for_next_token(pair.target, target_bm)
		first_logits = self.target_engine.model_runner.call("run", [pair.target], False, True)
		first_pred = int(first_logits[0].argmax(dim=-1).item())

		if first_pred != drafted[0]:
			pair.target.append_token(first_pred)
			committed = [first_pred]
			self._sync_draft_to_target(pair, old_len, committed)
			return committed

		for i, token in enumerate(drafted):
			if i > 0:
				self._prepare_slot_for_next_token(pair.target, target_bm)
			pair.target.append_token(token)

		pair.target.processed_token_len = old_len
		pair.target.this_step_token_len = len(drafted)
		verify_logits = self.target_engine.model_runner.call("run", [pair.target], True, True, True)

		accepted = 1
		correction = None
		for i in range(1, len(drafted)):
			pred = int(verify_logits[i - 1].argmax(dim=-1).item())
			if pred == drafted[i]:
				accepted += 1
			else:
				correction = pred
				break

		if correction is not None:
			target_bm.rewind(pair.target, old_len + accepted)
			self._prepare_slot_for_next_token(pair.target, target_bm)
			pair.target.append_token(correction)
			committed = drafted[:accepted] + [correction]
			self._sync_draft_to_target(pair, old_len, committed)
			return committed

		bonus = int(verify_logits[len(drafted) - 1].argmax(dim=-1).item())
		self._prepare_slot_for_next_token(pair.target, target_bm)
		pair.target.append_token(bonus)
		committed = drafted + [bonus]
		self._sync_draft_to_target(pair, old_len, committed)
		return committed

	def _generate_one(self, prompt: str | list[int], sampling_params: SamplingParams) -> list[int]:
		pair = self._init_sequence(prompt, sampling_params)
		eos = self.target_engine.scheduler.eos
		while pair.target.num_completion_tokens < pair.target.max_tokens:
			remaining = pair.target.max_tokens - pair.target.num_completion_tokens
			drafted = self._draft_decode_k(pair, min(self.k, remaining))
			committed = self._verify_and_commit(pair, drafted)
			if committed and (not pair.target.ignore_eos) and committed[-1] == eos:
				break
		return pair.target.completion_token_ids

	def generate(
		self,
		prompts: list[str] | list[list[int]],
		sampling_params: SamplingParams | list[SamplingParams],
		use_tqdm: bool = True,
	):
		if not isinstance(sampling_params, list):
			sampling_params = [sampling_params] * len(prompts)
		outputs = []
		pbar = tqdm(total=len(prompts), desc="Speculative Generating", dynamic_ncols=True) if use_tqdm else None
		for prompt, sp in zip(prompts, sampling_params):
			token_ids = self._generate_one(prompt, sp)
			outputs.append({"text": self.target_engine.tokenizer.decode(token_ids), "token_ids": token_ids})
			if pbar is not None:
				pbar.update(1)
		if pbar is not None:
			pbar.close()
		return outputs
