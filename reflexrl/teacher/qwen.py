"""Qwen3-VL teacher: frames -> action distribution over the scenario vocabulary.

One batched forward pass per call; the distribution is the softmax of the
next-token logits restricted to the answer letters. Nothing is generated.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from reflexrl.env.scenarios import Scenario
from reflexrl.teacher.prompts import action_letters, build_messages

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
# Below this total next-token probability on the answer letters the model is
# not answering the question, and a letter-renormalised distribution is noise.
MIN_LETTER_MASS = 0.5
FP32_ATTN = "fp32_eager"


def _fp32_eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0,
                          **kwargs):
    """Eager attention with scores and softmax in fp32.

    On GPUs without bf16 (Turing) Qwen3-VL runs in fp16, where the q.k^T
    product can overflow before scaling and poison the logits with NaN.
    Upcasting only the (tiny, ~300-token) score computation fixes it.
    """
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:  # grouped-query attention
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    scores = torch.matmul(query.float(), key.float().transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]].float()
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, value.float()).to(value.dtype)
    return out.transpose(1, 2).contiguous(), probs


def _register_fp32_attention() -> None:
    from transformers import AttentionInterface
    AttentionInterface.register(FP32_ATTN, _fp32_eager_attention)


def default_dtype(device: str) -> torch.dtype:
    if not device.startswith("cuda"):
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    # No bf16 before Ampere. fp16 is NOT a fallback: Qwen3-VL's ~1.5e4
    # residual activations wreck it (gibberish generations, ~0.03% mass on
    # the answer letters; see results/teacher_diagnosis). Use fp32 instead.
    return torch.bfloat16 if major >= 8 else torch.float32


class QwenTeacher:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cuda",
                 dtype: torch.dtype | None = None, attn_implementation: str | None = None,
                 device_map: str | None = None, load_4bit: bool = False):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_id = model_id
        self.device = device
        self.dtype = dtype or default_dtype(device)
        if attn_implementation is None:
            attn_implementation = FP32_ATTN if self.dtype == torch.float16 else "sdpa"
        if attn_implementation == FP32_ATTN:
            _register_fp32_attention()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"  # last position = answer slot
        quant = None
        if load_4bit:
            # 4-bit NF4 weights, fp32 compute: fits an 8B model on one T4 without
            # ever doing the fp16 arithmetic that corrupts Qwen3-VL.
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=self.dtype,
                                       bnb_4bit_use_double_quant=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=self.dtype, device_map=device_map or device,
            attn_implementation=attn_implementation, quantization_config=quant)
        self.model.eval()
        self._prompt_cache: dict[tuple[str, int], str] = {}
        self._letter_ids_cache: dict[str, torch.Tensor] = {}
        self.calls = 0
        self.samples = 0
        self.seconds = 0.0
        self.nonfinite_retries = 0
        self.letter_mass_sum = 0.0

    def _prompt(self, scenario: Scenario, n_frames: int) -> str:
        key = (scenario.name, n_frames)
        if key not in self._prompt_cache:
            self._prompt_cache[key] = self.processor.apply_chat_template(
                build_messages(scenario, n_frames), tokenize=False,
                add_generation_prompt=True)
        return self._prompt_cache[key]

    def _letter_ids(self, scenario: Scenario) -> torch.Tensor:
        if scenario.name not in self._letter_ids_cache:
            ids = []
            for letter in action_letters(scenario):
                tok = self.processor.tokenizer.encode(letter, add_special_tokens=False)
                if len(tok) != 1:
                    raise ValueError(f"answer letter {letter!r} is not a single token: {tok}")
                ids.append(tok[0])
            self._letter_ids_cache[scenario.name] = torch.tensor(ids, device=self.device)
        return self._letter_ids_cache[scenario.name]

    @torch.no_grad()
    def choice_probs(self, frames: list[list[np.ndarray]], question: str,
                     n_options: int) -> np.ndarray:
        """Generic lettered multiple choice: frames + question -> (B, n_options) probs."""
        content = [{"type": "image"} for _ in range(len(frames[0]))]
        content.append({"type": "text", "text": question})
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        ids = []
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_options]:
            tok = self.processor.tokenizer.encode(letter, add_special_tokens=False)
            ids.append(tok[0])
        images = [Image.fromarray(f) for obs in frames for f in obs]
        batch = self.processor(text=[prompt] * len(frames), images=images,
                               return_tensors="pt", padding=True)
        batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
        logits = self.model(**batch).logits[:, -1, :].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("teacher produced non-finite logits")
        full = torch.softmax(logits, -1).index_select(1, torch.tensor(ids, device=logits.device))
        mass = full.sum(-1)
        self.letter_mass_sum += float(mass.sum())
        if float(mass.min()) < MIN_LETTER_MASS:
            raise RuntimeError(f"teacher is not answering: letter mass {float(mass.min()):.4f}")
        return (full / mass[:, None]).cpu().numpy().astype(np.float32)

    def action_probs(self, frames: list[list[np.ndarray]], scenario: Scenario,
                     retries: int = 3) -> np.ndarray:
        """frames: B lists of full-res RGB frames (oldest first) -> (B, A) float32 probs.

        A non-finite forward is retried, then raised; it is never replaced by
        a made-up distribution.
        """
        for attempt in range(retries + 1):
            try:
                return self._action_probs(frames, scenario)
            except FloatingPointError:
                self.nonfinite_retries += 1
                if attempt == retries:
                    raise
        raise AssertionError("unreachable")

    @torch.no_grad()
    def _action_probs(self, frames: list[list[np.ndarray]], scenario: Scenario) -> np.ndarray:
        n_frames = len(frames[0])
        text = [self._prompt(scenario, n_frames)] * len(frames)
        images = [Image.fromarray(f) for obs in frames for f in obs]
        t0 = time.perf_counter()
        batch = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
        logits = self.model(**batch).logits[:, -1, :].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("teacher produced non-finite logits")
        full = torch.softmax(logits, dim=-1).index_select(1, self._letter_ids(scenario).to(logits.device))
        mass = full.sum(-1)
        self.letter_mass_sum += float(mass.sum())
        if float(mass.min()) < MIN_LETTER_MASS:
            raise RuntimeError(f"teacher is not answering: letter mass {float(mass.min()):.4f} "
                               f"< {MIN_LETTER_MASS} (check dtype/prompt)")
        probs = (full / mass[:, None]).cpu().numpy()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        self.samples += len(frames)
        return probs.astype(np.float32)


class CalibratedQwenTeacher:
    """The probe-validated teacher (variant C): permutation-averaged, blank-calibrated.

    Each decision scores the action list in ``n_perm`` option orders (batched
    into one forward), maps probabilities back to canonical actions, averages
    them, divides by the same quantity measured on a blank frame (the model's
    prior over actions), and renormalises.
    """

    def __init__(self, teacher: QwenTeacher, n_perm: int = 4, seed: int = 0):
        self.teacher = teacher
        self.n_perm = n_perm
        self.seed = seed
        self._perms: dict[str, list[np.ndarray]] = {}
        self._prior: dict[str, np.ndarray] = {}

    def __getattr__(self, name):  # expose counters (calls, samples, seconds, ...)
        return getattr(self.teacher, name)

    def _perm_list(self, scenario: Scenario) -> list[np.ndarray]:
        if scenario.name not in self._perms:
            rng = np.random.default_rng(self.seed)
            n = len(scenario.actions)
            self._perms[scenario.name] = [np.arange(n)] + [rng.permutation(n) for _ in range(self.n_perm - 1)]
        return self._perms[scenario.name]

    def _question(self, scenario: Scenario, perm: np.ndarray) -> str:
        from reflexrl.teacher.prompts import LETTERS, build_prompt_text
        base = build_prompt_text(scenario)
        options = "\n".join(f"{LETTERS[k]}. {scenario.action_names[i]}" for k, i in enumerate(perm))
        return (base[: base.index("Choose the best action")]
                + f"Choose the best action right now:\n{options}\nAnswer with a single letter.")

    def _averaged(self, frames: list[list[np.ndarray]], scenario: Scenario) -> np.ndarray:
        n = len(scenario.actions)
        total = np.zeros((len(frames), n), np.float64)
        t0 = time.perf_counter()
        for perm in self._perm_list(scenario):
            p = self.teacher.choice_probs(frames, self._question(scenario, perm), n)
            total[:, perm] += p
        self.teacher.seconds += time.perf_counter() - t0
        return total / self.n_perm

    def action_probs(self, frames: list[list[np.ndarray]], scenario: Scenario) -> np.ndarray:
        if scenario.name not in self._prior:
            blank = [[np.zeros_like(f) for f in frames[0]]]
            self._prior[scenario.name] = self._averaged(blank, scenario)[0]
        p = self._averaged(frames, scenario) / self._prior[scenario.name]
        self.teacher.calls += 1
        self.teacher.samples += len(frames)
        return (p / p.sum(1, keepdims=True)).astype(np.float32)
