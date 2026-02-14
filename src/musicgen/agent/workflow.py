from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import torch

from musicgen.config_loader import Preset
from musicgen.codec import AudioCodec
from musicgen.inference import generate_music
from musicgen.model import MusicGenModel
from musicgen.text_encoder import TextEncoder
from musicgen.types import AgentFeedback, GenerationResult

FeedbackMode = Literal["none", "classifier", "llm"]


@dataclass
class AgentConfig:
    max_iterations: int = 3
    temperature: float = 1.0
    top_k: Optional[int] = 50
    duration_seconds: float = 5.0
    feedback_mode: FeedbackMode = "none"
    score_threshold: float = 0.9


class AgentWorkflow:
    def __init__(
        self,
        codec: AudioCodec,
        model: MusicGenModel,
        text_encoder: TextEncoder,
        device: torch.device,
        config: AgentConfig = AgentConfig(),
    ) -> None:
        self.codec = codec
        self.model = model
        self.text_encoder = text_encoder
        self.device = device
        self.config = config

    def generate(
        self, prompt: str, temperature: Optional[float] = None
    ) -> GenerationResult:
        return generate_music(
            prompt,
            self.codec,
            self.model,
            self.text_encoder,
            self.device,
            duration_seconds=self.config.duration_seconds,
            temperature=temperature or self.config.temperature,
            top_k=self.config.top_k,
        )

    def _get_feedback_fn(self) -> Optional[Callable[[GenerationResult], AgentFeedback]]:
        if self.config.feedback_mode == "none":
            return None
        if self.config.feedback_mode == "classifier":
            from musicgen.agent.feedback import classifier_feedback

            return classifier_feedback
        if self.config.feedback_mode == "llm":
            from musicgen.agent.feedback import llm_feedback

            def fn(r: GenerationResult) -> AgentFeedback:
                return llm_feedback(r, device=str(self.device))

            return fn
        return None

    def run(
        self,
        prompt: str,
        feedback_fn: Optional[Callable[[GenerationResult], AgentFeedback]] = None,
    ) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        current_prompt = prompt
        feedback_fn = feedback_fn or self._get_feedback_fn()

        for i in range(self.config.max_iterations):
            result = self.generate(current_prompt)
            results.append(result)

            if feedback_fn and i < self.config.max_iterations - 1:
                feedback = feedback_fn(result)
                feedback = AgentFeedback(
                    score=feedback.score,
                    feedback_text=feedback.feedback_text,
                    iteration=i,
                )
                if feedback.score >= self.config.score_threshold:
                    break
                current_prompt = f"{prompt}. Refinement: {feedback.feedback_text}"

        return results


def run_agent_workflow(
    prompt: str,
    codec_path: str,
    model_path: str,
    output_path: str,
    max_iterations: int = 3,
    feedback_mode: FeedbackMode = "none",
    preset: Preset = "small",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from musicgen.config_loader import get_text_embed_dim, get_text_encoder_name, load_preset

    codec_config, transformer_config = load_preset(preset)
    text_embed_dim = get_text_embed_dim(preset)
    codec = AudioCodec(codec_config)
    codec.load_state_dict(torch.load(codec_path, map_location="cpu"))
    codec = codec.to(device).eval()

    model = MusicGenModel(codec_config, transformer_config, text_embed_dim=text_embed_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(device).eval()

    text_encoder = TextEncoder(get_text_encoder_name(preset)).to(device).eval()

    workflow = AgentWorkflow(
        codec,
        model,
        text_encoder,
        device,
        AgentConfig(
            max_iterations=max_iterations,
            feedback_mode=feedback_mode,
        ),
    )
    results = workflow.run(prompt)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, result in enumerate(results):
        import soundfile as sf

        wav = result.waveform.cpu().numpy()
        if wav.ndim > 1:
            wav = wav[0]
        sf.write(
            output_dir / f"output_{i}.wav",
            wav,
            result.sample_rate,
        )
