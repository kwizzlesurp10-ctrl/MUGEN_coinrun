from __future__ import annotations

import pytest
import torch

from musicgen.agent.feedback import classifier_feedback
from musicgen.types import GenerationResult


def test_classifier_feedback() -> None:
    waveform = torch.randn(24000) * 0.3
    result = GenerationResult(
        waveform=waveform,
        sample_rate=24000,
        prompt="test",
        duration_seconds=1.0,
    )
    feedback = classifier_feedback(result)
    assert 0 <= feedback.score <= 1
    assert isinstance(feedback.feedback_text, str)
    assert len(feedback.feedback_text) > 0
