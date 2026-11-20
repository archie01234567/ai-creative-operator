"""QA is the only thing standing between a bad render and a paid campaign."""

from conftest import needs_ffmpeg

from creative_operator.config import Settings
from creative_operator.models import CreativeScores
from creative_operator.qa import gate, technical_qa


@needs_ffmpeg
def test_technical_qa_accepts_spec_video_and_explains_each_rejection(brief, make_video, tmp_path):
    good = technical_qa(make_video("good.mp4", 12), brief)
    assert good.valid and good.playable
    assert (good.resolution, good.aspect_ratio) == ("1080x1920", "9:16")
    assert 11.9 <= good.duration <= 12.1

    landscape_short = technical_qa(make_video("bad.mp4", 4, size="1280x720"), brief)
    assert not landscape_short.valid
    reasons = " | ".join(landscape_short.failures)
    assert "aspect ratio" in reasons and "duration" in reasons

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(make_video("src.mp4", 12).read_bytes()[:20_000])  # truncated download
    report = technical_qa(corrupt, brief)
    assert report.exists and not report.valid and report.failures

    assert not technical_qa(tmp_path / "missing.mp4", brief).exists


def scores(**overrides) -> CreativeScores:
    base = dict(observed_defects=[], shot_by_shot=[], fix_suggestions=[], realism=8,
                visual_consistency=8, brief_adherence=8, hook_strength=8, advertising_quality=8)
    return CreativeScores(**{**base, **overrides})  # fmt: skip


def test_creative_gate_is_decided_by_thresholds_not_by_the_model():
    settings = Settings()

    assert gate(scores(), settings).passed

    # One unusable dimension sinks the ad even when the average still looks healthy.
    one_bad = gate(scores(realism=4, hook_strength=10, brief_adherence=10), settings)
    assert one_bad.mean_score >= settings.qa_min_mean
    assert not one_bad.passed and "realism" in one_bad.failures[0]

    # Uniformly mediocre: every dimension clears the floor, the mean does not.
    mediocre = gate(scores(realism=6, visual_consistency=6, brief_adherence=6,
                           hook_strength=6, advertising_quality=6), settings)  # fmt: skip
    assert not mediocre.passed and "mean" in mediocre.failures[0]
