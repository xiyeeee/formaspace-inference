from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    seed: int = Field(default=1234, ge=0, le=10_000_000)
    steps: int = Field(default=30, ge=2, le=50)
    guidanceScale: float = Field(default=5.0, ge=0, le=10)
    finalResolution: int = Field(default=256, ge=128, le=256)
    previewEnabled: bool = True
    previewStep: int = Field(default=24, ge=1, le=49)
    previewResolution: int = Field(default=128, ge=64, le=128)
    pointCount: int = Field(default=5000, ge=1, le=5000)
    numChunks: int = Field(default=4000, ge=1000, le=8000)

    @model_validator(mode="after")
    def check_profile(self):
        if self.finalResolution not in (128, 256) or self.previewResolution not in (64, 128):
            raise ValueError("Unsupported resolution")
        if self.previewEnabled and not 1 <= self.previewStep < self.steps:
            raise ValueError("previewStep must be before the last sampling step")
        return self
