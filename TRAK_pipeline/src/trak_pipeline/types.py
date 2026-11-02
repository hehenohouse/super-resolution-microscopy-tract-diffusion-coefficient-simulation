from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class Sample:
    sample_id: str
    directory: Path
    metadata: dict[str, Any]
    C1_file: Path
    C2_file: Path
    ROI_file: Path
    legacy_output: Path | None = None
    legacy_figures: Path | None = None
    @property
    def id(self): return self.sample_id

@dataclass
class Artifact:
    id: str
    stage: str
    files: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    upstream_ids: dict[str, str] = field(default_factory=dict)
    method_id: str = ''
    method_version: str = '1'
    def file(self, name):
        if name not in self.files: raise KeyError(f'{self.stage} artifact lacks file {name}')
        return Path(self.files[name])

@dataclass
class StageResult:
    files: dict[str, str | Path]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class StageContext:
    config: dict[str, Any]
    sample: Sample
    parameters: dict[str, Any]
    inputs: dict[str, Artifact]
    output_dir: Path
    package_root: Path
    runtime: dict[str, str] = field(default_factory=dict)
    def input(self, stage): return self.inputs[stage]
    def file(self, stage, name): return self.inputs[stage].file(name)
    def progress(self, message, **values):
        import json
        print(json.dumps(dict(sample_id=self.sample_id,stage=self.parameters.get('_stage',''),message=message,**values)),flush=True)
    @property
    def sample_id(self): return self.sample.sample_id
