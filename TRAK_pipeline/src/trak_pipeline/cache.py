"""Portable, independently committed stage caches with content verification."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from .identity import scientific_key, digest
from .types import Artifact, StageResult

SCHEMA_VERSION = 1
SMALL_FILE_BYTES = 2 * 1024 * 1024
SAMPLE_BYTES = 64 * 1024

class CacheError(RuntimeError):
    pass


def file_fingerprint(path, small_bytes=SMALL_FILE_BYTES):
    path = Path(path)
    st = path.stat()
    if not path.is_file():
        raise CacheError('Expected regular file: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as f:
        if st.st_size <= small_bytes:
            for part in iter(lambda: f.read(1024 * 1024), b''):
                h.update(part)
            kind = 'complete_sha256'
        else:
            h.update(f.read(SAMPLE_BYTES))
            f.seek(max(0, st.st_size - SAMPLE_BYTES))
            h.update(f.read(SAMPLE_BYTES))
            kind = 'first_last_64KiB'
    return {'bytes': st.st_size, 'digest': h.hexdigest(), 'kind': kind,
            'mtime_ns_provenance_only': st.st_mtime_ns}


def fingerprint_identity(descriptor):
    return {k: descriptor[k] for k in ('bytes', 'digest', 'kind')}


def source_identity(path):
    return digest({'source_file': fingerprint_identity(file_fingerprint(path))})


def atomic_json(path, value):
    """Unique temporary names prevent parallel stage state writers from colliding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
            from .io import safe
            json.dump(safe(value), f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        from .io import replace
        replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def portable_path(path, data_root, package_root=None):
    p = Path(path).resolve()
    roots = [('data', Path(data_root).resolve())]
    if package_root:
        roots.append(('package', Path(package_root).resolve()))
    for name, root in roots:
        try:
            return {'root': name, 'relative': p.relative_to(root).as_posix()}
        except ValueError:
            pass
    raise CacheError('Artifact path is outside portable data/package roots: ' + str(p))


def resolve_portable(record, data_root, package_root=None):
    name = record.get('root')
    root = Path(data_root) if name == 'data' else Path(package_root) if name == 'package' and package_root else None
    if root is None:
        raise CacheError('Unknown portable path root: ' + str(name))
    raw = str(record.get('relative', '')).replace('\\', '/')
    if raw.startswith('/') or ':' in raw or '..' in Path(raw).parts:
        raise CacheError('Invalid portable relative path: ' + raw)
    p = (root / raw).resolve()
    if not p.is_relative_to(root.resolve()):
        raise CacheError('Portable path escaped its root')
    return p


def describe_path(path, data_root, package_root=None):
    p = Path(path)
    result = {'location': portable_path(p, data_root, package_root)}
    if p.is_dir():
        result.update(type='directory', entries={})
        for f in sorted(p.rglob('*')):
            if f.is_file() and not f.name.startswith('._') and not f.name.endswith('.tmp') and f.name != 'artifact.json':
                result['entries'][f.relative_to(p).as_posix()] = file_fingerprint(f)
    elif p.is_file():
        result.update(type='file', fingerprint=file_fingerprint(p))
    else:
        raise CacheError('Stage output is missing: ' + str(p))
    return result


def verify_descriptor(record, data_root, package_root=None):
    try:
        p = resolve_portable(record['location'], data_root, package_root)
        if record['type'] == 'file':
            actual = file_fingerprint(p)
            return fingerprint_identity(actual) == fingerprint_identity(record['fingerprint'])
        if record['type'] == 'directory':
            if not p.is_dir():
                return False
            actual_files = {f.relative_to(p).as_posix() for f in p.rglob('*')
                            if f.is_file() and not f.name.startswith('._') and not f.name.endswith('.tmp') and f.name != 'artifact.json'}
            if actual_files != set(record['entries']):
                return False
            return all(fingerprint_identity(file_fingerprint(p / rel)) == fingerprint_identity(fp)
                       for rel, fp in record['entries'].items())
        return False
    except (OSError, ValueError, KeyError, CacheError):
        return False


class CacheStore:
    def __init__(self, config, sample):
        self.config = config
        self.sample = sample
        self.data_root = Path(config['data_root']).resolve()
        self.package_root = Path(config.get('package_root', Path(__file__).parents[2])).resolve()
        self.root = Path(sample.directory) / 'analysis_results' / 'modular_cache'
        if not self.root.resolve().is_relative_to(self.data_root):
            raise CacheError('Sample cache escaped portable data root')
        self.last_miss_reason = None

    @staticmethod
    def key(spec, parameters, inputs):
        return scientific_key(spec.method_id, spec.method_version, parameters,
                              {d: inputs[d].id for d in spec.dependencies})

    def output_dir(self, spec, parameters, inputs):
        # Pure path computation permits read-only plan and refresh modes.
        return self.root / spec.stage / self.key(spec, parameters, inputs)

    def lookup(self, spec, parameters, inputs):
        out = self.output_dir(spec, parameters, inputs)
        marker = out / 'artifact.json'
        self.last_miss_reason = None
        try:
            data = json.loads(marker.read_text(encoding='utf-8'))
            key = self.key(spec, parameters, inputs)
            if data.get('schema_version') != SCHEMA_VERSION or not data.get('complete'):
                raise CacheError('Incomplete or unsupported cache commit')
            if data['artifact']['id'] != key or data['artifact']['stage'] != spec.stage:
                raise CacheError('Artifact scientific identity mismatch')
            descriptors = data.get('descriptors', {})
            if not descriptors or not all(verify_descriptor(d, self.data_root, self.package_root) for d in descriptors.values()):
                raise CacheError('Committed stage output missing or content fingerprint changed')
            payload = dict(data['artifact'])
            payload['files'] = {k: str(resolve_portable(v['location'], self.data_root, self.package_root))
                                for k, v in descriptors.items()}
            return Artifact(**payload)
        except FileNotFoundError:
            self.last_miss_reason = 'no_complete_commit'
            return None
        except (OSError, ValueError, KeyError, CacheError) as e:
            self.last_miss_reason = str(e)
            return None

    def commit(self, spec, parameters, inputs, result: StageResult):
        out = self.output_dir(spec, parameters, inputs)
        from .io import guard, safe
        out = guard(out, self.data_root)
        out.mkdir(parents=True, exist_ok=True)
        if not result.files:
            raise CacheError('Cannot commit a stage without scientific outputs')
        files = {name: str(Path(path).resolve()) for name, path in result.files.items()}
        for name, path in files.items():
            owned = Path(path).is_relative_to(out.resolve())
            presentation_root = Path(self.sample.directory) / ('Analysis_Figures_modular_' + self.key(spec, parameters, inputs)[:12])
            published = spec.stage == 'presentation' and not spec.scientific and Path(path).is_relative_to(presentation_root.resolve())
            if not (owned or published):
                raise CacheError('New scientific stage output must be stage-owned; presentation is restricted to its declared figure directory: ' + name)
            guard(path, self.data_root)
        descriptors = {name: describe_path(path, self.data_root, self.package_root) for name, path in files.items()}
        # Writers must close memmaps/streams before returning StageResult. Hashes,
        # not mtimes, determine validity after Windows closes writable mappings.
        art = Artifact(self.key(spec, parameters, inputs), spec.stage, files,
                       safe(result.metadata), dict(parameters),
                       {d: inputs[d].id for d in spec.dependencies}, spec.method_id, spec.method_version)
        payload = dataclasses.asdict(art)
        payload['files'] = {name: descriptors[name]['location'] for name in files}
        atomic_json(out / 'artifact.json', {'schema_version': SCHEMA_VERSION, 'complete': True,
                    'artifact': payload, 'descriptors': descriptors})
        return art
