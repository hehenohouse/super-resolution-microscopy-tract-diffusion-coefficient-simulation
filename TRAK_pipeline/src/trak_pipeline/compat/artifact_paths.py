"""Explicit, checksum-bound relocation of public figure directories only.

Historical completion JSONs and portable registries remain read-only. No path
search, best-match inference, source-image relocation, or scientific fallback.
"""
from pathlib import Path, PurePosixPath, PureWindowsPath
import hashlib
import json
import re


class ArtifactPathAliasError(ValueError):
    pass


_VERIFIED = set()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root, value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ArtifactPathAliasError('Expected an explicit portable relative figure path')
    relative = PurePosixPath(value)
    if relative.is_absolute() or PureWindowsPath(value).drive or '..' in relative.parts:
        raise ArtifactPathAliasError('Absolute or traversing alias path rejected')
    if any(part.startswith('.') for part in relative.parts):
        raise ArtifactPathAliasError('Hidden alias paths are not public figure artifacts')
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ArtifactPathAliasError('Figure alias escapes the data root')
    return path


def _candidate(path, root):
    original = Path(path)
    try:
        original.resolve().relative_to(root)
        return original.resolve()
    except ValueError:
        normalized = str(path).replace('\\', '/')
        token = '/' + root.name + '/'
        if token in normalized:
            return _relative(root, normalized.rsplit(token, 1)[1])
        return None


def _verify_alias(root, alias, old, new, manifest_sha):
    if old.exists():
        raise ArtifactPathAliasError('Ambiguous figure relocation: original directory exists')
    if not new.is_dir():
        raise ArtifactPathAliasError('Explicit relocated figure directory is missing')
    if not old.name.startswith('Analysis_Figures_') or old.parent != new.parent:
        raise ArtifactPathAliasError('Aliases may relocate figure directories within one cell only')
    match = re.fullmatch(r'(WT|TDK|MDK)_TRAK1_(BR\d+)_(T\d+)', str(alias.get('sample_id', '')))
    if not match or old.parent.parent.name != 'TRAK1_' + match[1] + '_' + match[2] or old.parent.parent.parent.name != match[1]:
        raise ArtifactPathAliasError('Figure alias sample identity disagrees with its directory')
    completion = _relative(root, alias['source_completion_relative'])
    if not completion.is_relative_to(old.parent / 'analysis_results') or completion.name != 'COMPLETED.json':
        raise ArtifactPathAliasError('Figure alias completion must belong to this cell')
    records = alias.get('registered_artifacts', [])
    if not records or len(records) != alias.get('verified_original_artifact_count'):
        raise ArtifactPathAliasError('Figure alias has no complete registered artifact list')
    stats = []
    for record in records:
        actual = _relative(new, record['suffix'])
        if not actual.is_file() or actual.stat().st_size != record['bytes']:
            raise ArtifactPathAliasError('Relocated figure artifact is missing or has changed size')
        st = actual.stat()
        stats.append((str(actual), st.st_size, st.st_mtime_ns))
    completion_stat = completion.stat()
    key = (manifest_sha, str(old), str(new), completion_stat.st_size, completion_stat.st_mtime_ns, tuple(stats))
    if key in _VERIFIED:
        return
    if _sha(completion) != alias['source_completion_sha256']:
        raise ArtifactPathAliasError('Historical completion JSON changed after alias verification')
    original = json.loads(completion.read_text(encoding='utf-8'))
    if original.get('complete') is not True or original.get('sample_id') != alias['sample_id'] or original.get('scientific_fingerprint') != alias['scientific_fingerprint']:
        raise ArtifactPathAliasError('Figure alias does not match a committed scientific result')
    registered = {}
    for record in original.get('artifacts', []):
        candidate = _candidate(record['path'], root)
        if candidate is not None and candidate.is_relative_to(old):
            registered[candidate.relative_to(old).as_posix()] = record
    if set(registered) != {record['suffix'] for record in records}:
        raise ArtifactPathAliasError('Alias artifact list differs from historical completion')
    for record in records:
        prior = registered[record['suffix']]
        if prior['bytes'] != record['bytes'] or prior.get('sha256') != record.get('sha256'):
            raise ArtifactPathAliasError('Alias artifact fingerprint differs from historical completion')
        if _sha(_relative(new, record['suffix'])) != record['sha256']:
            raise ArtifactPathAliasError('Relocated figure artifact SHA256 changed')
    _VERIFIED.add(key)


def resolve_artifact_path(path, data_root, control, binding=None):
    """Resolve only a frozen, verified figure-prefix alias; keep other paths exact."""
    original = Path(path)
    root, control = Path(data_root).resolve(), Path(control).resolve()
    manifest = control / 'figure_directory_aliases.json'
    if not manifest.is_file():
        if binding is not None and Path(binding).is_file():
            raise ArtifactPathAliasError('Frozen figure alias manifest is missing')
        return original
    if binding is None or not Path(binding).is_file():
        raise ArtifactPathAliasError('Figure aliases require a frozen checksum binding')
    binding_data = json.loads(Path(binding).read_text(encoding='utf-8'))
    if binding_data.get('relative_path') != 'figure_directory_aliases.json':
        raise ArtifactPathAliasError('Unexpected figure alias binding target')
    manifest_data = manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    if manifest_sha != binding_data.get('sha256'):
        raise ArtifactPathAliasError('Figure alias manifest checksum differs from the frozen release')
    value = json.loads(manifest_data)
    if value.get('schema_version') != 1 or value.get('data_root_name') != root.name:
        raise ArtifactPathAliasError('Figure alias schema or data root mismatch')
    candidate = _candidate(path, root)
    if candidate is None:
        return original
    matches = []
    for alias in value.get('aliases', []):
        old, new = _relative(root, alias['old_relative']), _relative(root, alias['new_relative'])
        if candidate == old or candidate.is_relative_to(old):
            matches.append((alias, old, new))
    if not matches:
        return original
    if len(matches) != 1:
        raise ArtifactPathAliasError('Overlapping figure aliases are ambiguous')
    alias, old, new = matches[0]
    _verify_alias(root, alias, old, new, manifest_sha)
    actual = new / candidate.relative_to(old)
    if not actual.exists():
        raise ArtifactPathAliasError('Requested relocated figure artifact is missing')
    return actual
