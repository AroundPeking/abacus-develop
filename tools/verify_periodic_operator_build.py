"""Accept a completed build without rebuilding; preserve any failed wrapper status."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(root, expected_commit):
    source, build = root/'source', root/'build'
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args])
    if git('rev-parse', 'HEAD').decode().strip() != expected_commit or git('status', '--porcelain').strip():
        raise ValueError('source commit or clean-tree contract mismatch')
    if expected_commit not in (build/'commit.h').read_text():
        raise ValueError('compiled commit mismatch')
    cache = (build/'CMakeCache.txt').read_text()
    for flag in ('ENABLE_MPI', 'ENABLE_LCAO', 'ENABLE_LIBRI', 'ENABLE_LIBCOMM', 'DEBUG_INFO'):
        if flag + ':BOOL=ON\n' not in cache:
            raise ValueError('missing build feature: ' + flag)
    tests = subprocess.check_output(['ctest', '--test-dir', str(build), '--output-on-failure', '-R',
        '^(MODULE_IO_read_item_serial|MODULE_RI_sternheimer_(basis_opt_periodic|abacus_st_smoke)_test)$'],
        stderr=subprocess.STDOUT).decode()
    if '100% tests passed, 0 tests failed out of 3' not in tests:
        raise ValueError('expected three registered test targets')
    files, gitlinks = {}, {}
    for entry in git('ls-files', '--stage', '-z').split(b'\0'):
        if not entry:
            continue
        metadata, path = entry.decode().split('\t', 1)
        mode, object_id, stage = metadata.split()
        if stage != '0':
            raise ValueError('unmerged source index')
        if mode == '160000':
            gitlinks[path] = object_id
        else:
            files[path] = sha(source/path)
    result = dict(status='success', scope='build_and_unit_tests_only', source_commit=expected_commit,
        verification_job_id=os.environ['SLURM_JOB_ID'], binary_sha256=sha(build/'abacus_3p'),
        cache_sha256=sha(build/'CMakeCache.txt'), source_files=files, gitlinks=gitlinks,
        ctest=tests, original_wrapper_status=(root/'result/BUILD_STATUS').read_text(),
        build_repeated=False, physical_admission=False)
    with (root/'result/BUILD_ACCEPTANCE.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_files', 'ctest')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('commit')
    args = parser.parse_args()
    verify(args.root, args.commit)
