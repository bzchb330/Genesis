"""Record existing validation/artifact provenance; no physics integration."""
import hashlib
import re
import subprocess

import numpy as np

from seqgrasp import phase3cp05r as p


def git(*args):
    return subprocess.run(['git',*args],cwd=p.ROOT,capture_output=True,text=True)


def main():
    raw=(p.OUTPUT/'pytest.log').read_bytes()
    log=raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8',errors='replace')
    match=re.search(r'(\d+ passed, \d+ warnings in [\d.]+s)',log)
    if not match or 'FAILED' in log or 'ERROR ' in log:
        raise RuntimeError('Complete passing pytest log required')
    check=git('diff','--check')
    assert check.returncode==0 and not check.stdout and not check.stderr
    pending=git('ls-files','--others','--exclude-standard').stdout.splitlines()
    text_checked=[]
    for path in pending:
        if (p.ROOT/path).suffix in ('.py','.md','.yaml'):
            result=git('-c','core.autocrlf=false','diff','--no-index','--check','--','NUL',path)
            assert result.returncode in (0,1) and not result.stdout and not result.stderr,(path,result)
            text_checked.append(path)
    preserved=p.read('protocol.json')['preserved_p05_outputs']
    assert all(hashlib.sha256((p.ROOT/path).read_bytes()).hexdigest()==sha for path,sha in preserved.items())
    summary=p.read('summary.json'); extra=[]
    for row in summary['trials']:
        trace=p.old.load_trace(row['trace'])
        extra.append(dict(physics_name=row['physics_name'],dt_s=row['nominal_dt_s'],
            solver_iterations_mean=float(np.mean([x['solver']['iterations'] for x in trace])),
            samples=len(trace),makes=row['makes'],breaks=row['breaks']))
    snap=summary['equilibrium_audit'][0]['snapshot']
    confirmation=p.old.load_trace(snap['restore_confirmation_trace'])
    drift=max(float(np.max(np.abs(np.asarray(x['qpos'][:25])-snap['qpos_eq'][:25]))) for x in confirmation)
    p.save('supplemental_descriptors.json',dict(physics_names=p.config()['candidates'],physics_steps=0,
        trials=extra,confirmation_max_hand_qpos_drift_rad=drift))
    figures=p.read('figures.json')['figures']; videos=p.read('videos.json')['generated']
    assert len(figures)==20 and len(videos)==4
    assert all((p.ROOT/x).is_file() for x in figures)
    assert all((p.ROOT/x['path']).is_file() for x in videos)
    status=git('status','--short').stdout
    assert not git('diff','--name-only').stdout and not git('diff','--cached','--name-only').stdout
    p.save('validation.json',dict(physics_names=p.config()['candidates'],
        pytest_command=r'.\.venv\Scripts\python.exe -m pytest -v',pytest_result=match.group(1),pytest_exit_code=0,
        git_diff_check_command='git diff --check',git_diff_check_exit_code=check.returncode,
        untracked_text_whitespace_checked=text_checked,p05_outputs_preserved_count=len(preserved),
        p05_outputs_preserved=True,branch=git('branch','--show-current').stdout.strip(),
        head=git('rev-parse','HEAD').stdout.strip(),git_status_short=status,
        intentionally_uncommitted=True,main_merge=False,all_figures_rendered_and_visually_reviewed=True,
        video_first_middle_last_frames_visually_reviewed=True,physics_steps=0))
    paths={x for x in p.OUTPUT.rglob('*') if x.is_file() and x.name!='artifact_manifest.json'}
    paths.update(p.ROOT/x for x in pending)
    manifest={x.relative_to(p.ROOT).as_posix():dict(bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in sorted(paths)}
    p.save('artifact_manifest.json',dict(physics_names=p.config()['candidates'],base_commit=summary['base_commit'],
        paths_relative_to_repository=True,artifacts=manifest,self_excluded=True))
    print(match.group(1))
    print('git diff --check: exit 0; new text whitespace checks:',len(text_checked))
    print('Preserved P0.5 outputs:',len(preserved),'Manifest artifacts:',len(manifest))
    print('Supplemental descriptors:',extra,'Confirmation hand drift:',drift)
    print(status)


if __name__=='__main__': main()
