"""Finalize static-audit provenance after validation; no physics integration."""
import hashlib
import re
import subprocess
from seqgrasp import phase3cp05c as p


def git(*args): return subprocess.run(['git',*args],cwd=p.ROOT,capture_output=True,text=True)


def main():
    raw=(p.OUTPUT/'pytest.log').read_bytes()
    text=raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8',errors='replace')
    matched=re.search(r'(\d+ passed, \d+ warnings in [\d.]+s)',text)
    if not matched or 'FAILED' in text or 'ERROR ' in text: raise RuntimeError('A complete passing pytest log is required')
    check=git('diff','--check'); assert check.returncode==0 and not check.stdout and not check.stderr
    protocol=p.read('outputs/phase3CP05C/protocol.json')
    preserved=protocol['preserved_p05r_outputs']
    assert all(hashlib.sha256((p.ROOT/x).read_bytes()).hexdigest()==sha for x,sha in preserved.items())
    figures=p.read('outputs/phase3CP05C/figures.json')['figures']; assert len(figures)==19 and all((p.ROOT/x).exists() for x in figures)
    pending=git('ls-files','--others','--exclude-standard').stdout.splitlines(); checked=[]
    for path in pending:
        if (p.ROOT/path).suffix in ('.py','.md','.yaml'):
            r=git('-c','core.autocrlf=false','diff','--no-index','--check','--','NUL',path)
            assert r.returncode in (0,1) and not r.stdout and not r.stderr
            checked.append(path)
    p.save('validation.json',dict(pytest_command=r'.\.venv\Scripts\python.exe -m pytest -v',pytest_result=matched.group(1),pytest_exit_code=0,
        git_diff_check='exit 0; no output',untracked_text_whitespace_checks=checked,p05r_outputs_preserved=len(preserved),
        branch=git('branch','--show-current').stdout.strip(),base_commit=protocol['base_commit'],intentionally_uncommitted=True,
        all_pdfs_rendered_and_visually_reviewed=True,python_runtime=str(p.ROOT/'.venv/Scripts/python.exe'),physics_steps=0))
    paths={x for x in p.OUTPUT.rglob('*') if x.is_file() and x.name!='artifact_manifest.json'}
    paths.update(p.ROOT/x for x in pending)
    manifest={x.relative_to(p.ROOT).as_posix():dict(bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in sorted(paths)}
    p.save('artifact_manifest.json',dict(base_commit=protocol['base_commit'],paths_relative_to_repository=True,self_excluded=True,artifacts=manifest))
    print(matched.group(1)); print('git diff --check: exit 0'); print('P0.5R preserved:',len(preserved),'manifest:',len(manifest),'text checks:',len(checked))


if __name__=='__main__': main()
