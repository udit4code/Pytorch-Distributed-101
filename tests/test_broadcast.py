"""Native collective subprocess coverage (requires implementing the demo TODO)."""
import json, os, platform, shutil, socket, subprocess
import pytest

@pytest.mark.skipif(os.environ.get("PHASE2_RUN_DISTRIBUTED") != "1", reason="opt in to torchrun subprocess tests")
@pytest.mark.parametrize("src,value", [(0,100),(2,999)])
def test_broadcast_all_ranks_receive_root_value(src, value):
    torchrun = shutil.which("torchrun")
    if not torchrun: pytest.skip("torchrun unavailable")
    with socket.socket() as s: s.bind(("127.0.0.1",0)); port=s.getsockname()[1]
    env=os.environ.copy(); env["GLOO_SOCKET_IFNAME"]="lo0" if platform.system()=="Darwin" else "lo"
    p=subprocess.run([torchrun,"--standalone","--nproc-per-node=4","-m","phase2.broadcast_demo","--src",str(src)],capture_output=True,text=True,timeout=30,env=env)
    assert p.returncode==0,p.stdout+p.stderr
    records=[json.loads(x) for x in p.stdout.splitlines() if x.startswith("{") and '"phase": "after"' in x]
    assert len(records)==4 and all(r["value"]==value for r in records)
