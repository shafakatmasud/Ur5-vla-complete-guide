"""Planning requests are pumped by their caller, never a background executor."""
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
from planning_service_node import PlanningServiceNode


def test_concurrent_workers_serialize_request_and_response_on_the_same_thread():
    node = object.__new__(PlanningServiceNode)
    node.service_lock = threading.Lock()
    node.request_count = node.response_timeout_count = 0
    active = []; peak = []; errors = []
    def request(value):
        active.append(value);peak.append(len(active))
        f=SimpleNamespace(complete=False, caller=threading.get_ident())
        f.done=lambda:f.complete
        return f
    def spin(future, timeout_sec):
        assert future.caller == threading.get_ident()
        assert timeout_sec > 0
        time.sleep(.01)
        future.complete=True
        active.pop()
    node.service_executor = SimpleNamespace(spin_until_future_complete=spin)
    client = SimpleNamespace(call_async=request)
    def worker(value):
        try:assert node.request_and_wait(client,value,1.).done()
        except Exception as error:errors.append(error)
    threads=[threading.Thread(target=worker,args=(i,)) for i in range(4)]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    assert not errors
    assert max(peak)==1 and node.request_count==4 and node.response_timeout_count==0


def test_unanswered_future_is_returned_pending_for_fail_closed_cancellation():
    node = object.__new__(PlanningServiceNode)
    node.service_lock = threading.Lock()
    node.request_count = node.response_timeout_count = 0
    future=SimpleNamespace(done=lambda:False)
    node.service_executor=SimpleNamespace(spin_until_future_complete=lambda *args,**kwargs:None)
    assert node.request_and_wait(SimpleNamespace(call_async=lambda _:future),object(),.01) is future
    assert node.response_timeout_count==1


def test_closed_worker_does_not_send_another_request():
    node=object.__new__(PlanningServiceNode)
    node.service_lock=threading.Lock()
    node.service_closed=True
    sent=[]
    with pytest.raises(RuntimeError,match='worker has stopped'):
        node.request_and_wait(SimpleNamespace(call_async=lambda req:sent.append(req)),object(),.01)
    assert not sent
