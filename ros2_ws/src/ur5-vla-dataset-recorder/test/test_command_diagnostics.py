import sys
from pathlib import Path
import threading
import time
import json
from concurrent.futures import Future
from collections import deque
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'scripts'))
from command_diagnostics import QueuedCommandLogger
from planned_transport import PlannedBowlTransport
from test_execution_profile import harness,Message


def test_actual_transport_handoff_keeps_logging_off_command_callback(tmp_path):
    logger=QueuedCommandLogger();backend_started=threading.Event();backend_done=threading.Event()
    def slow_log(message):
        backend_started.set();time.sleep(.75);backend_done.set()
    backend=SimpleNamespace(info=slow_log,error=slow_log,warning=slow_log)
    h=harness('policy_executor.py','PolicyExecutor',{'publish_planned_transport'},
              dict(time=time,Float64MultiArray=Message))
    h.command_logger=logger;h.get_logger=lambda:backend
    h.queue_lock=threading.Lock();h.action_queue=deque([[.01]*6+[0.]])
    h.raw_action_queue=deque([[.01]*6+[0.]]);h.inference_generation=0;h.offset_fault=False
    h.last_action=[0.]*7;h.buffer=SimpleNamespace(snapshot=lambda *a:None)
    h.planned_transport=PlannedBowlTransport(SimpleNamespace(),logger,tmp_path)
    h.planned_transport.stage='requested';h.planned_transport.closed=0.;h.planned_transport.closed_at=time.monotonic()
    messages=[];h.publisher=SimpleNamespace(publish=lambda m:messages.append((time.monotonic(),m.data)))
    before=time.monotonic();h.publish_planned_transport([0.]*6,before)
    assert time.monotonic()-before<.15
    assert len(messages)==1 and not backend_started.is_set()
    assert not h.action_queue and not h.raw_action_queue
    worker=threading.Thread(target=logger.flush,args=(backend,));worker.start()
    assert backend_started.wait(.5)
    for _ in range(3):
        time.sleep(.05);now=time.monotonic();h.planned_transport.closed_at=now
        h.publish_planned_transport([0.]*6,now)
    assert len(messages)==4 and not backend_done.is_set() and not h.offset_fault
    worker.join(2)
    assert backend_done.is_set()


def test_diagnostics_preserve_severity_and_order():
    queue=QueuedCommandLogger();seen=[]
    queue.warning('first');queue.error('fault');queue.info('context')
    backend=SimpleNamespace(**{level:(lambda message,level=level:seen.append((level,message))) for level in ['info','warning','error']})
    queue.flush(backend)
    assert seen==[('warning','first'),('error','fault'),('info','context')]


def test_mixed_severity_flush_uses_valid_rclpy_call_sites():
    import pytest
    logging=pytest.importorskip('rclpy.logging')
    queue=QueuedCommandLogger()
    for _ in range(2):
        queue.info('diagnostic context')
        queue.warning('diagnostic warning')
        queue.error('diagnostic fault')
    # The old single getattr(...)(...) call site raised ValueError here.
    queue.flush(logging.get_logger('mixed_severity_regression'))
    assert queue.messages.empty()


def engine_status_harness(fault):
    h=harness('policy_executor.py','PolicyExecutor',{'check_engine_status'},
              dict(json=json,Trigger=SimpleNamespace(Request=lambda:object())))
    h.offset_fault=False;h.queue_lock=threading.Lock()
    h.action_queue=deque([[.01]*7]);h.raw_action_queue=deque([[.02]*7])
    h.inference_generation=4;h.command_logger=QueuedCommandLogger()
    future=Future();future.set_result(SimpleNamespace(success=True,message=json.dumps({'fault':fault})))
    h.engine_status_future=future;h.requests=[]
    h.engine_status_client=SimpleNamespace(service_is_ready=lambda:True,
        call_async=lambda request:h.requests.append(request) or Future())
    return h


def test_engine_fault_discards_actions_and_invalidates_worker():
    h=engine_status_harness('joint feedback stale')
    h.check_engine_status()
    assert h.offset_fault and not h.action_queue and not h.raw_action_queue
    assert h.inference_generation==5 and not h.requests
    level,message=h.command_logger.messages.get_nowait()
    assert level=='error' and 'joint feedback stale' in message
    h.check_engine_status()
    assert h.inference_generation==5


def test_healthy_engine_and_pending_status_do_not_block_feedback():
    h=engine_status_harness(None)
    h.check_engine_status()
    assert not h.offset_fault and h.action_queue and h.raw_action_queue
    assert h.inference_generation==4 and len(h.requests)==1
    h.check_engine_status()
    assert len(h.requests)==1 and not h.offset_fault
