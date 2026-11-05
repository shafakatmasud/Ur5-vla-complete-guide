"""Exercise GPU reservation lifetime through exec, and refusal before execution."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
WRAPPER=ROOT/'experiments/model_job.py'


class ModelJobTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.folder=Path(self.temp.name)
        self.fake=self.folder/'nvidia-smi'
        self.fake.write_text('#!/bin/sh\nprintf "0\\n"\n');self.fake.chmod(0o755)
        self.env=os.environ.copy();self.env['PATH']=str(self.folder)+os.pathsep+self.env['PATH']

    def tearDown(self):self.temp.cleanup()

    def test_gpu_lock_survives_exec_and_blocks_second_job(self):
        code="print('READY',flush=True); input()"
        command=[sys.executable,str(WRAPPER),'--gpus','97','--',sys.executable,'-u','-c',code]
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                 text=True,env=self.env)
        try:
            self.assertIn('RESERVATION',process.stdout.readline())
            self.assertEqual(process.stdout.readline().strip(),'READY')
            second=subprocess.run(command,input='\n',capture_output=True,text=True,env=self.env,timeout=5)
            self.assertNotEqual(second.returncode,0)
            self.assertIn('reserved by another',second.stderr)
            self.assertIsNone(process.poll())
        finally:
            process.communicate('\n',timeout=5)
        self.assertEqual(process.returncode,0)
        third=subprocess.run(command,input='\n',capture_output=True,text=True,env=self.env,timeout=5)
        self.assertEqual(third.returncode,0,third.stderr)

    def test_busy_gpu_rejects_before_running_command(self):
        self.fake.write_text('#!/bin/sh\nprintf "1301\\n"\n')
        marker=self.folder/'should_not_exist'
        command=[sys.executable,str(WRAPPER),'--gpus','98','--',sys.executable,'-c',
                 f"from pathlib import Path; Path({str(marker)!r}).touch()"]
        result=subprocess.run(command,capture_output=True,text=True,env=self.env,timeout=5)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('already using1301',result.stderr.replace('using 1301','using1301'))
        self.assertFalse(marker.exists())


if __name__=='__main__':unittest.main()
