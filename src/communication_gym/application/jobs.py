"""Durable jobs with atomic generation checks before committing model results."""
from __future__ import annotations
import asyncio
import time
from communication_gym.domain.models import GymError,new_id,utc_now

TERMINAL={'succeeded','failed','cancelled','interrupted'}

class JobContext:
    def __init__(self,manager,job):
        self.manager,self.job=manager,job
    def valid(self):
        repo=self.manager.repo
        job=repo.get('jobs',self.job['id'])
        if job['status']!='running': raise asyncio.CancelledError()
        if job.get('generation',0)!=self.job.get('generation',0):raise asyncio.CancelledError()
        sid=job.get('session_id')
        if sid and repo.get_session(sid).get('generation',0)!=job.get('session_generation',0):
            raise asyncio.CancelledError()
        aid=job.get('attempt_id')
        if aid and repo.get('attempts',aid).get('generation',0)!=job.get('attempt_generation',0):
            raise asyncio.CancelledError()
    def commit(self,callback):
        with self.manager.repo.transaction():
            self.valid()
            return callback()
    def progress(self,message,value=None):
        self.valid()
        job=self.manager.repo.get('jobs',self.job['id'])
        # Provider stage text is bounded and never persisted as arbitrary response content.
        safe_stage='working'
        lowered=str(message).lower()
        for name in ('loading','decoding','recording','transcribing','aligning','critic','listener','scoring','comparing','question','preparing'):
            if name in lowered: safe_stage=name; break
        self.manager.repo.put('jobs',{**job,'phase':safe_stage,'progress':max(0,min(1,float(value))) if value is not None else None})

class JobManager:
    def __init__(self,repo):
        self.repo=repo
        self.tasks:dict[str,asyncio.Task]={}
        self._closing=False
    def start(self,operation,work,*,session_id=None,attempt_id=None,operation_key=None):
        if self._closing:raise GymError('runtime_closing','The application is shutting down. Restart it before starting another action.',503,True)
        with self.repo.transaction():
            if operation_key:
                existing=self.repo.find('jobs',operation_key=operation_key)
                for row in existing:
                    if row.get('session_id')==session_id:
                        if row['operation']!=operation or row.get('attempt_id')!=attempt_id:
                            raise GymError('operation_conflict','This operation key belongs to another action.',409)
                        return row
            if session_id:
                active=[j for j in self.repo.find('jobs',session_id=session_id) if j['status'] not in TERMINAL]
                if active: raise GymError('job_running','Finish or cancel the current session action first.',409)
            session=self.repo.get_session(session_id) if session_id else {}
            attempt=self.repo.get('attempts',attempt_id) if attempt_id else {}
            job=self.repo.put('jobs',{'id':new_id(),'session_id':session_id,'attempt_id':attempt_id,'operation':operation,'operation_key':operation_key,'generation':0,'session_generation':session.get('generation',0),'attempt_generation':attempt.get('generation',0),'status':'queued','phase':'queued','progress':None})
        self.tasks[job['id']]=asyncio.create_task(self._run(job,work))
        return job
    async def _run(self,job,work):
        start=time.monotonic()
        try:
            job=self.repo.put('jobs',{**job,'status':'running','started_at':utc_now()})
            context=JobContext(self,job)
            async with asyncio.timeout(1800): result=await work(context)
            def finish():
                current=self.repo.get('jobs',job['id'])
                self.repo.put('jobs',{**current,'status':'succeeded','phase':'complete','progress':1.0,'result':result,'finished_at':utc_now(),'duration_seconds':round(time.monotonic()-start,3)})
            context.commit(finish)
        except asyncio.CancelledError:
            self._terminal(job,'cancelled','cancelled')
        except GymError as exc:
            self._terminal(job,'failed',exc.code,exc.message)
        except TimeoutError:
            self._terminal(job,'failed','job_timeout','This operation reached its deadline. Your saved recording is unchanged.')
        except Exception:
            self._terminal(job,'failed','subsystem_error','This subsystem could not finish. Your saved checkpoint is unchanged; retry the action.')
        finally:
            self.tasks.pop(job['id'],None)
    def _terminal(self,job,status,code,message=None):
        try:
            current=self.repo.get('jobs',job['id'])
            if current['status'] not in TERMINAL:
                self.repo.put('jobs',{**current,'status':status,'safe_error':code,'safe_message':message,'finished_at':utc_now()})
        except GymError: pass
    async def cancel(self,job_id):
        job=self.repo.get('jobs',job_id)
        if job['status'] in TERMINAL: return job
        self.repo.put('jobs',{**job,'status':'cancel_requested','generation':job.get('generation',0)+1})
        task=self.tasks.get(job_id)
        if task:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
        self._terminal(job,'cancelled','cancelled')
        return self.repo.get('jobs',job_id)
    async def cancel_session(self,sid):
        for job in self.repo.find('jobs',session_id=sid):
            if job['status'] not in TERMINAL: await self.cancel(job['id'])
    async def close(self):
        self._closing=True
        for job_id in list(self.tasks): await self.cancel(job_id)
