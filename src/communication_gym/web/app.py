"""Loopback-only HTTP transport. Domain/application services own training behavior."""
from __future__ import annotations
import asyncio
import secrets
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from communication_gym.domain.models import GymError,new_id
from communication_gym.settings import Settings,CredentialStore,load_settings,save_settings,storage_root
from communication_gym.storage.repository import Repository
from communication_gym.application.runtime import Runtime
from communication_gym.application.privacy import PrivacyService

WEB=Path(__file__).parent
MUTATIONS={'POST','PUT','PATCH','DELETE'}

def create_app(root:Path|None=None,settings:Settings|None=None,*,port:int=8787,providers:dict|None=None):
    root=Path(root or storage_root()).resolve()
    initial_settings=settings
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    app.state.bootstrap_token=secrets.token_urlsafe(32)
    app.state.cookie_token=secrets.token_urlsafe(32)
    app.state.csrf=secrets.token_urlsafe(32)
    app.state.port=port
    app.state.used_bootstrap=False
    app.state.deletion_pending=False
    @asynccontextmanager
    async def lifespan(_app):
        repo=Repository(root);repo.initialize()
        cfg=initial_settings or load_settings(root)
        credentials=CredentialStore(isolated=bool(os.environ.get('GYM_DATA_DIR')) or settings is not None)
        runtime=Runtime(repo,cfg,credentials)
        for name,value in (providers or {}).items():
            setattr(runtime,name,value);setattr(runtime.resources,name,value)
        app.state.runtime=runtime
        app.state.privacy=PrivacyService(runtime)
        app.state.privacy.drain()
        try:yield
        finally:
            await app.state.runtime.close()
            app.state.runtime.repo.close()
    app.router.lifespan_context=lifespan
    @app.exception_handler(GymError)
    async def gym_error(_request,exc):
        payload=exc.as_dict()
        if hasattr(exc,'current_revision'):payload['current_revision']=exc.current_revision
        return JSONResponse(payload,status_code=exc.status)
    @app.exception_handler(RequestValidationError)
    async def validation(_request,_exc):return JSONResponse({'error':{'code':'validation_error','message':'Check the request fields and try again.','retryable':False}},status_code=422)
    @app.middleware('http')
    async def guard(request,call_next):
        host=request.headers.get('host','')
        allowed={f'127.0.0.1:{port}',f'localhost:{port}','testserver'}
        origin=request.headers.get('origin')
        if host not in allowed or (origin and origin not in {f'http://{host}',f'https://{host}'}):
            return JSONResponse({'error':{'code':'origin_denied','message':'Use the local launch URL.','retryable':False}},status_code=403)
        public=request.url.path in ('/','/api/health','/api/bootstrap') or request.url.path.startswith('/static/')
        if not public:
            token=request.cookies.get('gym_session','')
            if not secrets.compare_digest(token,app.state.cookie_token):return JSONResponse({'error':{'code':'authentication_required','message':'Open the current local launch URL to unlock your data.','retryable':False}},status_code=401)
            if request.method in MUTATIONS and not secrets.compare_digest(request.headers.get('x-csrf-token',''),app.state.csrf):return JSONResponse({'error':{'code':'csrf_denied','message':'Reload the local application before changing data.','retryable':False}},status_code=403)
            if app.state.deletion_pending and request.url.path not in ('/api/local-data','/api/browser-session'):
                return JSONResponse({'error':{'code':'deletion_pending','message':'Deletion is incomplete. Close playback and retry Delete all Gym data, or restart to resume the purge.','retryable':True}},status_code=409)
        response=await call_next(request)
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        return response
    def rt():return app.state.runtime
    async def body(request):
        if int(request.headers.get('content-length','0') or 0)>2*1024*1024:raise GymError('request_too_large','This request exceeds the metadata limit.',413)
        try:
            value=await request.json()
            if not isinstance(value,dict):raise ValueError
            return value
        except (ValueError,TypeError):raise GymError('validation_error','Send a JSON object.',422) from None
    @app.get('/')
    async def shell():return FileResponse(WEB/'templates/index.html')
    app.mount('/static',StaticFiles(directory=WEB/'static'),name='static')
    @app.get('/api/health')
    async def health():return {'status':'ok','application':'CommunicationGym','version':'0.1.0'}
    @app.post('/api/bootstrap')
    async def bootstrap(request:Request):
        data=await body(request)
        if app.state.used_bootstrap or not secrets.compare_digest(str(data.get('token','')),app.state.bootstrap_token):raise GymError('invalid_launch_token','Use the current local launch URL or restart the app.',403)
        app.state.used_bootstrap=True
        response=JSONResponse({'csrf':app.state.csrf})
        response.set_cookie('gym_session',app.state.cookie_token,httponly=True,samesite='strict')
        return response
    @app.get('/api/browser-session')
    async def browser_session():return {'csrf':app.state.csrf}
    @app.get('/api/settings')
    async def settings_get():
        data=rt().settings.model_dump()
        data['secret_present']=bool(rt().credentials.get('coach'))
        return data
    @app.patch('/api/settings')
    async def settings_patch(request:Request):
        data=await body(request)
        key=data.pop('api_key',None)
        current=rt().settings.model_dump()
        for name,values in data.items():
            if name not in current or not isinstance(values,dict):raise GymError('validation_error','Update a named settings section.',422)
            current[name].update(values)
        if rt().jobs.tasks:raise GymError('job_running','Finish or cancel active work before changing providers.',409)
        try:cfg=Settings.model_validate(current);save_settings(root,cfg)
        except ValueError:raise GymError('invalid_settings','Check provider URL, privacy consent and supported settings values.',422) from None
        await rt().resources.close()
        rt().speech=rt().coach=rt().tts=None
        rt().resources.speech=rt().resources.coach=rt().resources.tts=None
        rt().settings=cfg
        if key is not None:
            if key:rt().credentials.set('coach',key)
            else:rt().credentials.delete('coach')
        return {'saved':True}
    @app.get('/api/capabilities')
    async def capabilities():
        result={}
        for name,getter in (('stt',rt().get_speech),('coach',rt().get_coach),('tts',rt().get_tts)):
            try:result[name]=await getter().capabilities()
            except GymError as exc:result[name]={'available':False,'unavailable_reason':exc.message,'code':exc.code}
            except Exception:result[name]={'available':False,'unavailable_reason':'Provider prerequisites or endpoint are unavailable.'}
        return result
    @app.get('/api/topics')
    async def topics(mode:str='explain',language:str='en'):
        if mode not in ('explain','compare','work_update','proposal','technical_explanation','interview') or language not in ('en','hi','hinglish'):raise GymError('validation_error','Choose a supported language and training mode.',422)
        return rt().topics(mode,language)
    @app.post('/api/topics/generate')
    async def generate_topic(request:Request):
        data=await body(request)
        async def work(ctx):
            from communication_gym.application.analysis import AnalysisEngine
            async with rt().resources.lease('coach'):
                return await AnalysisEngine(rt().get_coach()).generate_topic(data['mode'],data['language'])
        return rt().jobs.start('topic',work,operation_key=data.get('operation_key'))
    @app.get('/api/sessions')
    async def sessions_list():return rt().repo.list_sessions()
    @app.post('/api/sessions')
    async def sessions_create(request:Request):return rt().sessions.create(await body(request))
    @app.get('/api/sessions/{sid}')
    async def session_get(sid:str):return rt().repo.get_session(sid)
    @app.patch('/api/sessions/{sid}/prep')
    async def session_prep(sid:str,request:Request):
        data=await body(request);rt().check_revision(sid,data)
        return rt().sessions.prep(sid,data.get('bullets',[]),data.get('expected_revision'))
    @app.post('/api/sessions/{sid}/attempts')
    async def attempt_create(sid:str,request:Request):
        data=await body(request)
        # Service handles operation replay before stale revision rejection.
        return rt().sessions.create_attempt(sid,data.get('role','initial'),data.get('question_id'),data.get('expected_revision'),data.get('operation_key'))
    @app.post('/api/attempts/{aid}/recording')
    async def recording_create(aid:str,request:Request):return rt().recording(aid,await body(request))
    @app.put('/api/recordings/{rid}/chunks/{sequence}')
    async def recording_chunk(rid:str,sequence:int,request:Request):
        data=bytearray()
        async for block in request.stream():
            data.extend(block)
            if len(data)>4*1024*1024:raise GymError('chunk_too_large','Use audio chunks at most 4 MiB.',413)
        return await rt().chunk(rid,sequence,bytes(data))
    @app.post('/api/recordings/{rid}/finalize')
    async def recording_finalize(rid:str,request:Request):return rt().finalize(rid,await body(request))
    @app.get('/api/recordings/{rid}/audio')
    async def recording_audio(rid:str):
        record=rt().repo.get('audio_recordings',rid)
        if record['status']!='ready':raise GymError('audio_unavailable','Audio was not finalized or has been deleted.',404)
        path=rt().repo.path(record.get('prepared_path') or record['relative_path'])
        if not path.is_file():raise GymError('audio_unavailable','The local audio file is unavailable.',404)
        return FileResponse(path,media_type='audio/wav' if path.suffix=='.wav' else record['media_type'])
    @app.post('/api/attempts/{aid}/transcribe')
    async def transcribe(aid:str,request:Request):return rt().transcribe(aid,await body(request))
    @app.get('/api/jobs/{jid}')
    async def job_get(jid:str):return rt().repo.get('jobs',jid)
    @app.post('/api/jobs/{jid}/cancel')
    async def job_cancel(jid:str):return await rt().jobs.cancel(jid)
    @app.post('/api/attempts/{aid}/discard')
    async def discard_attempt(aid:str,request:Request):
        data=await body(request);att=rt().repo.get('attempts',aid);rt().check_revision(att['session_id'],data)
        await rt().jobs.cancel_session(att['session_id'])
        if att['reviewed']:raise GymError('accepted_attempt','Accepted attempts remain historical. Correct their transcript instead.',409)
        return rt().repo.put('attempts',{**att,'status':'cancelled','generation':att['generation']+1})
    @app.post('/api/attempts/{aid}/transcript-revisions')
    async def correction(aid:str,request:Request):
        data=await body(request);att=rt().repo.get('attempts',aid);rt().check_revision(att['session_id'],data)
        await rt().jobs.cancel_session(att['session_id'])
        return rt().sessions.correct_transcript(aid,data['segments'])
    @app.post('/api/attempts/{aid}/accept-transcript')
    async def accept(aid:str,request:Request):
        data=await body(request)
        return rt().sessions.accept_transcript(aid,data['transcript_id'],data.get('expected_revision'))
    @app.post('/api/attempts/{aid}/analyze')
    async def analyze(aid:str,request:Request):return rt().analyze(aid,await body(request))
    @app.post('/api/sessions/{sid}/questions')
    async def question(sid:str,request:Request):return rt().question(sid,await body(request))
    @app.post('/api/questions/{qid}/evaluate')
    async def evaluate(qid:str,request:Request):return rt().evaluate_answer(qid,await body(request))
    @app.post('/api/questions/{qid}/skip')
    async def skip(qid:str,request:Request):
        data=await body(request)
        return rt().sessions.skip_question(qid,data.get('expected_revision'))
    @app.post('/api/sessions/{sid}/finish-questions')
    async def finish_questions(sid:str,request:Request):
        data=await body(request)
        return rt().sessions.finish_questions(sid,data.get('expected_revision'))
    @app.post('/api/sessions/{sid}/compare')
    async def compare(sid:str,request:Request):return rt().compare(sid,await body(request))
    @app.post('/api/sessions/{sid}/complete')
    async def complete(sid:str,request:Request):
        data=await body(request)
        return rt().sessions.complete(sid,data.get('expected_revision'))
    @app.post('/api/sessions/{sid}/model-answer')
    async def model_answer(sid:str,request:Request):return rt().model_answer(sid,await body(request))
    @app.get('/api/progress')
    async def progress():return rt().profile()
    @app.get('/api/profile/goal')
    async def goal_get():return rt().repo.find('training_goals')
    @app.patch('/api/profile/goal')
    async def goal_patch(request:Request):
        data=await body(request)
        for row in rt().repo.find('training_goals'):rt().repo.remove('training_goals',row['id'])
        if not data.get('focus'):return {'cleared':True}
        return rt().repo.put('training_goals',{'id':new_id(),'focus':str(data['focus'])[:200],'active':True})
    @app.post('/api/sessions/{sid}/export')
    async def export(sid:str,request:Request):
        data=await body(request);rt().check_revision(sid,data)
        async def work(ctx):return ctx.commit(lambda:app.state.privacy.export(sid,bool(data.get('include_audio',False))))
        return rt().jobs.start('export',work,session_id=sid,operation_key=data.get('operation_key'))
    @app.get('/api/exports/{eid}')
    async def download_export(eid:str):return FileResponse(rt().repo.path(f'exports/{eid}/session.zip'),media_type='application/zip',filename='communication-gym-session.zip')
    @app.delete('/api/sessions/{sid}')
    async def delete_session(sid:str,request:Request):
        rt().check_revision(sid,await body(request));return await app.state.privacy.delete_session(sid)
    @app.delete('/api/sessions/{sid}/audio')
    async def delete_audio(sid:str,request:Request):
        rt().check_revision(sid,await body(request));return await app.state.privacy.delete_audio(sid)
    @app.delete('/api/local-data')
    async def delete_all(request:Request):
        data=await body(request)
        if data.get('confirmation')!='DELETE ALL':
            raise GymError('confirmation_required','Type DELETE ALL to remove all Gym-owned local data.',422)
        app.state.deletion_pending=True
        result=await app.state.privacy.delete_all(data.get('confirmation'))
        repo=Repository(root);repo.initialize()
        runtime=Runtime(repo,Settings(),CredentialStore(isolated=bool(os.environ.get('GYM_DATA_DIR')) or settings is not None))
        app.state.runtime=runtime;app.state.privacy=PrivacyService(runtime)
        app.state.deletion_pending=False
        app.state.bootstrap_token=secrets.token_urlsafe(32);app.state.cookie_token=secrets.token_urlsafe(32);app.state.csrf=secrets.token_urlsafe(32);app.state.used_bootstrap=False
        response=JSONResponse({**result,'launch_token':app.state.bootstrap_token});response.delete_cookie('gym_session');return response
    @app.get('/api/tts/voices')
    async def voices():return await rt().get_tts().list_voices()
    @app.post('/api/tts/confirm-idle')
    async def idle():
        result=rt().get_tts().confirm_idle()
        if asyncio.iscoroutine(result):await result
        return {'confirmed':True}
    @app.post('/api/sessions/{sid}/tts')
    async def tts(sid:str,request:Request):
        data=await body(request);session=rt().check_revision(sid,data)
        examples=[a for a in session['analyses'] if a.get('subsystem')=='model_answer' and a.get('status')=='succeeded']
        if not examples:raise GymError('example_required','Request an improved example after retry before playback.',409)
        example=examples[-1]['payload'];text=data.get('text') or example.get('text')
        if data.get('text') and data['text'] not in example.get('text',''):raise GymError('invalid_excerpt','Select an exact excerpt from the improved example.',422)
        async def work(ctx):
            async with rt().resources.lease('tts'):
                artifact=await rt().get_tts().synthesize({'text':text,'language':example.get('language',session['language']),'profile_id':rt().settings.tts.profile_id,'session_id':sid})
            return ctx.commit(lambda:rt().repo.put('audio_recordings',{**artifact,'id':new_id(),'session_id':sid,'media_type':'audio/wav','status':'ready','provenance':'tts'}))
        return rt().jobs.start('tts',work,session_id=sid,operation_key=data.get('operation_key'))
    return app
