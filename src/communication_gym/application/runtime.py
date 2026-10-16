"""Application orchestration: model outputs are committed only to their originating revision."""
from __future__ import annotations
import asyncio
import copy
import hashlib
import inspect
import json
from pathlib import Path
from importlib.resources import files
from communication_gym.domain.models import GymError,new_id,utc_now
from communication_gym.application.jobs import JobManager
from communication_gym.application.resources import ResourceCoordinator
from communication_gym.application.sessions import SessionService
from communication_gym.settings import is_local_url

class Runtime:
    def __init__(self,repo,settings,credentials):
        self.repo,self.settings,self.credentials=repo,settings,credentials
        self.sessions=SessionService(repo)
        self.jobs=JobManager(repo)
        self.speech=self.coach=self.tts=None
        self.resources=ResourceCoordinator()
        self.record_locks={}
    def get_speech(self):
        if self.speech is None:
            from communication_gym.providers.stt import SpeechProvider
            self.speech=SpeechProvider(self.settings.stt.model_dump(),self.repo.root)
            self.resources.speech=self.speech
        return self.speech
    def get_coach(self,session=None):
        config=self.settings.coach.model_dump()
        if not is_local_url(config['base_url']):
            consent=(session or {}).get('consent',{})
            if not (self.settings.privacy.allow_cloud and consent.get('approved') and consent.get('endpoint')==config['base_url'] and consent.get('model')==config['model']):
                raise GymError('cloud_consent_required','Confirm transcript sharing for this endpoint and model in this session.',403)
            config.update(allow_cloud=True,session_consent=consent)
        if self.coach is None:
            if config['provider']=='ollama':
                from communication_gym.providers.ollama import OllamaProvider
                self.coach=OllamaProvider(config)
            else:
                from communication_gym.providers.openai_compatible import OpenAICompatibleProvider
                self.coach=OpenAICompatibleProvider(config,self.credentials.get('coach'))
            self.resources.coach=self.coach
        # Remote consent is session-specific; never leave a prior session's consent cached.
        if hasattr(self.coach,'config') and isinstance(self.coach.config,dict): self.coach.config.update(config)
        return self.coach
    def get_tts(self):
        if self.tts is None:
            from communication_gym.providers.voicestudio import VoiceStudioProvider
            self.tts=VoiceStudioProvider(self.settings.tts.model_dump(),self.repo.root)
            self.resources.tts=self.tts
        return self.tts
    def topics(self,mode,language):
        data=json.loads((files('communication_gym')/'data/topics.v1.json').read_text(encoding='utf-8'))
        return [{'title':x['title'],'prompt':x['prompts'][language],'audience':x['audience'],'mode':mode,'language':language,'source':'curated'} for x in data['exercises'] if x['mode']==mode]
    def check_revision(self,sid,body):
        session=self.repo.get_session(sid)
        expected=body.get('expected_revision')
        if expected is None: raise GymError('revision_required','Reload the session and send its expected revision.',422)
        SessionService._revision(session,expected)
        return session
    def recording(self,aid,body):
        attempt=self.repo.get('attempts',aid)
        self.check_revision(attempt['session_id'],body)
        if attempt['status'] not in ('created','recording'):
            raise GymError('invalid_transition','Start a new recording attempt or resume the saved audio.',409)
        if attempt.get('recording_id'): return self.repo.get('audio_recordings',attempt['recording_id'])
        mime=body.get('media_type','audio/webm')
        if mime not in ('audio/webm','audio/webm;codecs=opus','audio/mp4','audio/ogg','audio/ogg;codecs=opus','audio/wav','audio/x-wav','audio/mpeg','video/mp4'):
            raise GymError('unsupported_audio','Choose a supported microphone recording or WAV/MP3/MP4/Ogg file.',422)
        rid=new_id()
        rel=f'data/sessions/{attempt["session_id"]}/recordings/{rid}/source.bin'
        path=self.repo.path(rel);path.parent.mkdir(parents=True,exist_ok=True)
        with self.repo.transaction():
            record=self.repo.put('audio_recordings',{'id':rid,'session_id':attempt['session_id'],'attempt_id':aid,'relative_path':rel,'media_type':mime,'status':'partial','provenance':body.get('provenance','microphone'),'chunks':{},'size':0,'next_sequence':0})
            self.repo.put('attempts',{**attempt,'recording_id':rid,'status':'recording'})
        return record
    async def chunk(self,rid,sequence,data):
        lock=self.record_locks.setdefault(rid,asyncio.Lock())
        async with lock:
            record=self.repo.get('audio_recordings',rid)
            if record['status']!='partial': raise GymError('recording_sealed','This recording is already finalized.',409)
            if len(data)>4*1024*1024: raise GymError('chunk_too_large','Upload audio in chunks no larger than 4 MiB.',413)
            digest=hashlib.sha256(data).hexdigest()
            known=record['chunks'].get(str(sequence))
            if known:
                if known['sha256']!=digest: raise GymError('chunk_conflict','This audio sequence already contains different bytes.',409)
                return record
            if sequence!=record['next_sequence']: raise GymError('sequence_gap','Upload the next missing audio chunk first.',409)
            if record['size']+len(data)>256*1024*1024: raise GymError('recording_too_large','Recording exceeds the 256 MiB limit.',413)
            path=self.repo.path(record['relative_path']).parent/f'{sequence}.chunk'
            temp=path.with_suffix('.tmp');temp.write_bytes(data);temp.replace(path)
            record['chunks'][str(sequence)]={'sha256':digest,'size':len(data)}
            return self.repo.put('audio_recordings',{**record,'size':record['size']+len(data),'next_sequence':sequence+1})
    def finalize(self,rid,body):
        record=self.repo.get('audio_recordings',rid)
        attempt=self.repo.get('attempts',record['attempt_id'])
        self.check_revision(attempt['session_id'],body)
        if record['size']==0: raise GymError('empty_audio','No audio was saved. Record or upload again.',422)
        async def work(ctx):
            record=self.repo.get('audio_recordings',rid)
            source=self.repo.path(record['relative_path'])
            partial=source.with_suffix('.partial')
            digest=hashlib.sha256()
            with partial.open('wb') as out:
                for seq in range(record['next_sequence']):
                    data=(source.parent/f'{seq}.chunk').read_bytes()
                    if hashlib.sha256(data).hexdigest()!=record['chunks'][str(seq)]['sha256']: raise GymError('audio_corrupt','A saved audio chunk failed verification.',409)
                    out.write(data);digest.update(data)
            partial.replace(source)
            prepared=source.parent/'prepared.wav'
            async with self.resources.lease('stt'):
                decoded=await self.get_speech().prepare_audio(source,prepared)
            duration=decoded.get('duration_seconds',decoded.get('duration',0))
            if not 0<duration<=902: raise GymError('invalid_audio_duration','Audio must contain a nonempty recording of at most 15 minutes.',422)
            def commit():
                latest=self.repo.get('audio_recordings',rid)
                result=self.repo.put('audio_recordings',{**latest,'status':'ready','sha256':digest.hexdigest(),'duration_seconds':duration,'sample_rate':16000,'prepared_path':prepared.relative_to(self.repo.root).as_posix(),'diagnostics':decoded.get('diagnostics',{})})
                att=self.repo.get('attempts',attempt['id'])
                self.repo.put('attempts',{**att,'captured_duration':duration})
                return result
            return ctx.commit(commit)
        return self.jobs.start('prepare_audio',work,session_id=attempt['session_id'],attempt_id=attempt['id'],operation_key=body.get('operation_key'))
    def transcribe(self,aid,body):
        attempt=self.repo.get('attempts',aid);session=self.check_revision(attempt['session_id'],body)
        if attempt.get('active_transcript_id'): raise GymError('transcript_exists','Review the existing transcript or create a new recording.',409)
        record=self.repo.get('audio_recordings',attempt['recording_id'])
        if record['status']!='ready': raise GymError('audio_not_ready','Finish saving and decoding the recording first.',409)
        async def work(ctx):
            async with self.resources.lease('stt'):
                stage='cross_exam' if attempt['role']=='cross_exam' else f'{attempt["role"]}_transcribing'
                ctx.commit(lambda:self.repo.advance_session(session['id'],stage,None))
                result=await self.get_speech().transcribe(self.repo.path(record['prepared_path']),session['language'],self.repo.path(f'data/sessions/{session["id"]}/checkpoints/{ctx.job["id"]}'),ctx.progress)
            from communication_gym.providers.stt import build_transcript
            bundle=build_transcript(result,aid)
            def commit():
                tid=bundle.get('id') or new_id()
                original_path=self.repo.path(f'data/sessions/{session["id"]}/transcripts/{tid}.json');original_path.parent.mkdir(parents=True,exist_ok=True)
                original_path.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
                bundle.update(id=tid,session_id=session['id'],attempt_id=aid,kind='asr_original',original_json_path=original_path.relative_to(self.repo.root).as_posix())
                transcript=self.repo.put('transcripts',bundle)
                current=self.repo.get('attempts',aid)
                self.repo.put('attempts',{**current,'status':'review','active_transcript_id':transcript['id']})
                self.repo.advance_session(session['id'],'cross_exam' if attempt['role']=='cross_exam' else f'{attempt["role"]}_review',None)
                return {'transcript_id':transcript['id']}
            return ctx.commit(commit)
        return self.jobs.start('transcribe',work,session_id=session['id'],attempt_id=aid,operation_key=body.get('operation_key'))
    def analyze(self,aid,body):
        attempt=self.repo.get('attempts',aid);session=self.check_revision(attempt['session_id'],body)
        if not attempt.get('reviewed'): raise GymError('review_required','Accept or correct the verbatim transcript before analysis.',409)
        if attempt['role']=='cross_exam': raise GymError('question_evaluation_required','Evaluate this answer against its question.',409)
        transcript=self.repo.get_transcript(attempt['active_transcript_id'])
        async def work(ctx):
            from communication_gym.application.analysis import AnalysisEngine
            engine=AnalysisEngine(self.get_coach(session))
            try:
                async with self.resources.lease('coach'):
                    result=await engine.analyze(transcript,session['exercise'],session['language'],ctx.progress)
            except BaseException:
                # Retain successfully validated subsystem evidence, never label a failed listener complete.
                for item in getattr(engine,'window_artifacts',[]):
                    if isinstance(item,dict):
                        try:ctx.commit(lambda item=item:self.repo.put('analyses',{'id':new_id(),'session_id':session['id'],'attempt_id':aid,'transcript_id':transcript['id'],'subsystem':'partial_window','status':'incomplete','payload':item}))
                        except (GymError,asyncio.CancelledError):break
                raise
            def commit():
                result.update(id=new_id(),session_id=session['id'],attempt_id=aid,transcript_id=transcript['id'],subsystem='complete',status='succeeded')
                saved=self.repo.put('analyses',result)
                current=self.repo.get('attempts',aid)
                self.repo.put('attempts',{**current,'status':'analyzed'})
                self.repo.advance_session(session['id'],'breakdown' if attempt['role']=='initial' else 'retry_analysis',None)
                return {'analysis_id':saved['id']}
            return ctx.commit(commit)
        return self.jobs.start('analysis',work,session_id=session['id'],attempt_id=aid,operation_key=body.get('operation_key'))
    def question(self,sid,body):
        session=self.check_revision(sid,body)
        if session['stage'] not in ('breakdown','cross_exam'): raise GymError('invalid_transition','Finish the first analysis before follow-up questions.',409)
        opened=[q for q in session['questions'] if q['status']=='open']
        if opened:return opened[0]
        if len(session['questions'])>=10: raise GymError('question_limit','Ten questions are enough for this cycle. Make your retry.',409)
        initial=next(a for a in session['attempts'] if a['role']=='initial' and a.get('analysis'))
        async def work(ctx):
            from communication_gym.application.analysis import AnalysisEngine
            async with self.resources.lease('coach'):
                result=await AnalysisEngine(self.get_coach(session)).question(session['initial_analysis'],initial['transcript'],session['exercise'],session['questions'])
            def commit():
                row={**result,'id':new_id(),'session_id':sid,'ordinal':len(session['questions'])+1,'status':'open'}
                saved=self.repo.put('cross_exam_questions',row)
                self.repo.advance_session(sid,'cross_exam',None)
                return saved
            return ctx.commit(commit)
        return self.jobs.start('question',work,session_id=sid,operation_key=body.get('operation_key'))
    def evaluate_answer(self,qid,body):
        question=self.repo.get('cross_exam_questions',qid);session=self.check_revision(question['session_id'],body)
        answers=[a for a in session['attempts'] if a.get('question_id')==qid and a.get('reviewed')]
        if question['status']!='open' or not answers:raise GymError('answer_required','Record and accept an answer to the open question first.',409)
        answer=answers[-1];original=next(a['transcript'] for a in session['attempts'] if a['role']=='initial' and a.get('transcript'))
        async def work(ctx):
            from communication_gym.application.analysis import AnalysisEngine
            async with self.resources.lease('coach'):
                result=await AnalysisEngine(self.get_coach(session)).evaluate_answer(question,original,answer['transcript'])
            def commit():
                saved=self.repo.put('cross_exam_questions',{**question,'status':'answered','resolution':result,'answer_attempt_id':answer['id']})
                self.repo.update_session(session['id'],{},None)
                return saved
            return ctx.commit(commit)
        return self.jobs.start('answer_evaluation',work,session_id=session['id'],attempt_id=answer['id'],operation_key=body.get('operation_key'))
    def compare(self,sid,body):
        session=self.check_revision(sid,body)
        if not session.get('initial_analysis') or not session.get('retry_analysis'):raise GymError('retry_required','Analyze both attempts before comparison.',409)
        initial=next(a for a in session['attempts'] if a['role']=='initial' and a.get('analysis'))
        retry=next(a for a in session['attempts'] if a['role']=='retry' and a.get('analysis'))
        async def work(ctx):
            from communication_gym.application.analysis import AnalysisEngine
            async with self.resources.lease('coach'):
                result=await AnalysisEngine(self.get_coach(session)).compare(session['initial_analysis'],session['retry_analysis'],initial['transcript'],retry['transcript'])
            def commit():
                saved=self.repo.put('retry_comparisons',{**result,'id':new_id(),'session_id':sid,'status':'succeeded','initial_analysis_id':session['initial_analysis']['id'],'retry_analysis_id':session['retry_analysis']['id']})
                self.repo.update_session(sid,{'stage':'comparison','active_comparison_id':saved['id']},None)
                return saved
            return ctx.commit(commit)
        return self.jobs.start('comparison',work,session_id=sid,operation_key=body.get('operation_key'))
    def model_answer(self,sid,body):
        session=self.check_revision(sid,body)
        if not session.get('comparison') or session['stage'] not in ('comparison','complete'):raise GymError('retry_required','Try the improvement yourself and compare it before requesting an example.',409)
        transcripts=[a['transcript'] for a in session['attempts'] if a.get('transcript') and a['role'] in ('initial','retry')]
        language=body.get('language',session['language'])
        if language not in ('en','hi','hinglish'):raise GymError('invalid_language','Choose English, Hindi or Hinglish.',422)
        async def work(ctx):
            from communication_gym.application.analysis import AnalysisEngine
            async with self.resources.lease('coach'):
                result=await AnalysisEngine(self.get_coach(session)).model_answer(transcripts,language)
            return ctx.commit(lambda:self.repo.put('analyses',{'id':new_id(),'session_id':sid,'attempt_id':transcripts[-1]['attempt_id'],'transcript_id':transcripts[-1]['id'],'subsystem':'model_answer','status':'succeeded','payload':result}))
        return self.jobs.start('model_answer',work,session_id=sid,operation_key=body.get('operation_key'))
    def profile(self):
        from communication_gym.application.profile import compute_profile
        sessions=[self.repo.get_session(s['id']) for s in self.repo.list_sessions() if s['stage']=='complete']
        return compute_profile(sessions)
    async def close(self):
        await self.jobs.close()
        await self.resources.close()
