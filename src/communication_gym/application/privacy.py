"""Deletion and exports operate only on marked Gym storage, never shared source/model roots."""
from __future__ import annotations
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PureWindowsPath
from communication_gym.domain.models import GymError,new_id,require_id,utc_now


_PRIVATE_KEYS = {'path','source_file','source_path','python_path','repo_path','api_key','authorization','credential','credentials','password','secret','token','access_token','api_token','original_json_path','prepared_path','relative_path','original_path','checkpoint_path','artifact_path','export_path'}


def _sanitized(value):
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            normalized=key.lower().replace('-','_')
            if normalized in _PRIVATE_KEYS or normalized.endswith(('_path','_directory','_dir')):
                continue
            if normalized=='source' and isinstance(item,str) and (Path(item).is_absolute() or PureWindowsPath(item).is_absolute()):
                continue
            result[key]=_sanitized(item)
        return result
    if isinstance(value,list):return [_sanitized(v) for v in value]
    return value


class PrivacyService:
    def __init__(self,runtime):
        self.runtime=runtime
        self.repo=runtime.repo

    def _purge(self,path:Path):
        resolved=path.resolve()
        if resolved==self.repo.root or not resolved.is_relative_to(self.repo.root):
            raise GymError('unsafe_path','Refusing to delete outside owned storage.',422)
        self.repo.assert_unlinked_tree(path)
        if path.is_dir():shutil.rmtree(path)
        else:path.unlink(missing_ok=True)

    def _schedule_backups(self):
        backups=self.repo.root/'backups'
        if backups.exists():
            for path in backups.iterdir():
                self.repo.put('deletion_tasks',{'id':new_id(),'status':'pending','relative_path':path.relative_to(self.repo.root).as_posix()})

    def drain(self):
        pending=[]
        for task in self.repo.find('deletion_tasks'):
            if task.get('status')=='succeeded':continue
            try:
                self._purge(self.repo.path(task['relative_path']))
                self.repo.put('deletion_tasks',{**task,'status':'succeeded','safe_error':None,'finished_at':utc_now(),'attempts':task.get('attempts',0)+1})
            except (OSError,GymError):
                self.repo.put('deletion_tasks',{**task,'status':'pending','safe_error':'file_locked_or_unsafe','attempts':task.get('attempts',0)+1})
                pending.append(task['id'])
        return pending

    async def delete_session(self,sid):
        await self.runtime.jobs.cancel_session(sid)
        self.repo.delete_session(sid)
        pending=self.drain()
        return {'deleted':not pending,'pending_deletions':pending,'external_exports_remain':True}

    async def delete_audio(self,sid):
        await self.runtime.jobs.cancel_session(sid)
        session=self.repo.get_session(sid)
        with self.repo.transaction():
            self.repo.update_session(sid,{'generation':session.get('generation',0)+1},None)
            for rec in self.repo.find('audio_recordings',session_id=sid):
                for key in ('relative_path','prepared_path','original_path'):
                    if rec.get(key):
                        self.repo.put('deletion_tasks',{'id':new_id(),'session_id':sid,'status':'pending','relative_path':rec[key]})
                self.repo.put('audio_recordings',{**rec,'status':'deleted','chunks':{}})
            for directory in ('recordings','checkpoints','tts'):
                self.repo.put('deletion_tasks',{'id':new_id(),'session_id':sid,'status':'pending','relative_path':f'data/sessions/{sid}/{directory}'})
            # Internal ZIP copies may contain original audio; external copies cannot be recalled.
            for relative in self.repo.session_export_paths(sid):
                self.repo.put('deletion_tasks',{'id':new_id(),'session_id':sid,'status':'pending','relative_path':relative})
            self._schedule_backups()
        pending=self.drain()
        return {'audio_deleted':not pending,'metrics_retained':True,'pending_deletions':pending,'external_exports_remain':True}

    def export(self,sid,include_audio=False):
        # Capture every revision and analysis in one database snapshot. No stored evidence is rewritten.
        with self.repo.transaction():
            session=self.repo.get_session(sid)
            recordings=self.repo.find('audio_recordings',session_id=sid)
            transcripts=self.repo.find('transcripts',session_id=sid)
        eid=new_id();path=self.repo.path(f'exports/{eid}/session.zip');path.parent.mkdir(parents=True,exist_ok=True)
        manifest={'schema_version':1,'export_id':eid,'session_id':sid,'created_at':utc_now(),'audio_included':include_audio,'evidence_redactions':['local_paths','credentials'],'files':[],'warning':'External copies are outside future Gym deletion control.'}
        # Persist ownership before the first private byte, including interrupted exports.
        sidecar=path.parent/'manifest.json'
        sidecar.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        temp=path.with_suffix('.partial')
        try:
            with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                def add_json(name,value):
                    data=json.dumps(_sanitized(value),ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
                    archive.writestr(name,data)
                    manifest['files'].append({'path':name,'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data)})
                add_json('session.json',session)
                title=(session.get('exercise') or {}).get('title','Practice session')
                lines=[f'# {title}',f'Language: {session["language"]}',f'Mode: {session["mode"]}']
                for attempt in session['attempts']:
                    transcript=attempt.get('transcript')
                    if transcript:lines.extend([f'## {attempt["role"]}',transcript['text']])
                archive.writestr('summary.md','\n\n'.join(lines))
                for transcript in transcripts:
                    if transcript.get('original_json_path'):
                        tid=require_id(transcript['id'])
                        source=self.repo.path(transcript['original_json_path'])
                        add_json(f'evidence/{tid}/asr-original.json',json.loads(source.read_text(encoding='utf-8')))
                if include_audio:
                    for rec in recordings:
                        if rec.get('status')!='ready':continue
                        rid=require_id(rec['id'])
                        seen=set()
                        for key,label in (('relative_path','original'),('prepared_path','prepared')):
                            relative=rec.get(key)
                            if not relative or relative in seen:continue
                            seen.add(relative)
                            source=self.repo.path(relative)
                            if not source.is_file():
                                raise GymError('export_audio_missing','A selected audio artifact is unavailable. Export without audio or restore the owned file.',409)
                            name=f'audio/{rid}/{label}{source.suffix}'
                            archive.write(source,name)
                            manifest['files'].append({'path':name,'recording_id':rid,'kind':rec.get('provenance'),'size_bytes':source.stat().st_size})
                archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
            temp.replace(path)
            sidecar.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        return {'export_id':eid,'relative_path':path.relative_to(self.repo.root).as_posix(),'warning':manifest['warning']}

    async def delete_all(self,confirmation):
        if confirmation!='DELETE ALL':raise GymError('confirmation_required','Type DELETE ALL to remove all Gym-owned local data.',422)
        marker=self.repo.root/'.communication-gym-owned'
        if marker.is_symlink() or not marker.is_file() or marker.read_text(encoding='utf-8')!='CommunicationGym:1\n':
            raise GymError('unsafe_data_root','The data root ownership marker is invalid.',409)
        self.repo.assert_unlinked_tree(self.repo.root)
        await self.runtime.close()
        # Credential deletion precedes the durable purge intent, so crash recovery never
        # silently leaves a Gym credential behind after claiming all local data is gone.
        self.runtime.credentials.delete('coach')
        self.repo.close(release_lock=False)
        self.repo.purge_all_files()
        self.repo.close()
        return {'deleted':True,'external_exports_remain':True,'restart_required':True}
