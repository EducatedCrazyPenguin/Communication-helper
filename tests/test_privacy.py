from pathlib import Path
import json
import pytest
from fastapi.testclient import TestClient
from communication_gym.web.app import create_app
from communication_gym.settings import Settings


def client_for(root):
    app=create_app(root,Settings())
    client=TestClient(app)
    return app,client

def unlock(app,client):
    result=client.post('/api/bootstrap',json={'token':app.state.bootstrap_token})
    assert result.status_code==200
    return {'x-csrf-token':result.json()['csrf']}

def create(client,headers):
    response=client.post('/api/sessions',headers=headers,json={'language':'hinglish','mode':'compare','exercise':{'source':'custom','title':'Battery A/B','prompt':'दो batteries compare करो','audience':'A new listener','mode':'compare','language':'hinglish'}})
    assert response.status_code==200,response.text
    return response.json()

def test_auth_origin_csrf_and_immutable_session_resume(tmp_path):
    app,client=client_for(tmp_path/'root')
    with client:
        assert client.get('/api/sessions').status_code==401
        headers=unlock(app,client)
        assert client.post('/api/sessions',json={}).status_code==403
        assert client.get('/api/sessions',headers={'Origin':'https://evil.example'}).status_code==403
        session=create(client,headers)
        rejected=client.patch(f'/api/sessions/{session["id"]}/prep',headers=headers,json={'expected_revision':session['revision'],'bullets':['one','two','three','four','five','six']})
        assert rejected.status_code==422
        saved=client.patch(f'/api/sessions/{session["id"]}/prep',headers=headers,json={'expected_revision':session['revision'],'bullets':['A main','B backup']})
        assert saved.status_code==200
        assert saved.json()['prep_bullets']==['A main','B backup']
    app2,client2=client_for(tmp_path/'root')
    with client2:
        unlock(app2,client2)
        restored=client2.get(f'/api/sessions/{session["id"]}').json()
        assert restored['prep_bullets']==['A main','B backup']
        assert restored['exercise']['prompt']=='दो batteries compare करो'
        assert app2.state.runtime.speech is None and app2.state.runtime.coach is None

def test_chunk_idempotence_conflict_and_owned_deletion(tmp_path):
    root=tmp_path/'root';outside=tmp_path/'outside.txt';outside.write_text('untouched')
    app,client=client_for(root)
    with client:
        headers=unlock(app,client);session=create(client,headers)
        saved=client.patch(f'/api/sessions/{session["id"]}/prep',headers=headers,json={'expected_revision':session['revision'],'bullets':[]}).json()
        attempt=client.post(f'/api/sessions/{session["id"]}/attempts',headers=headers,json={'expected_revision':saved['revision'],'role':'initial','operation_key':'first'}).json()
        current=client.get(f'/api/sessions/{session["id"]}').json()
        recording=client.post(f'/api/attempts/{attempt["id"]}/recording',headers=headers,json={'expected_revision':current['revision'],'media_type':'audio/wav'}).json()
        path=f'/api/recordings/{recording["id"]}/chunks/0'
        assert client.put(path,headers=headers,content=b'original audio').status_code==200
        assert client.put(path,headers=headers,content=b'original audio').json()['size']==14
        assert client.put(path,headers=headers,content=b'changed audio').status_code==409
        assert client.put(path.replace('/0','/2'),headers=headers,content=b'gap').status_code==409
        current=client.get(f'/api/sessions/{session["id"]}').json()
        result=client.delete(f'/api/sessions/{session["id"]}',headers=headers,json={'expected_revision':current['revision']}) if False else client.request('DELETE',f'/api/sessions/{session["id"]}',headers=headers,json={'expected_revision':current['revision']})
        assert result.status_code==200,result.text
        assert result.json()['deleted']
        assert client.get(f'/api/sessions/{session["id"]}').status_code==404
        assert outside.read_text()=='untouched'

def test_export_and_complete_delete_do_not_touch_other_roots(tmp_path):
    root=tmp_path/'root';sentinel=tmp_path/'models';sentinel.mkdir();(sentinel/'model.bin').write_bytes(b'shared')
    app,client=client_for(root)
    with client:
        headers=unlock(app,client);session=create(client,headers)
        result=app.state.privacy.export(session['id'])
        import zipfile
        with zipfile.ZipFile(root/result['relative_path']) as archive:
            assert set(archive.namelist())=={'manifest.json','session.json','summary.md'}
            assert 'api_key' not in archive.read('session.json').decode()
        denied=client.request('DELETE','/api/local-data',headers=headers,json={'confirmation':'no'})
        assert denied.status_code==422
        deleted=client.request('DELETE','/api/local-data',headers=headers,json={'confirmation':'DELETE ALL'})
        assert deleted.status_code==200,deleted.text
        token=deleted.json()['launch_token']
        assert client.get('/api/sessions').status_code==401
        bootstrap=client.post('/api/bootstrap',json={'token':token})
        assert bootstrap.status_code==200
        assert client.get('/api/sessions').json()==[]
        assert (sentinel/'model.bin').read_bytes()==b'shared'
