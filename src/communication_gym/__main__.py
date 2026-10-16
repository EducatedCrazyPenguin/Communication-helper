from __future__ import annotations
import argparse
import asyncio
import json
import socket
import threading
import webbrowser
from communication_gym.settings import load_settings,storage_root


def main():
    parser=argparse.ArgumentParser(prog='communication_gym')
    commands=parser.add_subparsers(dest='command',required=True)
    serve=commands.add_parser('serve')
    serve.add_argument('--host',default='127.0.0.1',choices=['127.0.0.1'])
    serve.add_argument('--port',type=int,default=8787)
    serve.add_argument('--no-browser',action='store_true')
    doctor=commands.add_parser('doctor');doctor.add_argument('--no-load',action='store_true',default=True)
    args=parser.parse_args()
    root=storage_root()
    if args.command=='doctor':
        from communication_gym.providers.stt import SpeechProvider
        from importlib.metadata import version
        config=load_settings(root)
        async def inspect():
            provider=SpeechProvider(config.stt.model_dump(),root)
            try:return await provider.capabilities()
            finally:await provider.close()
        report={'application':'CommunicationGym','root':str(root),'python':__import__('sys').version.split()[0],'dependencies':{n:version(n) for n in ('fastapi','pydantic','sqlalchemy','alembic','httpx')},'stt':asyncio.run(inspect()),'coach':{'provider':config.coach.provider,'model':config.coach.model,'inference_tested':False},'voice_studio_enabled':config.tts.enabled}
        print(json.dumps(report,ensure_ascii=True,indent=2));return
    import uvicorn
    from communication_gym.web.app import create_app
    try:
        probe=socket.socket();probe.bind((args.host,args.port));probe.close()
    except OSError:
        parser.error(f'Port {args.port} is in use. Choose --port; no other process was stopped.')
    app=create_app(root,port=args.port)
    url=f'http://127.0.0.1:{args.port}/#token={app.state.bootstrap_token}'
    print(f'Communication Gym local launch URL (do not share): {url}',flush=True)
    if not args.no_browser:
        timer=threading.Timer(1.0,lambda:webbrowser.open(url));timer.daemon=True;timer.start()
    uvicorn.run(app,host=args.host,port=args.port,access_log=False,log_level='warning',timeout_graceful_shutdown=8)

if __name__=='__main__':main()
