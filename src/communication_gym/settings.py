"""Machine-local configuration; credentials never enter settings JSON."""
from __future__ import annotations
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from pydantic import BaseModel, ConfigDict, Field
from platformdirs import user_data_path

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class ServerSettings(StrictModel):
    host: str = '127.0.0.1'
    port: int = Field(8787, ge=1024, le=65535)
class STTSettings(StrictModel):
    repo_path: str = str(Path.home()/'Documents/ChatGPT/voice to text')
    python_path: str = str(Path.home()/'Documents/ChatGPT/voice to text/.venv/Scripts/python.exe')
    device: str = Field('auto', pattern='^(auto|cuda|cpu)$')
    candidate: str = Field('qwen3', pattern='^qwen3$')
    alignment: str = Field('available', pattern='^(available|off)$')
class CoachSettings(StrictModel):
    provider: str = Field('ollama', pattern='^(ollama|openai_compatible)$')
    model: str = 'qwen3.5:9b-q8_0'
    managed: bool = True
    base_url: str = 'http://127.0.0.1:11435'
    context_tokens: int = Field(16384, ge=8192, le=65536)
class TTSSettings(StrictModel):
    enabled: bool = False
    base_url: str = 'http://127.0.0.1:3900'
    profile_id: str | None = None
    authorized_profiles: dict[str,str] = Field(default_factory=dict)
class PrivacySettings(StrictModel):
    allow_cloud: bool = False
class LoggingSettings(StrictModel):
    include_content: bool = False
class Settings(StrictModel):
    schema_version: int = Field(1, ge=1, le=1)
    server: ServerSettings = Field(default_factory=ServerSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    coach: CoachSettings = Field(default_factory=CoachSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def storage_root() -> Path:
    return Path(os.environ.get('GYM_DATA_DIR') or user_data_path('CommunicationGym',appauthor=False)).resolve()

def is_local_url(url: str) -> bool:
    p=urlparse(url)
    return p.scheme in ('http','https') and p.hostname in ('127.0.0.1','localhost','::1') and not p.username and not p.password and not p.query and not p.fragment

def validate_settings(settings: Settings) -> None:
    if settings.server.host != '127.0.0.1':
        raise ValueError('Communication Gym binds only to 127.0.0.1')
    if not is_local_url(settings.tts.base_url):
        raise ValueError('VoiceStudio must use a loopback URL')
    if not is_local_url(settings.coach.base_url):
        p=urlparse(settings.coach.base_url)
        if not settings.privacy.allow_cloud or p.scheme!='https' or p.username or p.password:
            raise ValueError('Remote coach requires explicit cloud setting and HTTPS without URL credentials')
        if settings.coach.provider=='ollama':
            raise ValueError('Default Ollama adapter is local only; use an explicitly configured HTTP provider')

def load_settings(root: Path) -> Settings:
    path=root/'settings.json'
    settings=Settings.model_validate_json(path.read_text(encoding='utf-8')) if path.exists() else Settings()
    validate_settings(settings)
    return settings

def save_settings(root: Path, settings: Settings) -> None:
    validate_settings(settings)
    root.mkdir(parents=True,exist_ok=True)
    temp=root/'settings.json.tmp'
    temp.write_text(settings.model_dump_json(indent=2),encoding='utf-8')
    temp.replace(root/'settings.json')

class CredentialStore:
    """Isolated runs deliberately cannot read the everyday Windows vault."""
    def __init__(self, isolated: bool = False):
        self.isolated=isolated
        self._values: dict[str,str]={}
    def get(self, account: str) -> str | None:
        if self.isolated: return self._values.get(account)
        import keyring
        return keyring.get_password('CommunicationGym',account)
    def set(self, account: str, value: str) -> None:
        if self.isolated: self._values[account]=value; return
        import keyring
        keyring.set_password('CommunicationGym',account,value)
    def delete(self, account: str) -> None:
        if self.isolated: self._values.pop(account,None); return
        import keyring
        try: keyring.delete_password('CommunicationGym',account)
        except keyring.errors.PasswordDeleteError: pass
