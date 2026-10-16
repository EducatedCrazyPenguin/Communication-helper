// Capture only after an explicit user action. No cloud/browser speech recognition.
export class LocalRecorder {
  constructor({api,onState,onSaved,waitJob,revision}){Object.assign(this,{api,onState,onSaved,waitJob,revision});this.stream=null;this.recorder=null;this.queue=Promise.resolve();this.sequence=0;this.failed=null;this.pending=[];this.timer=null;this.stopping=false;}
  async begin(attempt,maxSeconds){
    this.attempt=attempt;this.maxSeconds=Math.min(maxSeconds,900);
    try{
      this.stream=await navigator.mediaDevices.getUserMedia({audio:true});
      const mime=['audio/webm;codecs=opus','audio/mp4','audio/ogg;codecs=opus'].find(x=>MediaRecorder.isTypeSupported(x));
      if(!mime)throw new Error('This browser cannot record a supported audio format. Upload an audio file instead.');
      this.record=await this.api(`/api/attempts/${attempt.id}/recording`,{expected_revision:this.revision(),media_type:mime,provenance:'microphone'});
      this.recorder=new MediaRecorder(this.stream,{mimeType:mime});
      this.recorder.ondataavailable=event=>{if(event.data.size){const seq=this.sequence++;this.pending.push({seq,blob:event.data});this.queue=this.queue.then(()=>this.flush());}};
      this.recorder.onerror=()=>{this.failed=new Error('The microphone recorder failed. Keep saved chunks and record again.');this.stop();};
      this.recorder.onstop=()=>this.finish();
      this.start=performance.now();this.recorder.start(1000);this.onState('recording',0);
      this.timer=setInterval(()=>{const seconds=(performance.now()-this.start)/1000;this.onState('recording',seconds);if(seconds>=this.maxSeconds)this.stop();},250);
    }catch(error){this.cleanup();this.onState('error',error.message);throw error;}
  }
  async flush(){
    if(this.failed)return;
    while(this.pending.length){
      const item=this.pending[0];
      try{await this.api(`/api/recordings/${this.record.id}/chunks/${item.seq}`,item.blob,'PUT',true);this.pending.shift();}
      catch(error){this.failed=error;this.onState('upload_error','Audio upload stopped. Retry the buffered chunk before finalizing.');if(this.recorder?.state==='recording')this.recorder.stop();break;}
    }
  }
  stop(){if(this.stopping)return;this.stopping=true;clearInterval(this.timer);if(this.recorder?.state==='recording')this.recorder.stop();else this.cleanup();}
  async finish(){
    this.cleanup();await this.queue;
    if(this.failed){this.onState('upload_error',this.failed.message);return;}
    try{this.onState('saving');const job=await this.api(`/api/recordings/${this.record.id}/finalize`,{expected_revision:this.revision(),operation_key:crypto.randomUUID()});await this.waitJob(job);await this.onSaved();}catch(error){this.onState('error',error.message);}
  }
  async retry(){this.failed=null;await this.flush();if(!this.failed)await this.finish();}
  cleanup(){clearInterval(this.timer);this.stream?.getTracks().forEach(track=>track.stop());this.stream=null;}
  async dispose(){this.cleanup();if(this.recorder?.state==='recording')this.recorder.stop();}
}

export async function uploadAudio(file,attempt,{api,revision,waitJob,onProgress}){
  if(file.size>256*1024*1024)throw new Error('Choose a file no larger than 256 MiB.');
  const mime=file.type||'audio/wav';
  const record=await api(`/api/attempts/${attempt.id}/recording`,{expected_revision:revision(),media_type:mime,provenance:'upload'});
  const size=1024*1024;
  for(let start=0,seq=0;start<file.size;start+=size,seq++){
    await api(`/api/recordings/${record.id}/chunks/${seq}`,file.slice(start,start+size),'PUT',true);
    onProgress?.(Math.min(1,(start+size)/file.size));
  }
  const job=await api(`/api/recordings/${record.id}/finalize`,{expected_revision:revision(),operation_key:crypto.randomUUID()});await waitJob(job);return record;
}
