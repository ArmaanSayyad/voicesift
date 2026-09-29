import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

type Plan={day:string|null,time:string|null,party_size:number|null,status:string};
type Turn={index:number,input_asset:string,output_asset:string,transcript:string,response:string,plan:Plan,timings_s:Record<string,number>};
type Manifest={mode:string,variant:string,status:string,turns:Turn[],score?:{value:boolean,expected:Plan,actual:Plan,scope:string},models:unknown,llm:unknown,error?:string};
type Run={id:string,created:string,status:string,scenario:string,error?:string,manifest?:Manifest};
type Event={seq:number,kind:string,relative_ns:number,payload:Record<string,unknown>,parents:number[]};
let token='';
async function api<T>(path:string,options:RequestInit={}):Promise<T>{
 const response=await fetch('/api/v1'+path,{...options,headers:{'Content-Type':'application/json','X-Repair-Token':token,...options.headers}});
 if(!response.ok){const body=await response.json();throw new Error(body.detail||response.statusText)}
 return response.json();
}
function Wave({url}:{url:string}){
 const canvas=useRef<HTMLCanvasElement>(null);
 useEffect(()=>{const abort=new AbortController();let context:AudioContext|undefined;
 (async()=>{const response=await fetch(url,{signal:abort.signal});if(!response.ok)return;const bytes=await response.arrayBuffer();context=new AudioContext();const buffer=await context.decodeAudioData(bytes);if(abort.signal.aborted)return;const c=canvas.current;if(!c)return;const ctx=c.getContext('2d')!;const samples=buffer.getChannelData(0);ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#397369';for(let x=0;x<c.width;x++){const lo=Math.floor(x*samples.length/c.width),hi=Math.floor((x+1)*samples.length/c.width);let peak=0;for(let j=lo;j<hi;j++)peak=Math.max(peak,Math.abs(samples[j]));ctx.fillRect(x,(c.height-peak*c.height)/2,1,Math.max(1,peak*c.height))}})().catch(()=>{}).finally(()=>context?.close());
 return()=>{abort.abort();if(context&&context.state!=='closed')void context.close()};},[url]);
 return <canvas ref={canvas} width={600} height={55} className="wave" aria-label="Audio amplitude waveform"/>;
}
function Audio({run,asset,label}:{run:string,asset:string,label:string}){const url=`/api/v1/runs/${run}/audio/${asset}`;return <div className="clip"><div className="small-label">{label}</div><Wave url={url}/><audio aria-label={label} controls preload="metadata" src={url}/></div>}
function App(){
 const [runs,setRuns]=useState<Run[]>([]),[selected,setSelected]=useState<string|null>(null),[run,setRun]=useState<Run|null>(null),[events,setEvents]=useState<Event[]>([]),[error,setError]=useState(''),[starting,setStarting]=useState(false),[ready,setReady]=useState(false),[integrity,setIntegrity]=useState('');
 const active=runs.some(r=>!['complete','failed','aborted','interrupted'].includes(r.status));
 useEffect(()=>{api<{token:string}>('/bootstrap').then(x=>{token=x.token;setReady(true)}).catch(e=>setError(e.message));},[]);
 useEffect(()=>{if(!ready)return;let alive=true;const refresh=async()=>{try{const list=await api<Run[]>('/runs');if(alive){setRuns(list);setSelected(old=>old||list[0]?.id||null)}}catch(e){if(alive)setError(String(e))}};void refresh();const id=setInterval(refresh,1500);return()=>{alive=false;clearInterval(id)}},[ready]);
 useEffect(()=>{setRun(null);setEvents([]);setIntegrity('');if(!selected)return;let alive=true;const refresh=async()=>{try{const [r,e]=await Promise.all([api<Run>(`/runs/${selected}`),api<Event[]>(`/runs/${selected}/events`)]);if(alive){setRun(r);setEvents(e)}}catch(e){if(alive)setError(String(e))}};void refresh();const id=setInterval(refresh,1500);return()=>{alive=false;clearInterval(id)}},[selected]);
 async function start(){setStarting(true);setError('');try{const r=await api<{id:string}>('/runs',{method:'POST',body:JSON.stringify({scenario_id:'dinner-date-dev'})});setSelected(r.id);setRuns(await api<Run[]>('/runs'))}catch(e){setError(String(e))}finally{setStarting(false)}}
 const m=run?.manifest;
 return <div className="layout"><aside><a className="brand" href="/">↳ <span>Repair Bench</span></a><div className="small-label">EVIDENCE WORKSPACE</div><div className="nav-item">◉ &nbsp; Controlled runs</div><div className="roadmap">Next: live audio, matched C0/C1 comparisons, and human recording.</div><div className="local"><span className="dot"/> Local · private<br/><small>Development slice 01</small></div></aside><main><header><div><div className="eyebrow">CONVERSATION REPAIR / DEVELOPMENT</div><h1>Follow the correction.</h1><p>Inspect what the agent understood, proposed, and generated.</p></div><button disabled={!ready||starting||active} onClick={start}>{starting?'Starting…':active?'Run in progress…':'+ Run dinner correction'}</button></header>
 <div className="notice"><strong>Completed-utterance mode</strong><span>Synthetic speech · C0 history baseline · no live interruption or physical playback measurement</span></div>
 {error&&<div role="alert" className="error">{error}<button className="secondary" onClick={()=>setError('')}>Dismiss</button></div>}
 <div className="workspace"><section className="run-list"><h2>Runs <small>{runs.length}</small></h2>{runs.length===0?<p className="empty">Start a controlled run to create an evidence trail.</p>:runs.map(r=><button className={'run-card '+(r.id===selected?'chosen':'')} key={r.id} onClick={()=>setSelected(r.id)}><strong>Dinner · change the day</strong><span>{new Date(r.created).toLocaleString()}</span><span className={'badge '+r.status}>{r.status}</span><code>{r.id.slice(0,8)}</code></button>)}</section>
 <section className="detail">{!run?<div className="welcome"><span>01 / TRACE A TWO-TURN EXCHANGE</span><h2>“Actually Saturday.<br/>Keep everything else.”</h2><p>The agent first hears a Friday dinner request, then a day-only correction. Both inputs are generated speech. Expected answers are reserved for evaluation.</p><p>Starting a run performs real local ASR, language-model inference, and speech synthesis. It may take longer while models load.</p></div>:<><div className="detail-heading"><div><div className="eyebrow">RUN {run.id.slice(0,8)}</div><h2>Dinner · change the day</h2></div><span className={'badge '+run.status}>{run.status}</span></div>
 {run.error&&<div role="alert" className="error">{run.error}</div>}
 {!m&&<div className="loading">{events.at(-1)?.kind||'Queued'} — evidence appears below as the run progresses.</div>}
 {m?.score&&<div className="result"><div><div className="small-label">STRUCTURED PLAN CHECK</div><strong>{m.score.value?'Expected plan matched':'Plan differs from expected'}</strong><p>One development case. Unreviewed. Not a repair-success benchmark.</p></div><div className="state">{Object.entries(m.score.actual).map(([k,v])=><div key={k}><small>{k.replace('_',' ')}</small><b>{String(v??'unknown')}</b></div>)}</div></div>}
 {m?.turns.map(t=><article className="turn" key={t.index}><h3><span>{String(t.index+1).padStart(2,'0')}</span> {t.index===0?'Initial request':'Correction'}</h3><div className="transcript"><div><div className="small-label">ASR TRANSCRIPT</div><p>{t.transcript}</p></div><div><div className="small-label">GENERATED RESPONSE</div><p>{t.response}</p></div></div><div className="audio-pair"><Audio run={run.id} asset={t.input_asset} label={`Turn ${t.index+1} · synthetic input`}/><Audio run={run.id} asset={t.output_asset} label={`Turn ${t.index+1} · generated response`}/></div><div className="timings">{Object.entries(t.timings_s).map(([k,v])=><span key={k}>{k.toUpperCase()} <b>{v.toFixed(3)} s</b></span>)}<small>Compute duration, not heard-response latency</small></div></article>)}
 <div className="event-header"><h3>Event trail <small>{events.length}</small></h3><span>Host monotonic clock · seconds from run start</span></div><div className="events">{events.map(e=><details key={e.seq}><summary><code>{(e.relative_ns/1e9).toFixed(3)} s</code><strong>{e.kind}</strong><small>#{e.seq}</small></summary><pre>{JSON.stringify({parents:e.parents,...e.payload},null,2)}</pre></details>)}</div>
 {m&&<div className="footer-actions"><a href={`/api/v1/runs/${run.id}/manifest`}>Download manifest</a>{run.status==='complete'&&<button className="secondary" onClick={()=>api<{verified:boolean}>(`/runs/${run.id}/verify`).then(()=>setIntegrity('Artifact hashes and event chain verified.')).catch(e=>setIntegrity(e.message))}>Verify evidence</button>}<small role="status">{integrity}</small></div>}</> }</section></div></main></div>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
