import { GPT2Tokenizer } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
let session, tokenizer, manifest, dictionary, vocabulary, firstIndex;
let lastContext = null, lastLogits = null;
const blocked = new Set(['damn','hell','shit','fuck','bitch','bastard','asshole','crap','idiot']);
const sendStatus = message => self.postMessage({type:'status',message});
const root = new URL('./model/', self.location.href);
async function readJSON(name) {
  const r = await fetch(new URL(name,root));
  if(!r.ok) throw new Error(`Missing ${name}. Check the GitHub Actions model-export step.`);
  return r.json();
}
async function load() {
  if(session) return;
  sendStatus('Downloading tokenizer and dictionary…');
  const [config,raw,words,meta] = await Promise.all([
    readJSON('tokenizer_config.json'),readJSON('tokenizer.json'),readJSON('dictionary.json'),readJSON('manifest.json')]);
  tokenizer = new GPT2Tokenizer(raw,config);
  dictionary=words;manifest=meta;
  vocabulary=Object.entries(raw.model.vocab).filter(([token])=>token.startsWith('Ġ') && /^[A-Za-z]+$/.test(token.slice(1)))
    .map(([token,id])=>({word:token.slice(1),id})).filter(({word})=>!blocked.has(word.toLowerCase()));
  firstIndex=new Map();
  for(const word of Object.keys(words).sort((a,b)=>words[b]-words[a])) {
    if(blocked.has(word)) continue;
    const ids=tokenizer.encode(' '+word,{add_special_tokens:false});
    if(ids.length) {
      if(!firstIndex.has(ids[0])) firstIndex.set(ids[0],[]);
      const bucket=firstIndex.get(ids[0]);
      if(bucket.length<256) bucket.push(word);
    }
  }
  const response=await fetch(new URL('model.onnx',root));
  if(!response.ok) throw new Error('Converted model is missing. Check GitHub Actions export logs.');
  const reader=response.body.getReader(); const chunks=[];let total=0;
  while(true) {
    const {done,value}=await reader.read();if(done) break;
    chunks.push(value);total+=value.length;
    sendStatus(`Downloading model… ${Math.min(100,Math.round(total/manifest.bytes*100))}%`);
  }
  const buffer=new Uint8Array(total);let offset=0;
  for(const chunk of chunks){buffer.set(chunk,offset);offset+=chunk.length;}
  sendStatus('Starting OPT in your browser…');
  session=await ort.InferenceSession.create(buffer,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  await logits([manifest.bos]);
  sendStatus('Model ready');
}
async function logits(ids) {
  const context=ids.slice(-128);if(!context.length) context.push(manifest.bos);
  const key=context.join(',');
  if(lastContext===key) return lastLogits;
  const result=await session.run({
    input_ids:new ort.Tensor('int64',BigInt64Array.from(context,BigInt),[1,context.length]),
    attention_mask:new ort.Tensor('int64',BigInt64Array.from(context,()=>1n),[1,context.length])});
  const data=Float32Array.from(result.logits.data);
  for(const tensor of Object.values(result)) tensor.dispose();
  lastContext=key;lastLogits=data;return data;
}
function logProbability(values,id) {
  let max=-Infinity;for(const x of values) if(x>max) max=x;
  let sum=0;for(const x of values) sum+=Math.exp(x-max);
  return values[id]-max-Math.log(sum);
}
const encode=text=>tokenizer.encode(text,{add_special_tokens:false});
async function scoreWord(word,prefix,mean=false) {
  const left=encode(prefix.trimEnd());if(!left.length) left.push(manifest.bos);
  const suffix=encode((prefix.trim()?' ':'')+word);let score=0;
  for(const token of suffix){score+=logProbability(await logits(left),token);left.push(token);}
  return mean?score/Math.max(1,suffix.length):score;
}
function edits(word) {
  const alphabet='abcdefghijklmnopqrstuvwxyz',out=new Set();
  for(let i=0;i<=word.length;i++){
    const a=word.slice(0,i),b=word.slice(i);
    if(b)out.add(a+b.slice(1));
    if(b.length>1)out.add(a+b[1]+b[0]+b.slice(2));
    for(const c of alphabet){out.add(a+c+b);if(b)out.add(a+c+b.slice(1));}
  }
  return [...out].filter(w=>dictionary[w] && w!==word && !blocked.has(w))
    .sort((a,b)=>dictionary[b]-dictionary[a] || a.localeCompare(b)).slice(0,5);
}
function caseLike(word,original) {
  if(original===original.toUpperCase())return word.toUpperCase();
  return /^[A-Z]/.test(original)?word[0].toUpperCase()+word.slice(1):word;
}
async function predict(p) {
  if(!session) throw new Error('Click Load model first.');
  const started=performance.now(),trusted=new Set(p.trustedWords||[]);
  const history=(p.context||[]).map(x=>x.trim()).filter(Boolean).join(' ');
  const text=(history?history+' ':'')+p.draft;
  let mode='Starter',suggestions=[],punctuation=null;
  if(text.trim()) {
    mode=/\s$/.test(text)?'Next-Word':'Prefix-Completion';
    const match=mode==='Prefix-Completion'?text.match(/\S+$/):null;
    const partial=match?match[0]:'',prefix=match?text.slice(0,match.index).trimEnd():text.trimEnd();
    const values=await logits(encode(prefix));
    const candidates=new Map();
    for(const {word,id} of vocabulary){
      const lower=word.toLowerCase();
      if(!dictionary[lower]&&!trusted.has(lower))continue;
      if(partial&&(!lower.startsWith(partial.toLowerCase())||word.length<=partial.length))continue;
      candidates.set(lower,Math.max(candidates.get(lower)??-Infinity,logProbabilityCached(values,id)));
    }
    const ids=[...firstIndex.keys()].sort((a,b)=>values[b]-values[a]);const extra=new Set();
    for(const id of ids){
      let count=0;
      for(const word of firstIndex.get(id)){
        if(candidates.has(word)||partial&&(!word.startsWith(partial.toLowerCase())||word.length<=partial.length))continue;
        extra.add(word);if(++count>=2)break;
      }
      if(extra.size>=8)break;
    }
    for(const word of trusted) if(!candidates.has(word)&&(!partial||word.startsWith(partial.toLowerCase())&&word.length>partial.length))extra.add(word);
    for(const word of [...extra].slice(0,20))candidates.set(word,await scoreWord(word,prefix));
    const ranked=[...candidates].map(([word,score])=>[word,score+Math.min(.8,(p.cacheWeight||0)*Math.log1p(p.cache?.[word]||0))])
      .sort((a,b)=>b[1]-a[1]).slice(0,3);
    suggestions=ranked.map(([word])=>({word:partial?caseLike(word,partial):word==='i'?'I':word}));
    if(p.draft.trim() && /[A-Za-z0-9]\s*$/.test(p.draft) && /\s/.test(p.draft)){
      const vals=await logits(encode(text.trimEnd()));
      const punct=['.',',','?','!'].map(word=>({word,score:vals[encode(word)[0]]})).sort((a,b)=>b.score-a.score);
      punctuation={word:punct[0].word};
    }
  }
  const corrections=[];
  if(p.spelling){
    for(const m of p.draft.matchAll(/(?<![\w'’@/.-])([A-Za-z]+)(?=[,.!?;:]*\s)/g)){
      const word=m[1],lower=word.toLowerCase();
      if(word.length<3||word.length>24||dictionary[lower]||trusted.has(lower))continue;
      const candidates=edits(lower),prefix=(history?history+' ':'')+p.draft.slice(0,m.index),scores=[];
      for(const candidate of candidates){const display=caseLike(candidate,word);scores.push([display,await scoreWord(display,prefix,true)]);}
      scores.sort((a,b)=>b[1]-a[1]);
      const start=Array.from(p.draft.slice(0,m.index)).length;
      corrections.push({word,start,end:start+word.length,candidates:scores.slice(0,3).map(x=>x[0])});
    }
  }
  return {suggestions,punctuation,corrections,mode,latencyMs:performance.now()-started,
    footprint:(manifest.bytes/1048576).toFixed(1)+' MiB downloaded model',modelLabel:manifest.model,warning:''};
}
let probabilityContext=null,normalizer=0;
function logProbabilityCached(values,id){
  if(probabilityContext!==values){
    let max=-Infinity;for(const x of values)if(x>max)max=x;
    let sum=0;for(const x of values)sum+=Math.exp(x-max);
    normalizer=max+Math.log(sum);probabilityContext=values;
  }
  return values[id]-normalizer;
}
let chain=Promise.resolve();
self.onmessage=({data})=>{
  chain=chain.then(async()=>{
    try{const result=data.type==='load'?(await load(),{loaded:true}):await predict(data.payload);
      self.postMessage({id:data.id,result});}
    catch(error){self.postMessage({id:data.id,error:String(error.message||error)});}
  });
};
