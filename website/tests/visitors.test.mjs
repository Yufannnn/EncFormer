import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source=fs.readFileSync(new URL('../../docs/app.js',import.meta.url),'utf8');
const atlas=source.slice(source.indexOf('  // Global visitor map:'),source.indexOf('  // Cloudflare Web Analytics'));

async function setup({offline=false,optOut=false}={}){
  const elements=new Map(),calls=[],storage=new Map();
  function element(){return {textContent:'',children:[],style:{},events:{},classList:{toggle(){},remove(){}},replaceChildren(){this.children=[];},appendChild(child){this.children.push(child);},setAttribute(){},addEventListener(name,fn){this.events[name]=fn;}};}
  const $=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id);};
  const context={$,number:String,config:{visitorApi:'https://visitors.example'},data:{sampleVisits:{total:1248,today:47,countries:[{country:'SG',views:47}]}},document:{createElementNS:element},navigator:{doNotTrack:optOut?'1':'0'},location:{pathname:'/EncFormer/index.html'},sessionStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)},toast(){},console:{warn(){}},fetch:async(url,options={})=>{
    calls.push({url,options});
    if(url==='assets/countries.json')return {ok:true,json:async()=>({SG:{x:866,y:257,name:'Singapore'}})};
    if(offline)throw Error('Network unavailable');
    if(url.endsWith('/api/visit'))return {ok:true};
    return {ok:true,json:async()=>({total:2,today:2,countries:[{country:'SG',views:2}],viewerCountry:'SG'})};
  }};
  vm.runInNewContext(atlas,context);
  const initial={mode:$('visitorMode').textContent,total:$('visitorTotal').textContent};
  await new Promise(resolve=>setImmediate(resolve));
  return {$,calls,initial,context};
}

test('live atlas loads real data without displaying sample counts first',async()=>{
  const ui=await setup();
  assert.equal(ui.initial.mode,'CONNECTING');assert.equal(ui.initial.total,'—');
  assert.equal(ui.$('visitorMode').textContent,'LIVE DATA');
  assert.equal(ui.$('visitorTotal').textContent,'2');
  assert.equal(ui.$('mapPoints').children.length,2);
  assert.equal(ui.$('yourLocation').textContent,'VIEWING FROM / SINGAPORE');
  const visit=ui.calls.find(call=>call.url.endsWith('/api/visit'));
  assert.equal(JSON.parse(visit.options.body).page,'/EncFormer/index.html');
  await vm.runInNewContext('refreshLive(true)',ui.context);
  assert.equal(ui.calls.filter(call=>call.url.endsWith('/api/visit')).length,1);
});

test('tracking opt-out still permits viewing aggregate data',async()=>{
  const ui=await setup({optOut:true});
  assert.equal(ui.calls.some(call=>call.url.endsWith('/api/visit')),false);
  assert.equal(ui.$('visitorMode').textContent,'LIVE DATA');
});

test('offline atlas never substitutes sample totals; preview remains explicitly labelled',async()=>{
  const ui=await setup({offline:true});
  assert.equal(ui.$('visitorMode').textContent,'LIVE DATA UNAVAILABLE');
  assert.equal(ui.$('visitorTotal').textContent,'—');
  ui.$('visitorDemo').events.click();
  assert.equal(ui.$('visitorMode').textContent,'SAMPLE · NOT LIVE');
  assert.equal(ui.$('visitorTotal').textContent,'1248');
  ui.$('visitorDemo').events.click();
  assert.equal(ui.$('visitorTotal').textContent,'—');
});
