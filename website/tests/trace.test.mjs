import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// Small DOM adapter: tests exercise the delivered controls and their visible state.
function setup() {
  function element(dataset={}) {
    const values=new Set();
    return {dataset,children:[],attrs:{},events:{},innerHTML:'',textContent:'',hidden:false,
      style:{},disabled:false,offsetLeft:0,offsetWidth:80,clientWidth:800,
      classList:{add(x){values.add(x);},remove(x){values.delete(x);},toggle(x,on){on?values.add(x):values.delete(x);},contains(x){return values.has(x);}},
      setAttribute(k,v){this.attrs[k]=v;}, addEventListener(k,fn){this.events[k]=fn;},
      append(...children){this.children.push(...children);}, replaceChildren(...children){this.children=[...children];},
      querySelector(selector){return selector==='.is-active'?this.children.find(x=>x.classList.contains('is-active')):null;},
    };
  }
  const ids=Object.fromEntries(['globalTrace','layerTrace','traceLayerLabel','traceIndex','traceTitle','traceDescription','traceDetail','traceNote','traceScene','tracePattern','tracePrev','traceNext','tracePlay','traceModel'].map(x=>[x,element()]));
  ids.traceModel.value='12';
  const views=['single','multi'].map(view=>element({traceView:view}));
  let interval=null, delay=null;
  const document={getElementById:id=>ids[id],createElement:()=>element(),querySelectorAll:()=>views,addEventListener(){},hidden:false};
  const window={matchMedia:()=>({matches:false}),setInterval(fn,ms){interval=fn;delay=ms;return 1;},clearInterval(){interval=null;}};
  vm.runInNewContext(fs.readFileSync(new URL('../../docs/trace.js',import.meta.url),'utf8'),{document,window});
  return {ids,views,click(id){ids[id].events.click();},tick(){assert(interval,'Playback must be running');interval();},get delay(){return delay;},get playing(){return interval!==null;}};
}

test('first layer follows score → MBMax → value, with four complete conversion pairs',()=>{
  const ui=setup();
  ui.ids.globalTrace.children.find(x=>x.textContent==='Layer 1').events.click();
  const titles=[];
  for(let i=0;i<19;i++) {titles.push(ui.ids.traceTitle.textContent);if(i<18)ui.click('traceNext');}
  const find=(name)=>titles.findIndex(x=>x.includes(name));
  assert(find('Attention score')<find('Π_MBMax'));
  assert(find('Π_MBMax')<find('Attention value'));
  assert(find('FFN linear 1')<find('Π_GELU'));
  assert(find('Π_GELU')<find('FFN linear 2'));
  assert.equal(titles.filter(x=>x.startsWith('FHE → MPC:')).length,4);
  assert.equal(titles.filter(x=>x.startsWith('MPC → FHE:')).length,4);
  assert.equal(titles.at(-1),'Encrypted layer output');
  assert.equal(ui.ids.layerTrace.children.length,10);
});

test('playback expands layer 1, then fast-forwards layers 2–12 and decrypts once',()=>{
  const ui=setup();ui.click('tracePlay');assert.equal(ui.delay,3200);
  ui.tick();assert.equal(ui.ids.traceTitle.textContent,'Encrypt hidden states');
  ui.tick();assert.equal(ui.ids.traceTitle.textContent,'Encrypted hidden states');
  for(let i=0;i<18;i++)ui.tick();
  assert.equal(ui.ids.traceTitle.textContent,'Encrypted layer output');
  ui.tick();assert.match(ui.ids.traceTitle.textContent,/Layer 2.*fast-forward/);
  assert.equal(ui.delay,1000);assert.equal(ui.ids.layerTrace.hidden,true);
  for(let i=3;i<=12;i++){ui.tick();assert.match(ui.ids.traceTitle.textContent,new RegExp(`Layer ${i} / 12`));}
  ui.tick();assert.equal(ui.ids.traceTitle.textContent,'Final client-side decryption');
  assert.equal(ui.playing,false);assert.equal(ui.ids.traceNext.disabled,true);
});

test('pause/resume preserves the current operation and replay starts a fresh inference',()=>{
  const ui=setup();ui.click('tracePlay');ui.tick();ui.tick();ui.tick();
  const title=ui.ids.traceTitle.textContent;
  ui.click('tracePlay');assert.equal(ui.playing,false);
  ui.click('tracePlay');assert.equal(ui.ids.traceTitle.textContent,title);
  ui.ids.globalTrace.children.at(-1).events.click();
  ui.click('tracePlay');assert.equal(ui.ids.traceTitle.textContent,'Input sequence');
});

test('BERT-large provides 24 layers, with later layers collapsed until explicitly inspected',()=>{
  const ui=setup();ui.ids.traceModel.value='24';ui.ids.traceModel.events.change();
  assert.equal(ui.ids.globalTrace.children.filter(x=>x.dataset.layer).length,24);
  ui.ids.globalTrace.children.find(x=>x.textContent==='Layer 24').events.click();
  assert.match(ui.ids.traceTitle.textContent,/Layer 24 \/ 24/);assert.equal(ui.ids.layerTrace.hidden,true);
  ui.views[0].events.click();assert.equal(ui.ids.layerTrace.hidden,false);
  assert.equal(ui.ids.traceTitle.textContent,'Encrypted hidden states');
});
