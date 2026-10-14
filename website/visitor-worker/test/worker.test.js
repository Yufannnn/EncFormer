import {test} from 'node:test';
import assert from 'node:assert/strict';
import worker from '../src/index.js';

function dbMock(){
  const rows=new Map();
  const db={
    prepare(query){
      return {
        bind(...args){this.args=args;return this;},
        async run(){const [day,country]=this.args;const key=`${day}-${country}`;rows.set(key,(rows.get(key)||0)+1);return {success:true};},
        async first(){const entries=[...rows.entries()];
          if(query.includes('WHERE day'))return {count:entries.filter(([key])=>key.startsWith(new Date().toISOString().slice(0,10))).reduce((t,[,v])=>t+v,0)};
          return {count:entries.reduce((t,[,v])=>t+v,0)};},
        async all(){if(query.includes('GROUP BY country')){
          const agg=new Map();for(const [key,v] of rows)agg.set(key.slice(11),(agg.get(key.slice(11))||0)+v);
          return {results:[...agg].map(([country,views])=>({country,views}))};
        }
        const agg=new Map();for(const [key,v] of rows)agg.set(key.slice(0,10),(agg.get(key.slice(0,10))||0)+v);
        return {results:[...agg].map(([day,views])=>({day,views}))};}
      };
    }
  };return db;
}
const env={DB:dbMock(),ALLOWED_ORIGINS:'https://yufannnn.github.io',SITE_PATH:'/EncFormer/'};
function req(path,method='GET',origin='https://yufannnn.github.io',country='SG',body){
  return Object.assign(new Request('https://example.workers.dev'+path,{method,headers:{'Origin':origin,'Content-Type':'application/json','User-Agent':'Mozilla/5.0'},body:body?JSON.stringify(body):undefined}),{cf:{country}});
}
test('starts with an empty summary',async()=>{
  const resp=await worker.fetch(req('/api/summary'),env); assert.equal(resp.status,200);
  const data=await resp.json();assert.equal(data.total,0);assert.equal(data.countries.length,0);assert.equal(data.daily.length,14);
});
test('counts visits by UTC day and country without identifiers',async()=>{
  assert.equal((await worker.fetch(req('/api/visit','POST','https://yufannnn.github.io','SG',{page:'/EncFormer/'}),env)).status,201);
  assert.equal((await worker.fetch(req('/api/visit','POST','https://yufannnn.github.io','US',{page:'/EncFormer/'}),env)).status,201);
  const data=await(await worker.fetch(req('/api/summary'),env)).json();
  assert.equal(data.total,2); assert.equal(data.today,2); assert.equal(data.countries.length,2);
  assert.equal(data.viewerCountry,'SG');assert.equal(JSON.stringify(data).includes('192.168'),false);
});
test('CORS rejects disallowed browsers and invalid pages',async()=>{
  assert.equal((await worker.fetch(req('/api/visit','POST','https://bad.example','US',{page:'/EncFormer/'}),env)).status,403);
  assert.equal((await worker.fetch(req('/api/visit','POST','https://yufannnn.github.io','SG',{page:'/wrong/'}),env)).status,400);
});
test('preflight is supported only for allowed site origin',async()=>{
  const good=await worker.fetch(req('/api/visit','OPTIONS'),env);
  assert.equal(good.status,204); assert.equal(good.headers.get('access-control-allow-origin'),'https://yufannnn.github.io');
  assert.equal((await worker.fetch(req('/api/visit','OPTIONS','https://bad.example'),env)).status,403);
});
