/**
 * Optional EncFormer visitor telemetry.
 * Hosting: Cloudflare Workers + D1. Site itself stays on GitHub Pages.
 * Stores ONLY (UTC day, ISO country code, aggregate visit count).
 * NO stored IP, device fingerprint, cookies, session ID, or full request logs.
 * Not a unique-visitor counter. Public counters are approximate and can be gamed.
 */
const JSON_HEADERS = {'Content-Type': 'application/json; charset=utf-8','Cache-Control':'no-store'};
const MAX_BODY_LENGTH = 512;
const BOT_RE = /bot|crawler|spider|preview|prerender|headless|slack|discord|whatsapp|telegram|lighthouse|python-requests|curl\//i;

function allowedOrigins(env){
  return (env.ALLOWED_ORIGINS || 'https://yufannnn.github.io')
    .split(',').map(x => x.trim()).filter(Boolean);
}
function corsHeaders(origin, env){
  const allowed=allowedOrigins(env);
  return {
    'Access-Control-Allow-Origin': origin && allowed.includes(origin) ? origin : 'null',
    'Access-Control-Allow-Methods':'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers':'Content-Type',
    'Access-Control-Max-Age':'3600',
    'Vary':'Origin',
  };
}
function json(payload,status=200,headers={}){
  return new Response(JSON.stringify(payload),{status,headers:{...JSON_HEADERS,...headers}});
}
function getCountry(request){
  const cc=request.cf?.country || 'ZZ';
  return /^[A-Z]{2}$/.test(cc)?cc:'ZZ';
}
function formatUTCDate(date){return date.toISOString().slice(0,10);}
function last14days(dailyRows){
  const look=new Map(dailyRows.map(x=>[x.day,Number(x.views||0)]));
  const now=new Date();now.setUTCHours(0,0,0,0);
  return Array.from({length:14},(_,i)=>{
    const date=new Date(now);date.setUTCDate(date.getUTCDate()-(13-i));
    const day=formatUTCDate(date);return {day,views:look.get(day)||0};
  });
}
export default {
  async fetch(request,env){
    if(!env.DB)return json({error:'D1 database is not bound'},503);
    const origin=request.headers.get('Origin') || '';
    const url=new URL(request.url);
    const headers=corsHeaders(origin,env);
    if(request.method==='OPTIONS'){
      if(!origin || !allowedOrigins(env).includes(origin))return new Response(null,{status:403,headers});
      return new Response(null,{status:204,headers});
    }
    // A browser website must use an explicitly allowed origin; unknown-origin CLI GET is allowed.
    // This is not authentication: public reporting is intentional.
    if(origin && !allowedOrigins(env).includes(origin))return json({error:'Origin not permitted'},403,headers);
    try {
      if(url.pathname==='/api/visit' && request.method==='POST'){
        if(!origin || !allowedOrigins(env).includes(origin))return json({error:'Browser origin required'},403,headers);
        if(BOT_RE.test(request.headers.get('User-Agent')||''))return json({ok:true,skipped:'bot'},200,headers);
        const length=Number(request.headers.get('Content-Length')||0);
        if(length>MAX_BODY_LENGTH)return json({error:'Request too large'},413,headers);
        const bodyText=await request.text();
        if(bodyText.length>MAX_BODY_LENGTH)return json({error:'Request too large'},413,headers);
        let body;
        try{body=JSON.parse(bodyText);}catch{return json({error:'Invalid JSON'},400,headers);}
        const expected=env.SITE_PATH || '/EncFormer/';
        if(typeof body?.page!=='string' || !(body.page===expected || body.page===expected.slice(0,-1) || body.page===`${expected}index.html`)){
          return json({error:'Invalid site path'},400,headers);
        }
        const country=getCountry(request);
        const day=formatUTCDate(new Date());
        // D1 atomic upsert: no personal data is inserted.
        await env.DB.prepare(`INSERT INTO daily_country_views(day, country, views) VALUES (?, ?, 1)
          ON CONFLICT(day, country) DO UPDATE SET views=views+1`).bind(day,country).run();
        return json({ok:true},201,headers);
      }
      if(url.pathname==='/api/summary' && request.method==='GET'){
        const [t,d,c,trend]=await Promise.all([
          env.DB.prepare('SELECT COALESCE(SUM(views), 0) AS count FROM daily_country_views').first(),
          env.DB.prepare("SELECT COALESCE(SUM(views), 0) AS count FROM daily_country_views WHERE day = date('now')").first(),
          env.DB.prepare('SELECT country, SUM(views) AS views FROM daily_country_views GROUP BY country ORDER BY views DESC LIMIT 80').all(),
          env.DB.prepare("SELECT day, SUM(views) AS views FROM daily_country_views WHERE day >= date('now', '-13 days') GROUP BY day ORDER BY day").all(),
        ]);
        const countries=(c.results||[]).map(row=>({country:String(row.country),views:Number(row.views||0)}));
        return json({total:Number(t?.count||0),today:Number(d?.count||0),countries,daily:last14days(trend.results||[]),viewerCountry:getCountry(request),updatedAt:new Date().toISOString()},200,headers);
      }
      if(url.pathname==='/health' && request.method==='GET')return json({ok:true,service:'encformer-visitors'},200,headers);
      return json({error:'Not found'},404,headers);
    } catch(err){
      // Do not expose database internals to the browser.
      console.error('EncFormer telemetry error',err instanceof Error?err.message:'unknown');
      return json({error:'Analytics temporarily unavailable'},503,headers);
    }
  },
};
