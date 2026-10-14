/* EncFormer interactive research site — no build tools or framework required. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const data = window.ENCFORMER_DATA;
  const config = window.ENCFORMER_CONFIG || {};
  const number = (v) => Number(v).toLocaleString('en-US');
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let toastTimer = null;
  function toast(message) {
    const el = $('toast'); el.textContent = message; el.classList.add('is-visible');
    clearTimeout(toastTimer); toastTimer = window.setTimeout(() => el.classList.remove('is-visible'), 2900);
  }
  async function copyText(s) {
    try { await navigator.clipboard.writeText(s); toast('Copied to clipboard'); }
    catch { toast('Clipboard access unavailable; select and copy the text'); }
  }

  // Theme + navigation.
  try { if (localStorage.getItem('encformer-theme') === 'light') document.documentElement.dataset.theme = 'light'; } catch (_) {}
  $('themeToggle').addEventListener('click', () => {
    const light = document.documentElement.dataset.theme !== 'light';
    document.documentElement.dataset.theme = light ? 'light' : 'dark';
    try { localStorage.setItem('encformer-theme', light ? 'light' : 'dark'); } catch (_) {}
  });
  $('menuToggle').addEventListener('click', () => {
    const nav = $('mobileNav'); nav.hidden = !nav.hidden;
    $('menuToggle').setAttribute('aria-expanded', String(!nav.hidden));
  });
  document.querySelectorAll('.mobile-nav a').forEach((a) => a.addEventListener('click', () => {
    $('mobileNav').hidden = true; $('menuToggle').setAttribute('aria-expanded', 'false');
  }));
  let scrollScheduled = false;
  window.addEventListener('scroll', () => {
    if (scrollScheduled) return; scrollScheduled = true;
    window.requestAnimationFrame(() => {
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      $('readProgress').style.width = `${scrollable ? Math.min(100,window.scrollY / scrollable * 100) : 0}%`;
      scrollScheduled = false;
    });
  }, {passive:true});
  const sections = ['inferenceTrace','concept','architecture','algorithms','benchmarks'];
  if ('IntersectionObserver' in window) {
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((entry) => { if (!entry.isIntersecting) return;
        document.querySelectorAll('.desktop-nav a').forEach((a) => a.classList.toggle('is-active', a.hash === `#${entry.target.id}`));
      });
    }, {rootMargin:'-25% 0px -65% 0px'});
    sections.forEach((s) => obs.observe($(s)));
  }

  const arch = [
    {label:'STAGE 01 / CLIENT', title:'Private input, private key', desc:'The client holds the input and CKKS secret key. It sends ciphertext and evaluation material, never the input plaintext. The server owns the model.', repr:'Plaintext → CKKS ciphertext', security:'Client-held secret key'},
    {label:'STAGE 02 / GPU FHE', title:'Projection kernels, carefully packed', desc:'GPU-accelerated CKKS evaluates plaintext–ciphertext projections for Q, K and V. Weight pre-permutation places outputs in consumer-friendly segment-column and head-major orders.', repr:'Encrypted Q, K, V', security:'FHE · segment-column packing'},
    {label:'STAGE 03 / GPU FHE', title:'Attention scores, folded-diagonal', desc:'The ciphertext–ciphertext score kernel forms encrypted QKᵀ via specialized rotation banks. It exports attention scores in minimal folded-diagonal packing.', repr:'Segment-column → folded-diagonal', security:'FHE · no inter-kernel remap'},
    {label:'STAGE 04 / 2-PARTY MPC', title:'Secure attention nonlinearity', desc:'Protected conversion turns encrypted scores into secret shares. MPC computes the fifth-power MBMax softmax surrogate and sends the result back through a minimal conversion boundary.', repr:'CKKS ⇄ additive shares', security:'3 secure multiplication rounds'},
    {label:'STAGE 05 / GPU FHE', title:'Direct value and output kernels', desc:'The value kernel consumes folded-diagonal attention weights plus head-major V without a separate FHE remap. Plaintext output weights absorb layout adaptation.', repr:'Folded-diagonal + head-major → output', security:'FHE · stage-compatible layouts'},
    {label:'STAGE 06 / 2-PARTY MPC', title:'Secure normalization and activation', desc:'The encoder also uses two MBLN blocks and a BOLT-style GELU block. The latter uses comparisons and selection in MPC; the boundary-cost rule can justify CKKS pre-evaluation of GELU candidates.', repr:'CKKS ⇄ secret shares ⇄ CKKS', security:'0 MBLN + 4 GELU MPC rounds'},
  ];
  document.querySelectorAll('[data-arch]').forEach((b) => b.addEventListener('click', () => {
    const i = Number(b.dataset.arch), a = arch[i];
    document.querySelectorAll('[data-arch]').forEach((x) => { const on = x === b; x.classList.toggle('is-selected',on); x.setAttribute('aria-pressed',String(on)); });
    $('archLabel').textContent=a.label; $('archTitle').textContent=a.title; $('archDescription').textContent=a.desc;
    $('archRepresentation').textContent=a.repr; $('archSecurity').textContent=a.security;
  }));

  // SCP packing studio: conceptual 4x4 illustrations, not actual CKKS slot arithmetic.
  const layoutData = {
    segment:{label:'LAYOUT 01 / PROJECTION', title:'Segment-column', desc:'Each contiguous segment holds a token-column slice. Projection outputs can enter attention kernels directly in the required segment order.', cell:(i)=>({label:`t${i%4} · c${Math.floor(i/4)}`, sub:`SEG ${Math.floor(i/4)}`, head:Math.floor(i/4)})},
    diagonal:{label:'LAYOUT 02 / SCORES', title:'Folded-diagonal', desc:'The score kernel folds two attention diagonals into a ciphertext and exports the resulting attention tensor in minimal form at the MPC boundary.', cell:(i)=>({label:`D${Math.floor(i/4)} · t${i%4}`, sub:`OFFSET ${Math.floor(i/4)}`, head:(Math.floor(i/4)+i%4)%4})},
    head:{label:'LAYOUT 03 / VALUES', title:'Head-major', desc:'Features belonging to the same attention head are organized together, so the value kernel and pre-permuted output projection can consume them directly.', cell:(i)=>({label:`h${Math.floor(i/4)} · f${i%4}`, sub:`HEAD ${Math.floor(i/4)}`, head:Math.floor(i/4)})},
  };
  let brokenPacking=false;
  function updatePacking(which){
    const layout=layoutData[which], grid=$('slotGrid'); grid.replaceChildren();
    for(let i=0;i<16;i++){
      const v=layout.cell(i), cell=document.createElement('div');cell.className='slot-cell';cell.dataset.head=String(v.head);
      cell.style.animationDelay=`${i*15}ms`;
      const strong=document.createElement('strong');strong.textContent=v.label;
      const small=document.createElement('small');small.textContent=v.sub;
      cell.append(strong,small);grid.appendChild(cell);
    }
    $('layoutLabel').textContent=layout.label; $('layoutTitle').textContent=layout.title; $('layoutDescription').textContent=layout.desc;
    document.querySelectorAll('[data-layout]').forEach((b)=>{const yes=b.dataset.layout===which;b.classList.toggle('is-selected',yes);b.setAttribute('aria-selected',String(yes));});
    $('packingOutcome').textContent=brokenPacking ? 'Without a compatible producer/consumer interface, a ciphertext remap is required. SCP avoids these FHE-side remaps by design.' : 'SCP aligns producer and consumer layouts directly: no FHE-side repacking between adjacent kernels.';
  }
  document.querySelectorAll('[data-layout]').forEach((b)=>b.addEventListener('click',()=>updatePacking(b.dataset.layout)));
  $('breakPacking').addEventListener('click',()=>{
    brokenPacking=!brokenPacking;
    $('breakPacking').textContent = brokenPacking?'✓  Restore SCP compatibility':'↗  Show why layout compatibility matters';
    $('packingOutcome').style.borderColor=brokenPacking?'var(--orange)':'var(--aqua)';
    updatePacking(document.querySelector('[data-layout].is-selected').dataset.layout);
  });
  updatePacking('segment');

  // Complex conversion payload toy: precise count ceil(k/2), never presented as end-to-end timing.
  function updateConversion(){
    const n=Number($('vectorRange').value), packed=Math.ceil(n/2);
    $('vectorValue').textContent=n; $('realCount').textContent=`${n} CT`; $('complexCount').textContent=`${packed} CT`;
    $('conversionSaving').textContent=`${Math.round((n-packed)/n*100)}%`;
    const real=$('realPacking'), complex=$('complexPacking');real.replaceChildren();complex.replaceChildren();
    for(let i=0;i<n;i++){
      const tile=document.createElement('div');tile.className='ciphertext-token';
      const name=document.createElement('b');name.textContent=`ct${i+1}`;
      const label=document.createElement('small');label.textContent=`Re: v${i+1}`;
      tile.append(name,label);real.appendChild(tile);
    }
    for(let i=0;i<packed;i++){
      const tile=document.createElement('div');tile.className='ciphertext-token';
      const name=document.createElement('b');name.textContent=`ct${i+1}`;
      const label=document.createElement('small');label.textContent=`v${2*i+1} + i·${2*i+2<=n?'v'+(2*i+2):'0'}`;
      tile.append(name,label);complex.appendChild(tile);
    }
  }
  $('vectorRange').addEventListener('input',updateConversion);updateConversion();

  const protocols = {
    mbmax:{label:'ALGORITHM 01 / SECURE MBMAX',formula:'Y = (X + c)<sup>5</sup> / R<sub>d</sub>',desc:'Batch Power-Max uses a fifth-power softmax surrogate. A public shift and offline-distilled normalization constant keep the MPC multiplication chain short.',steps:['Shift by public c','Square securely','Square again → fourth power','Multiply by X+c; scale by 1/Rd'],note:'Paper §IV-B1: 3 interactive rounds. The constants c and R<sub>d</sub> are public and distilled offline; the display is symbolic.'},
    mbln:{label:'ALGORITHM 02 / SECURE MBLN',formula:'y = γ̃ ⊙ (x − μ) + β',desc:'Batch LayerNorm replaces variance-dependent normalization with a public scale. Shares can be summed, centered and affinely transformed without interactive multiplication.',steps:['Secure shares of a row','Local row sum','Subtract public-scaled mean','Local affine map'],note:'Paper §IV-B2: 0 interactive MPC rounds for MBLN under this surrogate; γ̃ folds in the public normalization scale.'},
    gelu:{label:'PROTOCOL / BOLT-STYLE GELU',formula:'select { 0, F<sub>0</sub>(x), F<sub>1</sub>(x), x }',desc:'EncFormer follows the BOLT piecewise GELU surrogate: CKKS evaluates candidate polynomial ciphertexts; MPC securely compares and selects the appropriate candidate.',steps:['Generate candidate polynomials','Protected CKKS → MPC conversion','Private interval comparisons','Select result with secure mux'],note:'Paper §IV-B3 and Appendix B: 4 interactive MPC rounds. Candidate evaluation may occur before conversion under a favorable boundary decision.'},
  };
  function selectProtocol(which){const p=protocols[which];
    $('protocolLabel').textContent=p.label; $('protocolFormula').innerHTML=p.formula; $('protocolDescription').textContent=p.desc; $('protocolNote').innerHTML=p.note;
    const container=$('protocolSteps');container.replaceChildren();p.steps.forEach((step,i)=>{const node=document.createElement('div');node.className='protocol-step';node.innerHTML=`<span>${String(i+1).padStart(2,'0')}</span><strong>${esc(step)}</strong>`;container.appendChild(node);});
    document.querySelectorAll('[data-protocol]').forEach((b)=>{let active=b.dataset.protocol===which;b.classList.toggle('is-selected',active);b.setAttribute('aria-selected',String(active));});
  }
  document.querySelectorAll('[data-protocol]').forEach((b)=>b.addEventListener('click',()=>selectProtocol(b.dataset.protocol)));selectProtocol('mbmax');

  // Boundary cost model: direct transcription of manuscript Eq. 5-6 (illustrative slider inputs).
  let profile='LAN';
  const inputs=['extraCts','ctMb','roundsSaved','computeSaved','roundMb','extraRtts'];
  const labels=['extraCtsOut','ctMbOut','roundsSavedOut','computeSavedOut','roundMbOut','extraRttsOut'];
  function updateCost(){
    inputs.forEach((id,i)=>$(labels[i]).textContent=(id==='ctMb'||id==='roundMb')?Number($(id).value).toFixed(1):$(id).value);
    const n=data.networks[profile], mbps=n.mbps, rtt=n.rtt;
    const extraCts=Number($('extraCts').value), ctMb=Number($('ctMb').value), roundsSaved=Number($('roundsSaved').value),
      compute=Number($('computeSaved').value), roundMb=Number($('roundMb').value), extraRtts=Number($('extraRtts').value);
    // 1 MB is 1e6 bytes, Mbps is 1e6 bits/s; seconds to ms => 8000 * MB/Mbps.
    const extra = extraCts*ctMb*8000/mbps + extraRtts*rtt;
    const savings = compute + roundsSaved*(rtt+roundMb*8000/mbps);
    const delta=extra-savings, good=delta<0;
    $('decisionDelta').textContent=`${delta<0?'−':delta>0?'+':''}${number(Math.round(Math.abs(delta)))} ms`;
    $('decisionDelta').classList.toggle('is-negative-result',!good);
    $('decisionOutcome').textContent=good?'CANDIDATE FAVORED':'MINIMAL BASELINE FAVORED';
    $('decisionOutcome').classList.toggle('is-negative-result',!good);
    $('extraCostValue').textContent=`${number(Math.round(extra))} ms`;
    $('savingsValue').textContent=`${number(Math.round(savings))} ms`;
    const max=Math.max(extra,savings,1);
    $('extraCostBar').style.width=`${Math.max(0,100*extra/max)}%`;
    $('savingsBar').style.width=`${Math.max(0,100*savings/max)}%`;
    $('profileDescription').textContent=`${number(mbps)} Mbps bandwidth · ${rtt} ms RTT`;
  }
  inputs.forEach((id)=>$(id).addEventListener('input',updateCost));
  document.querySelectorAll('[data-profile]').forEach((b)=>b.addEventListener('click',()=>{
    profile=b.dataset.profile;document.querySelectorAll('[data-profile]').forEach((x)=>x.classList.toggle('is-active',x===b)); updateCost();
  })); updateCost();

  // Exact end-to-end data from Table VIII.
  function updateBench(){
    const model=$('benchModel').value, network=$('benchNetwork').value, metric=$('benchMetric').value;
    const rows=data.benchmarks[model], names=['BOLT','BumbleBee','BLB','EncFormer'];
    const metricVal=(name)=>metric==='comm'?rows[name].comm:rows[name][network];
    const values=names.map(metricVal), max=Math.max(...values);
    const chart=$('benchmarkBars');chart.replaceChildren();
    names.forEach((name,i)=>{
      const val=values[i], row=document.createElement('div');row.className='chart-row'+(name==='EncFormer'?' is-ours':'');
      const label=document.createElement('span');label.className='chart-name';label.textContent=name;
      const axis=document.createElement('div');axis.className='chart-axis';
      const fill=document.createElement('div');fill.className='chart-bar';fill.style.width=`${Math.max(1,(val/max)*100)}%`;axis.appendChild(fill);
      const value=document.createElement('span');value.className='chart-value';value.textContent=`${val.toFixed(1)} ${metric==='comm'?'GB':'min'}`;
      row.append(label,axis,value);chart.appendChild(row);
    });
    const baseline=metricVal('BLB'), ours=metricVal('EncFormer'), ratio=baseline/ours;
    $('benchSummary').innerHTML=`<strong>${ratio.toFixed(2)}× lower ${metric==='comm'?'online MPC communication':'end-to-end latency'}</strong> versus BLB for ${esc(rows.label)}${metric==='comm'?' (network-independent bytes)':' on '+esc(network)}. Other ratios vary by model and setting.`;
  }
  ['benchModel','benchNetwork','benchMetric'].forEach(id=>$(id).addEventListener('change',updateBench));updateBench();
  function renderAccuracy(){
    const graph=$('accuracyGraph');graph.replaceChildren();
    data.accuracy.forEach((r)=>{
      const row=document.createElement('div');row.className='accuracy-row';
      const title=document.createElement('div');title.className='accuracy-row-title';title.innerHTML=`<span>${esc(r.name)}</span><span>${r.ours.toFixed(2)}% ENC</span>`;
      const bars=document.createElement('div');bars.className='accuracy-bars';
      [['Public baseline','full',r.full],['Surrogate teacher','teacher',r.teacher],['Encrypted','ours',r.ours]].forEach(([name,style,v])=>{
        const line=document.createElement('div');line.className='accuracy-bar-item '+style;line.title=`${name}: ${v.toFixed(2)}%`;
        const track=document.createElement('div');track.className='accuracy-bar-track';const fill=document.createElement('div');fill.className='accuracy-bar-fill';fill.style.width=`${v}%`;track.appendChild(fill);
        const label=document.createElement('span');label.textContent=`${v.toFixed(1)}%`;line.append(track,label);bars.appendChild(line);
      });row.append(title,bars);graph.appendChild(row);
    });
    const key=document.createElement('div');key.className='accuracy-key';
    key.innerHTML='<span><b></b>Public baseline</span><span><b></b>Surrogate teacher</span><span><b></b>EncFormer</span>';graph.appendChild(key);
  }renderAccuracy();

  // Global visitor map: generated from local Natural Earth outlines. Counters are never invented.
  const fallbackCoords={SG:{x:866,y:257,name:'Singapore'},US:{x:240,y:150,name:'United States'},GB:{x:541,y:132,name:'United Kingdom'},DE:{x:570,y:134,name:'Germany'},IN:{x:798,y:215,name:'India'},JP:{x:960,y:163,name:'Japan'},AU:{x:937,y:352,name:'Australia'},FR:{x:545,y:153,name:'France'}};
  let coords=fallbackCoords, live=false, showingSample=true, lastLiveData=null;
  const isoName=(iso)=>coords[iso]?.name || ({GB:'United Kingdom',SG:'Singapore',US:'United States'}[iso]||iso);
  function addSVG(parent,name,attrs){const el=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,String(v)));parent.appendChild(el);return el;}
  function drawVisitors(summary, sample=false){
    showingSample=sample;
    $('visitorMode').textContent=sample?'SAMPLE · NOT LIVE':'LIVE DATA';
    $('visitorMode').classList.toggle('is-live',!sample);
    $('visitorTotal').textContent=number(summary.total||0);
    $('visitorToday').textContent=number(summary.today||0);
    $('visitorTotalCaption').textContent=sample?'Illustrative numbers · not visits':'Recorded page visits · not unique users';
    const countryEntries=Array.isArray(summary.countries)?summary.countries:[];
    $('visitorCountries').textContent=number(countryEntries.filter(x=>x.country && x.country!=='ZZ').length);
    const points=$('mapPoints');points.replaceChildren();
    const max=Math.max(1,...countryEntries.map(x=>x.views));
    countryEntries.forEach((v,index)=>{
      const c=coords[v.country];if(!c)return;
      const radius=5+Math.sqrt(v.views/max)*12;
      const halo=addSVG(points,'circle',{cx:c.x,cy:c.y,r:radius+9,class:'map-point-halo'});
      halo.style.animationDelay=`${(index%8)*.23}s`;
      const point=addSVG(points,'circle',{cx:c.x,cy:c.y,r:radius,class:'map-point'});
      const title=addSVG(point,'title',{});title.textContent=`${isoName(v.country)}: ${number(v.views)} ${sample?'sample':'recorded'} visits`;
    });
    $('visitorDemo').textContent=sample?'Hide sample data':'Show sample data';
    if(sample)$('yourLocation').textContent='PREVIEW / GEOGRAPHY IS ILLUSTRATIVE';
  }
  function clearVisitors(){
    showingSample=false; $('visitorMode').textContent=live?'LIVE · NO VISITS YET':'TELEMETRY NOT CONFIGURED';
    $('visitorMode').classList.toggle('is-live',live);
    ['visitorTotal','visitorToday','visitorCountries'].forEach(id=>$(id).textContent=live?'0':'—');
    $('visitorTotalCaption').textContent=live?'Recorded page visits':'Configure an analytics endpoint for live data';
    $('mapPoints').replaceChildren();
    $('visitorDemo').textContent='Show sample data';
    $('yourLocation').textContent='NO LIVE LOCATIONS';
  }
  $('visitorDemo').addEventListener('click',()=>{
    if(showingSample){if(live&&lastLiveData)drawVisitors(lastLiveData,false);else clearVisitors();}
    else drawVisitors(data.sampleVisits,true);
  });
  async function loadCoords(){
    try{const r=await fetch('assets/countries.json');if(!r.ok)throw Error('Country lookup');coords={...fallbackCoords,...await r.json()};}
    catch(_){ /* hardcoded preview country locations remain available */ }
  }
  async function refreshLive(record){
    if(!config.visitorApi)return;
    const base=String(config.visitorApi).replace(/\/+$/,'');
    if(!/^https:\/\//.test(base)&&!/^http:\/\/localhost(?::\d+)?/.test(base)){toast('Analytics URL must use HTTPS');return;}
    try{
      let shouldCount=record;
      // Respect browser-level opt-outs. Counts are not unique visitors.
      if(navigator.doNotTrack==='1'||navigator.globalPrivacyControl)shouldCount=false;
      let visited=false;
      try{visited=sessionStorage.getItem('encformer-visit-recorded')==='yes';}catch(_){}
      if(shouldCount&&!visited){
        const p=await fetch(`${base}/api/visit`,{method:'POST',mode:'cors',headers:{'Content-Type':'application/json'},body:JSON.stringify({page:location.pathname}),cache:'no-store'});
        if(!p.ok)throw Error(`Tracking endpoint returned ${p.status}`);
        try{sessionStorage.setItem('encformer-visit-recorded','yes');}catch(_){}
      }
      const r=await fetch(`${base}/api/summary`,{cache:'no-store',mode:'cors'});if(!r.ok)throw Error(`Summary returned ${r.status}`);
      const s=await r.json();if(!s||typeof s.total!=='number'||!Array.isArray(s.countries))throw Error('Unexpected response');
      live=true;lastLiveData=s;drawVisitors(s,false);
      if(s.viewerCountry&&s.viewerCountry!=='ZZ')$('yourLocation').textContent=`VIEWING FROM / ${isoName(s.viewerCountry).toUpperCase()}`;
    }catch(err){
      console.warn('Visitor analytics unavailable:',err);
      live=false;
      if(!showingSample)clearVisitors();
      $('visitorMode').textContent='TELEMETRY OFFLINE · PREVIEW ONLY';
      $('visitorMode').classList.remove('is-live');
    }
  }
  $('visitorRefresh').addEventListener('click',()=>config.visitorApi?refreshLive(false):toast('No visitor endpoint configured; see website/DEPLOY.md'));
  async function initializeVisitors(){
    await loadCoords();drawVisitors(data.sampleVisits,true); // explicitly labeled sample, never misrepresented as a real count
    if(config.visitorApi)refreshLive(true);
  }
  initializeVisitors();

  // Cloudflare Web Analytics adds an owner's private analytics dashboard; never embeds a secret token.
  if(config.cloudflareAnalyticsToken && /^[a-zA-Z0-9_-]{10,100}$/.test(config.cloudflareAnalyticsToken)){
    const s=document.createElement('script');s.defer=true;s.src='https://static.cloudflareinsights.com/beacon.min.js';
    s.setAttribute('data-cf-beacon',JSON.stringify({token:config.cloudflareAnalyticsToken}));document.head.appendChild(s);
  }

  $('copyCitation').addEventListener('click',()=>copyText($('bibtex').textContent));
  $('shareSite').addEventListener('click',async()=>{
    const url=location.href.split('#')[0];
    if(navigator.share){try{await navigator.share({title:'EncFormer · Secure Transformer Inference',url});return;}catch(e){if(e.name==='AbortError')return;}}
    copyText(url);
  });

  // Lightweight, dependency-free command palette.
  const actions=[
    {label:'Acknowledgements, institutions & resources',short:'Page',fn:()=>{ window.location.href='acknowledgements.html'; }},
    {label:'Follow the EncFormer execution graph',short:'Trace',id:'inferenceTrace'},
    {label:'Understanding the problem',short:'Concept',id:'concept'},
    {label:'Explore the encrypted pipeline',short:'Architecture',id:'architecture'},
    {label:'Try Stage Compatible Patterns',short:'Algorithms',id:'algorithms'},
    {label:'Explore published benchmarks',short:'Results',id:'benchmarks'},
    {label:'Open the visitor atlas',short:'Global visitors',id:'visitors'},
    {label:'Find paper, code & citation',short:'Resources',id:'resources'},
    {label:'Copy BibTeX citation',short:'Action',fn:() => copyText($('bibtex').textContent)},
    {label:'Open official GitHub repository',short:'External',fn:()=>window.open('https://github.com/Yufannnn/EncFormer','_blank','noopener')},
  ];
  let commandChoices=[], commandSelection=0;
  function renderChoices(query=''){
    const term=query.trim().toLowerCase();commandChoices=actions.filter(a=>`${a.label} ${a.short}`.toLowerCase().includes(term));commandSelection=0;
    const host=$('commandResults');host.replaceChildren();
    commandChoices.forEach((a,i)=>{const b=document.createElement('button');b.type='button';b.className='command-choice'+(i===0?' is-selected':'');
      const t=document.createElement('strong');t.textContent=a.label;const hint=document.createElement('span');hint.textContent=a.short;b.append(t,hint);
      b.addEventListener('click',()=>runCommand(i));host.appendChild(b);
    });
    if(!commandChoices.length){const p=document.createElement('p');p.textContent='No matching pages or actions.';p.style.padding='10px 15px';host.appendChild(p);}
  }
  function updateChoiceFocus(){document.querySelectorAll('.command-choice').forEach((b,i)=>b.classList.toggle('is-selected',i===commandSelection));}
  function closeCommand(){$('commandDialog').hidden=true;document.body.style.overflow='';}
  function openCommand(){$('commandDialog').hidden=false;document.body.style.overflow='hidden';$('commandInput').value='';renderChoices();$('commandInput').focus();}
  function runCommand(i){const a=commandChoices[i];if(!a)return;closeCommand();if(a.id){$(a.id).scrollIntoView({behavior:prefersReducedMotion?'instant':'smooth'});location.hash=`#${a.id}`;}else a.fn?.();}
  $('searchToggle').addEventListener('click',openCommand);
  $('commandInput').addEventListener('input',(e)=>renderChoices(e.target.value));
  $('commandDialog').addEventListener('click',(e)=>{if(e.target===$('commandDialog'))closeCommand();});
  document.addEventListener('keydown',(e)=>{
    if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();$('commandDialog').hidden?openCommand():closeCommand();return;}
    if($('commandDialog').hidden)return;
    if(e.key==='Escape'){closeCommand();return;}
    if(e.key==='ArrowDown'){e.preventDefault();commandSelection=Math.min(commandSelection+1,commandChoices.length-1);updateChoiceFocus();}
    if(e.key==='ArrowUp'){e.preventDefault();commandSelection=Math.max(commandSelection-1,0);updateChoiceFocus();}
    if(e.key==='Enter'){e.preventDefault();runCommand(commandSelection);}
  });
})();
