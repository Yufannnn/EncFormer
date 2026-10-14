/* Educational BERT encoder graph: no encryption, model execution, or timing measurement. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const escape = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const phases = ['Hidden state','QKV projection','Score kernel','MBMax bridge','Value / output','LayerNorm 1','FFN linear 1','GELU bridge','FFN linear 2','Layer output'];
  const domains = {client:'CLIENT SIDE', fhe:'GPU CKKS / FHE', mpc:'TWO-PARTY MPC', boundary:'SECURE CONVERSION'};
  const node = (title, detail, domain) => ({title, detail, domain});
  const enc = (title, detail) => node(title, detail, 'fhe');
  const shares = (title, detail) => node(title, detail, 'mpc');
  const bridge = (title) => node(title, 'Complex CKKS ↔ additive shares', 'boundary');
  const operation = (key, phase, domain, title, desc, detail, note, nodes) => ({key, phase, domain, title, desc, detail, note, nodes});
  const steps = [
    operation('hidden',0,'fhe','Encrypted hidden states','The encrypted hidden representation enters this Transformer layer.','The server evaluates with ciphertexts and evaluation keys; the CKKS secret key stays with the client.','The layer input is retained for the attention residual connection.',[enc('Layer input','Encrypted hidden state X'),enc('SCP packing','Consumer-compatible layout'),enc('Residual branch','Retain encrypted X')]),
    operation('qkv',1,'fhe','Q / K / V projection','GPU CKKS evaluates the three encrypted linear projections.','Plaintext model weights act on encrypted hidden states. Q and K use score-compatible segment-column layouts; V uses head-major packing.','SCP coordinates the projection outputs with their immediate consumers.',[enc('Hidden states','Encrypted X'),enc('QKV projections','XWQ · XWK · XWV'),enc('Q / K / V','Encrypted, compatible layouts')]),
    operation('score',2,'fhe','Attention score kernel','The GPU computes encrypted attention scores from Q and K.','The ciphertext–ciphertext kernel forms QKᵀ. Scaled scores are exported in minimal folded-diagonal packing for the secure softmax boundary.','Attention weights do not exist yet: MBMax must run before the value kernel.',[enc('Encrypted Q, K','Projection outputs'),enc('Score kernel','S = QKᵀ / √dₕ'),enc('Encrypted scores','Folded-diagonal S')]),
    operation('scores-to-mpc',3,'boundary','FHE → MPC: attention scores','Encrypted attention scores cross a protected conversion boundary.','Complex CKKS conversion carries real and imaginary channels into additive shares. Neither party receives the unmasked scores.','Conversion pair 1 of 4: the attention nonlinearity.',[enc('Encrypted scores','Minimal folded-diagonal S'),bridge('FHE → MPC'),shares('Score shares','[S]₀ + [S]₁')]),
    operation('mbmax',3,'mpc','Π_MBMax softmax surrogate','The parties jointly compute the attention-weight surrogate.','A public shift and distilled scale support a fifth-power multiplication chain with three interactive MPC rounds. This replaces exact softmax.','The attention weights P are produced here, before computing PV.',[shares('Score shares','Client + server'),shares('Π_MBMax','(S + c)⁵ / R_d'),shares('Weight shares','Shared attention weights P')]),
    operation('weights-to-fhe',3,'boundary','MPC → FHE: attention weights','The secret-shared weights return to encrypted GPU computation.','The secure conversion produces CKKS ciphertexts in the folded-diagonal layout consumed directly by the value kernel.','This completes the first conversion pair; V remains encrypted on the server.',[shares('Weight shares','Shared P'),bridge('MPC → FHE'),enc('Encrypted weights','Folded-diagonal P')]),
    operation('value',4,'fhe','Attention value + output kernels','The GPU applies attention weights to V, then the output projection.','The value kernel computes O = PV using folded-diagonal P and head-major V. The output projection absorbs layout adaptation in pre-permuted plaintext weights.','This encrypted path follows MBMax; raw scores cannot be used as attention weights.',[enc('Encrypted P, V','Weights + values'),enc('Value / output','PV → OW_O'),enc('Attention output','Encrypted Z')]),
    operation('attention-to-mpc',5,'boundary','FHE → MPC: attention output','The attention output crosses into shares for residual addition and normalization.','The preserved layer input supplies the residual. Conversion protects the attention activations while the parties prepare the first LayerNorm block.','Conversion pair 2 of 4: attention residual + Π_MBLN.',[enc('Attention output','Encrypted Z + residual X'),bridge('FHE → MPC'),shares('Residual shares','Shares of Z + X')]),
    operation('ln1',5,'mpc','Residual + Π_MBLN₁','The parties add the attention residual and apply the LayerNorm surrogate.','Π_MBLN uses row centering and a distilled public affine scale. These are local operations on additive shares, so this protocol adds zero interactive MPC rounds.','The constant-scale surrogate replaces variance-dependent LayerNorm.',[shares('Residual shares','Z + X'),shares('Π_MBLN₁','Center + public affine map'),shares('Normalized state','Shared X̂')]),
    operation('ln1-to-fhe',5,'boundary','MPC → FHE: normalized hidden states','Normalized shares are converted back to CKKS for the feed-forward network.','The minimal boundary representation is chosen for the next plaintext–ciphertext projection kernel.','The first FFN projection now receives encrypted normalized hidden states.',[shares('Normalized shares','Shared X̂'),bridge('MPC → FHE'),enc('FFN input','Encrypted X̂')]),
    operation('ffn1',6,'fhe','Encrypted FFN linear 1','GPU CKKS evaluates the expanding feed-forward projection.','The shared projection kernel computes the first FFN linear transform. The public bias is incorporated before the activation protocol.','GELU belongs between the two FFN linear transforms.',[enc('FFN input','Encrypted X̂'),enc('First FFN projection','H₁ = X̂W₁ + b₁'),enc('Expanded activation','Encrypted H₁')]),
    operation('ffn-to-mpc',7,'boundary','FHE → MPC: GELU input','The expanded FFN activations cross the secure boundary.','The minimal baseline converts the activation tensor. The boundary-cost rule can instead justify converting CKKS-pre-evaluated polynomial candidates.','Conversion pair 3 of 4: Π_GELU. This trace shows the minimal baseline.',[enc('GELU input','Encrypted H₁'),bridge('FHE → MPC'),shares('Activation shares','Shared H₁')]),
    operation('gelu',7,'mpc','Π_GELU activation','The parties securely evaluate the BOLT-style piecewise GELU surrogate.','The baseline protocol uses private comparisons and selection with four interactive MPC rounds. Candidate pre-evaluation is a separate boundary optimization.','The baseline nonlinear round total is 3 MBMax + 4 GELU; both MBLN blocks add zero.',[shares('Activation shares','Shared H₁'),shares('Π_GELU','Private compare + select'),shares('Activated shares','Shared G')]),
    operation('gelu-to-fhe',7,'boundary','MPC → FHE: GELU output','The activation output returns to CKKS ciphertexts.','Secure re-encoding provides the encrypted input for the second FFN projection. No plaintext activation is exposed to the server.','The next stage is FFN linear 2, not the next Transformer layer.',[shares('Activated shares','Shared G'),bridge('MPC → FHE'),enc('Activated tensor','Encrypted G')]),
    operation('ffn2',8,'fhe','Encrypted FFN linear 2','GPU CKKS projects the activated FFN tensor back to the hidden width.','The second linear transform computes H₂ = GW₂ + b₂. The normalized attention state is retained for the FFN residual connection.','The layer still needs its second residual + LayerNorm block.',[enc('Activated tensor','Encrypted G'),enc('Second FFN projection','H₂ = GW₂ + b₂'),enc('FFN output','Encrypted H₂')]),
    operation('output-to-mpc',9,'boundary','FHE → MPC: FFN output','The FFN output crosses to MPC for the final residual and normalization.','The preserved attention state supplies the residual input. Secure conversion produces shares for the second Π_MBLN block.','Conversion pair 4 of 4: FFN residual + Π_MBLN.',[enc('FFN output','Encrypted H₂ + residual X̂'),bridge('FHE → MPC'),shares('Residual shares','Shares of H₂ + X̂')]),
    operation('ln2',9,'mpc','Residual + Π_MBLN₂','The parties add the FFN residual and normalize the layer output.','The second LayerNorm surrogate uses the same local row-centering and public affine operations, with its own model parameters.','Both normalization blocks operate on private shares.',[shares('Residual shares','H₂ + X̂'),shares('Π_MBLN₂','Center + public affine map'),shares('Layer-output shares','Shared X_next')]),
    operation('output-to-fhe',9,'boundary','MPC → FHE: next-layer input','The normalized layer output is re-encoded as encrypted hidden states.','The ciphertexts are prepared for the next layer’s projection interface. The representation stays private across the layer boundary.','This completes the fourth secure conversion pair.',[shares('Layer-output shares','Shared X_next'),bridge('MPC → FHE'),enc('Layer output','Encrypted X_next')]),
    operation('output',9,'fhe','Encrypted layer output','One encoder layer is complete; the next layer consumes its encrypted output.','The QKV → score → MBMax → value/output → MBLN → FFN/GELU → MBLN pattern repeats across the Transformer stack.','Client decryption occurs only after the final prescribed encrypted output.',[enc('Completed layer','Encrypted X_next'),enc('Next layer','Repeat the encoder graph'),enc('Hidden states','Remain encrypted')]),
  ];
  const globalSteps = {
    input: operation('input',-1,'client','Input sequence','The client prepares the tokenized sequence and initial hidden representation.','Input tokens are mapped into embeddings before the first Transformer layer. The sentence below is illustrative, not an inference request.','Plaintext input exists on the client at this stage.',[node('Input tokens','A · great · movie','client'),node('Embedding','Prepare hidden states','client'),node('Initial hidden states','Client-side representation','client')]),
    encrypt: operation('encrypt',-1,'client','Encrypt hidden states','The client encrypts the hidden representation before it leaves the client.','CKKS ciphertexts and required evaluation material are sent to the server. The secret key remains with the client.','The server receives encrypted hidden states, not plaintext input.',[node('Hidden states','Client-side representation','client'),node('CKKS encryption','Client-held secret key','client'),enc('Encrypted hidden states','Input for layer 1')]),
    decrypt: operation('decrypt',-1,'client','Final client-side decryption','After the last layer and any task-specific output processing, the prescribed encrypted output returns to the client.','The client decrypts the returned result to obtain the model output. This illustration does not compute logits or measure an actual prediction.','Only the prescribed result is revealed to the client.',[enc('Final ciphertext','Prescribed model output'),node('Client decryption','Client-held secret key','client'),node('Private result','Output available to client','client')]),
  };
  let mode='single', position='input', layer=1, stepIndex=0, layerCount=12, timer=null;
  function makeButton(text, selected, action) {
    const b=document.createElement('button'); b.type='button'; b.textContent=text;
    b.setAttribute('aria-pressed',String(selected)); b.classList.toggle('is-active',selected);
    b.addEventListener('click',action); return b;
  }
  function stop() { if(timer!==null)window.clearInterval(timer); timer=null; $('traceScene').classList.remove('is-playing'); updateControls(); }
  function selectGlobal(target, targetLayer=layer) { stop(); position=target; layer=targetLayer; stepIndex=0; render(); }
  function selectLayer(target) { stop(); mode=target===1?'single':'multi'; position='layer'; layer=target; stepIndex=0; render(); }
  function renderNavigation() {
    const global=$('globalTrace'); global.replaceChildren();
    global.append(makeButton('Input',position==='input',()=>selectGlobal('input')),makeButton('Encrypt',position==='encrypt',()=>selectGlobal('encrypt')));
    for(let i=1;i<=layerCount;i++) {
      const b=makeButton(`Layer ${i}`,position==='layer'&&layer===i,()=>selectLayer(i)); b.dataset.layer=String(i); global.append(b);
    }
    global.append(makeButton('Decrypt',position==='decrypt',()=>selectGlobal('decrypt')));
    const selected=global.querySelector('.is-active');
    if(selected)global.scrollLeft=Math.max(0,selected.offsetLeft-global.offsetLeft-global.clientWidth/2+selected.offsetWidth/2);
    const phaseNav=$('layerTrace'); phaseNav.replaceChildren(); phaseNav.hidden=mode==='multi';
    phases.forEach((title,i)=>{
      const b=makeButton(`${String(i+1).padStart(2,'0')} ${title}`,position==='layer'&&steps[stepIndex].phase===i,()=>{
        stop(); position='layer'; stepIndex=steps.findIndex(s=>s.phase===i); render();
      });
      b.dataset.domain=['fhe','fhe','fhe','boundary','fhe','mpc','fhe','boundary','fhe','mpc'][i]; phaseNav.append(b);
    });
    document.querySelectorAll('[data-trace-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.traceView===mode)));
  }
  function nodeVisual(s,n,i) {
    let visual='';
    if(n.domain==='boundary')visual='<div class="conversion-visual"><span>Re</span><span>Im</span><b>⇄</b><span>[x]₀</span><span>[x]₁</span></div>';
    else if(n.domain==='mpc')visual='<div class="scene-shares"><span>Client share</span><span>Server share</span></div>';
    else if(n.domain==='fhe'&&i===1)visual='<div class="scene-matrix">'+Array.from({length:9},(_,j)=>`<i style="--cell:${j}"></i>`).join('')+'</div>';
    else if(n.domain==='fhe')visual='<div class="scene-ciphers"><span>c₀</span><span>c₁</span><span>c₂</span></div>';
    else visual=`<span class="scene-symbol">${s.key==='decrypt'?'⌘':s.key==='encrypt'?'◇':'▤'}</span>`;
    return `<div class="scene-node" data-domain="${n.domain}"><span class="node-domain">${domains[n.domain]}</span><strong>${escape(n.title)}</strong>${visual}<small>${escape(n.detail)}</small></div>`;
  }
  function render() {
    renderNavigation();
    const summary=position==='layer'&&mode==='multi';
    const s=position==='layer'?steps[stepIndex]:globalSteps[position];
    $('traceLayerLabel').textContent=position==='layer'?`Layer ${layer} / ${layerCount}`:position==='decrypt'?'Client result':'Client preparation';
    $('traceIndex').textContent=position==='layer'?(summary?'REPEATED ENCODER GRAPH':`PHASE ${steps[stepIndex].phase+1} / 10 · OPERATION ${stepIndex+1} / ${steps.length}`):domains[s.domain];
    $('traceTitle').textContent=summary?`Layer ${layer} / ${layerCount}: fast-forward`:s.title;
    $('traceDescription').textContent=summary?'The same encoder pattern repeats. Hidden states remain encrypted between layers.':s.desc;
    $('traceDetail').textContent=summary?'Layer 1 shows the detailed graph. The remaining layers repeat its four conversion pairs: MBMax, MBLN₁, GELU, and MBLN₂.':s.detail;
    $('traceNote').textContent=summary?'The value kernel runs after MBMax supplies the attention weights.':s.note;
    $('traceNote').dataset.domain=summary?'boundary':s.domain;
    $('traceScene').innerHTML=summary?
      `<div class="trace-scene" data-domain="fhe"><div class="scene-heading"><span>REPEATED ENCODER LAYERS</span><span>FAST-FORWARD »</span></div><div class="repeat-counter"><span>Layer</span><strong>${layer}<small> / ${layerCount}</small></strong><div class="repeat-progress"><i style="width:${layer/layerCount*100}%"></i></div></div><div class="repeat-domain-chips"><span data-domain="fhe">GPU CKKS</span><b>⇄</b><span data-domain="mpc">MPC protocols</span></div><div class="scene-note">Same encoder graph · encrypted output becomes the next layer’s input</div></div>`:
      `<div class="trace-scene" data-domain="${s.domain}"><div class="scene-heading"><span>${domains[s.domain]}</span><span>${position==='layer'?`LAYER ${layer}`:'INFERENCE FLOW'}</span></div><div class="scene-flow">${s.nodes.map((n,i)=>nodeVisual(s,n,i)).join('<div class="scene-arrow" aria-hidden="true">→<i></i></div>')}</div></div>`;
    $('traceScene').setAttribute('aria-label',summary?`Layer ${layer} repeats the ten encoder phases with four conversion pairs.`:`${s.title}. ${s.desc}`);
    $('traceScene').classList.toggle('is-playing',timer!==null);
    $('tracePattern').textContent=position==='layer'?`Layer ${layer}/${layerCount} · 4 conversion pairs · MBMax: 3 rounds · GELU baseline: 4 rounds · MBLN: 0 rounds`:'Input → Embedding / CKKS encryption → Layer 1 → … → Layer L → Client decryption';
    updateControls();
  }
  function updateControls() {
    $('tracePrev').disabled=position==='input'; $('traceNext').disabled=position==='decrypt';
    $('traceNext').textContent=position==='layer'&&mode==='single'&&stepIndex===steps.length-1&&layer<layerCount?'Next layer →':'Next →';
    $('tracePlay').setAttribute('aria-pressed',String(timer!==null));
    $('tracePlay').textContent=timer!==null?'Ⅱ Pause':position==='decrypt'?'▶ Replay inference':position==='layer'&&mode==='single'&&stepIndex===steps.length-1?'▶ Replay layer':mode==='multi'?'▶ Play inference':'▶ Play trace';
  }
  function advance(manual=false) {
    if(position==='input')position='encrypt';
    else if(position==='encrypt'){position='layer';layer=1;stepIndex=0;}
    else if(position==='decrypt')return false;
    else if(mode==='single'&&stepIndex<steps.length-1)stepIndex++;
    else if(layer<layerCount){mode='multi';layer++;stepIndex=0;}
    else position='decrypt';
    render(); return true;
  }
  $('traceNext').addEventListener('click',()=>{stop();advance(true);});
  $('tracePrev').addEventListener('click',()=>{
    stop();
    if(position==='decrypt'){position='layer';layer=layerCount;stepIndex=steps.length-1;}
    else if(position==='layer'&&mode==='single'&&stepIndex>0)stepIndex--;
    else if(position==='layer'&&layer>1){layer--;stepIndex=steps.length-1;}
    else if(position==='layer')position='encrypt';
    else if(position==='encrypt')position='input';
    render();
  });
  function startPlayback() {
    if(timer!==null)window.clearInterval(timer);
    const delay=position==='layer'&&mode==='multi'&&layer>1?1000:3200;
    timer=window.setInterval(()=>{
      if(!advance()){stop();return;}
      if(position==='decrypt'){stop();return;}
      if(position==='layer'&&mode==='multi'&&layer===2){startPlayback();render();}
    },delay);
  }
  $('tracePlay').addEventListener('click',()=>{
    if(timer!==null){stop();return;}
    if(position==='decrypt'){position='input';layer=1;stepIndex=0;}
    else if(position==='layer'&&mode==='single'&&stepIndex===steps.length-1)stepIndex=0;
    startPlayback(); render();
  });
  document.querySelectorAll('[data-trace-view]').forEach(b=>b.addEventListener('click',()=>{stop();mode=b.dataset.traceView;render();}));
  $('traceModel').addEventListener('change',()=>{stop();layerCount=Number($('traceModel').value);layer=Math.min(layer,layerCount);render();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  if(reducedMotion)$('traceScene').classList.add('reduce-motion');
  render();
})();
