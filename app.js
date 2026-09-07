const MODEL_URL="./best.onnx";
const INPUT=640, CONF=0.05, IOU=0.45, FPS=2;
const CLASSES=["CLEAN","DIRTY"];

const video=document.getElementById("video");
const overlay=document.getElementById("overlay");
const ctx=overlay.getContext("2d");
const statusEl=document.getElementById("status");
const resultEl=document.getElementById("result");
const startBtn=document.getElementById("startBtn");

let session=null,running=false,lastRun=0;

ort.env.wasm.wasmPaths="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

(async()=>{
  try{
    session=await ort.InferenceSession.create(MODEL_URL,{
      executionProviders:["wasm"],
      graphOptimizationLevel:"all"
    });
    statusEl.textContent="AI ready";
    startBtn.disabled=false;
  }catch(e){
    console.error(e);
    statusEl.textContent="Model failed to load";
    resultEl.textContent=(e&&e.message)?e.message:String(e);
  }
})();

startBtn.onclick=async()=>{
  if(running){stopCamera();return;}
  try{
    const stream=await navigator.mediaDevices.getUserMedia({
      video:{facingMode:{ideal:"environment"},width:{ideal:1280},height:{ideal:720}},
      audio:false
    });
    video.srcObject=stream;
    await video.play();
    running=true;
    startBtn.textContent="Stop Camera";
    statusEl.textContent="Scanning…";
    resultEl.textContent="Waiting for first inference…";
    requestAnimationFrame(loop);
  }catch(e){
    statusEl.textContent="Camera permission failed";
    resultEl.textContent=(e&&e.message)?e.message:String(e);
  }
};

function stopCamera(){
  running=false;
  if(video.srcObject) video.srcObject.getTracks().forEach(t=>t.stop());
  video.srcObject=null;
  ctx.clearRect(0,0,overlay.width,overlay.height);
  startBtn.textContent="Start Camera";
  statusEl.textContent="Camera stopped";
  resultEl.textContent="";
}

async function loop(t){
  if(!running)return;
  if(t-lastRun>=1000/FPS && video.readyState>=2){
    lastRun=t;
    try{
      const detections=await detect();
      statusEl.textContent="Scanning…";
      draw(detections);
    }catch(e){
      console.error(e);
      statusEl.textContent="Detection error";
      resultEl.textContent=(e&&e.message)?e.message:String(e);
      running=false;
      startBtn.textContent="Start Camera";
      if(video.srcObject){video.srcObject.getTracks().forEach(t=>t.stop());video.srcObject=null;}
      return;
    }
  }
  requestAnimationFrame(loop);
}

async function detect(){
  const vw=video.videoWidth,vh=video.videoHeight;
  if(!vw||!vh)return[];

  const scale=Math.min(INPUT/vw,INPUT/vh);
  const nw=Math.round(vw*scale),nh=Math.round(vh*scale);
  const px=(INPUT-nw)/2,py=(INPUT-nh)/2;

  const c=document.createElement("canvas");
  c.width=c.height=INPUT;
  const x=c.getContext("2d",{willReadFrequently:true});
  x.fillStyle="rgb(114,114,114)";
  x.fillRect(0,0,INPUT,INPUT);
  x.drawImage(video,0,0,vw,vh,px,py,nw,nh);

  const rgba=x.getImageData(0,0,INPUT,INPUT).data;
  const area=INPUT*INPUT;
  const f=new Float32Array(3*area);

  for(let i=0;i<area;i++){
    f[i]=rgba[i*4]/255;
    f[area+i]=rgba[i*4+1]/255;
    f[2*area+i]=rgba[i*4+2]/255;
  }

  const tensor=new ort.Tensor("float32",f,[1,3,INPUT,INPUT]);
  const feeds={};
  feeds[session.inputNames[0]]=tensor;
  const out=await session.run(feeds);
  const output=out[session.outputNames[0]];
  if(!output) throw new Error("Model returned no output tensor");
  return nms(decode(output,vw,vh,scale,px,py));
}

function decode(o,ow,oh,scale,px,py){
  const d=o.data,s=o.dims;
  let count,major;
  if(s.length===3&&s[1]===6){count=s[2];major=true;}
  else if(s.length===3&&s[2]===6){count=s[1];major=false;}
  else throw new Error("Unexpected model output shape: "+JSON.stringify(s));

  const get=(ch,i)=>major?d[ch*count+i]:d[i*6+ch];
  const a=[];

  for(let i=0;i<count;i++){
    const cx=get(0,i),cy=get(1,i),w=get(2,i),h=get(3,i);
    const c0=get(4,i),c1=get(5,i);
    const cls=c1>c0?1:0,conf=Math.max(c0,c1);
    if(conf<CONF)continue;

    let x1=(cx-w/2-px)/scale,y1=(cy-h/2-py)/scale;
    let x2=(cx+w/2-px)/scale,y2=(cy+h/2-py)/scale;
    x1=Math.max(0,Math.min(ow,x1));y1=Math.max(0,Math.min(oh,y1));
    x2=Math.max(0,Math.min(ow,x2));y2=Math.max(0,Math.min(oh,y2));
    if(x2>x1&&y2>y1)a.push({x1,y1,x2,y2,cls,conf,label:CLASSES[cls]});
  }
  return a;
}

function nms(a){
  const keep=[];
  for(let cls=0;cls<2;cls++){
    const q=a.filter(z=>z.cls===cls).sort((p,q)=>q.conf-p.conf);
    while(q.length){
      const b=q.shift();keep.push(b);
      for(let i=q.length-1;i>=0;i--)if(iou(b,q[i])>IOU)q.splice(i,1);
    }
  }
  return keep.sort((p,q)=>q.conf-p.conf).slice(0,10);
}

function iou(a,b){
  const x1=Math.max(a.x1,b.x1),y1=Math.max(a.y1,b.y1);
  const x2=Math.min(a.x2,b.x2),y2=Math.min(a.y2,b.y2);
  const inter=Math.max(0,x2-x1)*Math.max(0,y2-y1);
  const aa=(a.x2-a.x1)*(a.y2-a.y1),bb=(b.x2-b.x1)*(b.y2-b.y1);
  return inter/Math.max(aa+bb-inter,1e-6);
}

function draw(a){
  const r=video.getBoundingClientRect(),dpr=devicePixelRatio||1;
  overlay.width=Math.round(r.width*dpr);
  overlay.height=Math.round(r.height*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,r.width,r.height);

  const vw=video.videoWidth,vh=video.videoHeight;
  const sc=Math.max(r.width/vw,r.height/vh);
  const ox=(r.width-vw*sc)/2,oy=(r.height-vh*sc)/2;

  if(!a.length){resultEl.textContent="No engine air filter detected";return;}
  resultEl.textContent=`${a[0].label} — ${(a[0].conf*100).toFixed(1)}%`;

  for(const z of a){
    const x=z.x1*sc+ox,y=z.y1*sc+oy,w=(z.x2-z.x1)*sc,h=(z.y2-z.y1)*sc;
    const label=`${z.label} ${(z.conf*100).toFixed(0)}%`;
    ctx.strokeStyle="#00ff66";ctx.lineWidth=4;ctx.strokeRect(x,y,w,h);
    ctx.font="bold 18px Arial";
    const tw=ctx.measureText(label).width;
    ctx.fillStyle="rgba(0,0,0,.75)";
    ctx.fillRect(x,Math.max(0,y-28),tw+14,28);
    ctx.fillStyle="#fff";
    ctx.fillText(label,x+7,Math.max(20,y-7));
  }
}
