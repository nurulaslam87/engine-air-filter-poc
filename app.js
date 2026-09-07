const MODEL_URL="./detector_v4.onnx?v=1";
const IOU=0.45, FPS=2;
const CLASSES=["CLEAN","DIRTY"];
let CONF=0.70;

const video=document.getElementById("video");
const overlay=document.getElementById("overlay");
const ctx=overlay.getContext("2d");
const statusEl=document.getElementById("status");
const resultEl=document.getElementById("result");
const startBtn=document.getElementById("startBtn");

let session=null,running=false,lastRun=0,INPUT=640;
let candidate=null, stableCount=0;
const REQUIRED_STABLE_FRAMES=3;
ort.env.wasm.wasmPaths="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

(async()=>{
 try{
  session=await ort.InferenceSession.create(MODEL_URL,{executionProviders:["wasm"],graphOptimizationLevel:"all"});
  const name=session.inputNames[0];
  const meta=session.inputMetadata && session.inputMetadata[name];
  const dims=meta && (meta.dimensions||meta.dims);
  if(dims && Number(dims[2])>0 && Number(dims[3])>0) INPUT=Number(dims[2]);
  statusEl.textContent=`AI ready — model input ${INPUT}×${INPUT}`;
  startBtn.disabled=false;
 }catch(e){console.error(e);statusEl.textContent="Model failed to load";resultEl.textContent=e?.message||String(e);}
})();

startBtn.onclick=async()=>{
 if(running){stopCamera();return;}
 try{
  const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:"environment"},width:{ideal:1280},height:{ideal:720}},audio:false});
  video.srcObject=stream;await video.play();running=true;startBtn.textContent="Stop Camera";
  candidate=null;stableCount=0;
  statusEl.textContent=`Scanning — ${INPUT}×${INPUT}`;resultEl.textContent="Looking for an engine air filter…";requestAnimationFrame(loop);
 }catch(e){statusEl.textContent="Camera permission failed";resultEl.textContent=e?.message||String(e);}
};
function stopCamera(){running=false;if(video.srcObject)video.srcObject.getTracks().forEach(t=>t.stop());video.srcObject=null;ctx.clearRect(0,0,overlay.width,overlay.height);startBtn.textContent="Start Camera";statusEl.textContent="Camera stopped";resultEl.textContent="";candidate=null;stableCount=0;}
async function loop(t){if(!running)return;if(t-lastRun>=1000/FPS&&video.readyState>=2){lastRun=t;try{const detections=await detect();statusEl.textContent=`Scanning — ${INPUT}×${INPUT}`;drawStable(detections);}catch(e){console.error(e);statusEl.textContent="Detection error";resultEl.textContent=e?.message||String(e);running=false;startBtn.textContent="Start Camera";if(video.srcObject){video.srcObject.getTracks().forEach(t=>t.stop());video.srcObject=null;}return;}}requestAnimationFrame(loop);}

async function detect(){
 const vw=video.videoWidth,vh=video.videoHeight;if(!vw||!vh)return[];
 const scale=Math.min(INPUT/vw,INPUT/vh),nw=Math.round(vw*scale),nh=Math.round(vh*scale),px=(INPUT-nw)/2,py=(INPUT-nh)/2;
 const c=document.createElement("canvas");c.width=c.height=INPUT;const x=c.getContext("2d",{willReadFrequently:true});x.fillStyle="rgb(114,114,114)";x.fillRect(0,0,INPUT,INPUT);x.drawImage(video,0,0,vw,vh,px,py,nw,nh);
 const rgba=x.getImageData(0,0,INPUT,INPUT).data,area=INPUT*INPUT,f=new Float32Array(3*area);
 for(let i=0;i<area;i++){f[i]=rgba[i*4]/255;f[area+i]=rgba[i*4+1]/255;f[2*area+i]=rgba[i*4+2]/255;}
 const tensor=new ort.Tensor("float32",f,[1,3,INPUT,INPUT]),feeds={};feeds[session.inputNames[0]]=tensor;const out=await session.run(feeds),output=out[session.outputNames[0]];if(!output)throw new Error("Model returned no output tensor");
 return nms(decode(output,vw,vh,scale,px,py));
}
function decode(o,ow,oh,scale,px,py){
 const d=o.data,s=o.dims;let count,major,channels;
 if(s.length===3){if(s[1]<=20){channels=s[1];count=s[2];major=true;}else if(s[2]<=20){channels=s[2];count=s[1];major=false;}}
 if(!channels)throw new Error("Unexpected model output shape: "+JSON.stringify(s));
 if(channels!==6)throw new Error(`Loaded model output is ${JSON.stringify(s)}. Detector must output [1,6,8400].`);
 const get=(ch,i)=>major?d[ch*count+i]:d[i*channels+ch],a=[];
 for(let i=0;i<count;i++){const cx=get(0,i),cy=get(1,i),w=get(2,i),h=get(3,i),c0=get(4,i),c1=get(5,i),cls=c1>c0?1:0,conf=Math.max(c0,c1);if(conf<CONF)continue;let x1=(cx-w/2-px)/scale,y1=(cy-h/2-py)/scale,x2=(cx+w/2-px)/scale,y2=(cy+h/2-py)/scale;x1=Math.max(0,Math.min(ow,x1));y1=Math.max(0,Math.min(oh,y1));x2=Math.max(0,Math.min(ow,x2));y2=Math.max(0,Math.min(oh,y2));if(x2>x1&&y2>y1)a.push({x1,y1,x2,y2,cls,conf,label:CLASSES[cls]});}return a;
}
function nms(a){const keep=[];for(let cls=0;cls<2;cls++){const q=a.filter(z=>z.cls===cls).sort((p,q)=>q.conf-p.conf);while(q.length){const b=q.shift();keep.push(b);for(let i=q.length-1;i>=0;i--)if(iou(b,q[i])>IOU)q.splice(i,1);}}return keep.sort((p,q)=>q.conf-p.conf).slice(0,1);}
function iou(a,b){const x1=Math.max(a.x1,b.x1),y1=Math.max(a.y1,b.y1),x2=Math.min(a.x2,b.x2),y2=Math.min(a.y2,b.y2),inter=Math.max(0,x2-x1)*Math.max(0,y2-y1),aa=(a.x2-a.x1)*(a.y2-a.y1),bb=(b.x2-b.x1)*(b.y2-b.y1);return inter/Math.max(aa+bb-inter,1e-6);}

function drawStable(a){
 const r=video.getBoundingClientRect(),dpr=devicePixelRatio||1;overlay.width=Math.round(r.width*dpr);overlay.height=Math.round(r.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,r.width,r.height);
 if(!a.length){candidate=null;stableCount=0;resultEl.textContent="No engine air filter detected";return;}
 const z=a[0];
 if(candidate && candidate.cls===z.cls && iou(candidate,z)>=0.30){stableCount++;candidate=z;}else{candidate=z;stableCount=1;}
 if(stableCount<REQUIRED_STABLE_FRAMES){resultEl.textContent=`Checking… ${stableCount}/${REQUIRED_STABLE_FRAMES}`;return;}
 const vw=video.videoWidth,vh=video.videoHeight,sc=Math.max(r.width/vw,r.height/vh),ox=(r.width-vw*sc)/2,oy=(r.height-vh*sc)/2;
 resultEl.textContent=`${z.label} — ${(z.conf*100).toFixed(1)}%`;
 const x=z.x1*sc+ox,y=z.y1*sc+oy,w=(z.x2-z.x1)*sc,h=(z.y2-z.y1)*sc,label=`${z.label} ${(z.conf*100).toFixed(0)}%`;
 ctx.strokeStyle="#00ff66";ctx.lineWidth=4;ctx.strokeRect(x,y,w,h);ctx.font="bold 18px Arial";const tw=ctx.measureText(label).width;ctx.fillStyle="rgba(0,0,0,.75)";ctx.fillRect(x,Math.max(0,y-28),tw+14,28);ctx.fillStyle="#fff";ctx.fillText(label,x+7,Math.max(20,y-7));
}
