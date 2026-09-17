const MODEL_URL="./autovision_v6_2.onnx?v=1";

const IOU=0.45;
const FPS=2;
let CONF=0.25;

const CLASSES=[
  "AIR_FILTER_CLEAN",
  "AIR_FILTER_DIRTY",
  "BRAKE_PAD",
  "SPARK_PLUG"
];

const video=document.getElementById("video");
const overlay=document.getElementById("overlay");
const ctx=overlay.getContext("2d");
const statusEl=document.getElementById("status");
const resultEl=document.getElementById("result");
const startBtn=document.getElementById("startBtn");

let session=null;
let running=false;
let lastRun=0;
let INPUT=640;

let candidate=null;
let stableCount=0;

const REQUIRED_STABLE_FRAMES=3;

ort.env.wasm.wasmPaths=
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";


// --------------------------------------------------
// LOAD AUTOVISION MODEL
// --------------------------------------------------

(async()=>{
 try{

  session=await ort.InferenceSession.create(
   MODEL_URL,
   {
    executionProviders:["wasm"],
    graphOptimizationLevel:"all"
   }
  );

  const name=session.inputNames[0];

  const meta=
   session.inputMetadata &&
   session.inputMetadata[name];

  const dims=
   meta &&
   (meta.dimensions||meta.dims);

  if(
   dims &&
   Number(dims[2])>0 &&
   Number(dims[3])>0
  ){
   INPUT=Number(dims[2]);
  }

  statusEl.textContent=
   `AutoVision AI ready — model input ${INPUT}×${INPUT}`;

  resultEl.textContent=
   "Ready to inspect supported components.";

  startBtn.disabled=false;

 }catch(e){

  console.error(e);

  statusEl.textContent=
   "AutoVision model failed to load";

  resultEl.textContent=
   e?.message||String(e);
 }
})();


// --------------------------------------------------
// START CAMERA
// --------------------------------------------------

startBtn.onclick=async()=>{

 if(running){
  stopCamera();
  return;
 }

 try{

  const stream=
   await navigator.mediaDevices.getUserMedia({
    video:{
     facingMode:{ideal:"environment"},
     width:{ideal:1280},
     height:{ideal:720}
    },
    audio:false
   });

  video.srcObject=stream;

  await video.play();

  running=true;

  startBtn.textContent="Stop Camera";

  candidate=null;
  stableCount=0;

  statusEl.textContent=
   `Scanning — ${INPUT}×${INPUT}`;

  resultEl.textContent=
   "Looking for a supported component…";

  requestAnimationFrame(loop);

 }catch(e){

  statusEl.textContent=
   "Camera permission failed";

  resultEl.textContent=
   e?.message||String(e);
 }
};


// --------------------------------------------------
// STOP CAMERA
// --------------------------------------------------

function stopCamera(){

 running=false;

 if(video.srcObject){
  video.srcObject
   .getTracks()
   .forEach(t=>t.stop());
 }

 video.srcObject=null;

 ctx.clearRect(
  0,
  0,
  overlay.width,
  overlay.height
 );

 startBtn.textContent=
  "Start Camera";

 statusEl.textContent=
  "Camera stopped";

 resultEl.textContent="";

 candidate=null;
 stableCount=0;
}


// --------------------------------------------------
// MAIN DETECTION LOOP
// --------------------------------------------------

async function loop(t){

 if(!running) return;

 if(
  t-lastRun>=1000/FPS &&
  video.readyState>=2
 ){

  lastRun=t;

  try{

   const detections=
    await detect();

   statusEl.textContent=
    `Scanning — ${INPUT}×${INPUT}`;

   drawStable(detections);

  }catch(e){

   console.error(e);

   statusEl.textContent=
    "Detection error";

   resultEl.textContent=
    e?.message||String(e);

   running=false;

   startBtn.textContent=
    "Start Camera";

   if(video.srcObject){
    video.srcObject
     .getTracks()
     .forEach(t=>t.stop());

    video.srcObject=null;
   }

   return;
  }
 }

 requestAnimationFrame(loop);
}


// --------------------------------------------------
// IMAGE PREPROCESSING + MODEL INFERENCE
// --------------------------------------------------

async function detect(){

 const vw=video.videoWidth;
 const vh=video.videoHeight;

 if(!vw||!vh) return [];

 const scale=
  Math.min(INPUT/vw,INPUT/vh);

 const nw=
  Math.round(vw*scale);

 const nh=
  Math.round(vh*scale);

 const px=
  (INPUT-nw)/2;

 const py=
  (INPUT-nh)/2;

 const c=
  document.createElement("canvas");

 c.width=INPUT;
 c.height=INPUT;

 const x=
  c.getContext(
   "2d",
   {willReadFrequently:true}
  );

 x.fillStyle=
  "rgb(114,114,114)";

 x.fillRect(
  0,
  0,
  INPUT,
  INPUT
 );

 x.drawImage(
  video,
  0,
  0,
  vw,
  vh,
  px,
  py,
  nw,
  nh
 );

 const rgba=
  x.getImageData(
   0,
   0,
   INPUT,
   INPUT
  ).data;

 const area=
  INPUT*INPUT;

 const f=
  new Float32Array(
   3*area
  );

 for(let i=0;i<area;i++){

  f[i]=
   rgba[i*4]/255;

  f[area+i]=
   rgba[i*4+1]/255;

  f[2*area+i]=
   rgba[i*4+2]/255;
 }

 const tensor=
  new ort.Tensor(
   "float32",
   f,
   [1,3,INPUT,INPUT]
  );

 const feeds={};

 feeds[
  session.inputNames[0]
 ]=tensor;

 const out=
  await session.run(feeds);

 const output=
  out[
   session.outputNames[0]
  ];

 if(!output)
  throw new Error(
   "Model returned no output tensor"
  );

 return nms(
  decode(
   output,
   vw,
   vh,
   scale,
   px,
   py
  )
 );
}


// --------------------------------------------------
// DECODE YOLO V6.2 OUTPUT
// 4 BOX VALUES + 4 CLASSES = 8 CHANNELS
// --------------------------------------------------

function decode(
 o,
 ow,
 oh,
 scale,
 px,
 py
){

 const d=o.data;
 const s=o.dims;

 let count;
 let major;
 let channels;

 if(s.length===3){

  if(s[1]<=20){

   channels=s[1];
   count=s[2];
   major=true;

  }else if(s[2]<=20){

   channels=s[2];
   count=s[1];
   major=false;
  }
 }

 if(!channels){

  throw new Error(
   "Unexpected model output shape: "+
   JSON.stringify(s)
  );
 }

 if(channels!==8){

  throw new Error(
   `Loaded model output is ${JSON.stringify(s)}. `+
   `AutoVision V6.2 must output [1,8,8400].`
  );
 }

 const get=(ch,i)=>
  major
   ?d[ch*count+i]
   :d[i*channels+ch];

 const a=[];

 for(let i=0;i<count;i++){

  const cx=get(0,i);
  const cy=get(1,i);
  const w=get(2,i);
  const h=get(3,i);

  const scores=[
   get(4,i),
   get(5,i),
   get(6,i),
   get(7,i)
  ];

  let cls=0;

  for(
   let j=1;
   j<scores.length;
   j++
  ){

   if(
    scores[j]>
    scores[cls]
   ){
    cls=j;
   }
  }

  const conf=
   scores[cls];

  if(conf<CONF)
   continue;

  let x1=
   (cx-w/2-px)/scale;

  let y1=
   (cy-h/2-py)/scale;

  let x2=
   (cx+w/2-px)/scale;

  let y2=
   (cy+h/2-py)/scale;

  x1=
   Math.max(
    0,
    Math.min(ow,x1)
   );

  y1=
   Math.max(
    0,
    Math.min(oh,y1)
   );

  x2=
   Math.max(
    0,
    Math.min(ow,x2)
   );

  y2=
   Math.max(
    0,
    Math.min(oh,y2)
   );

  if(
   x2>x1 &&
   y2>y1
  ){

   a.push({
    x1,
    y1,
    x2,
    y2,
    cls,
    conf,
    label:CLASSES[cls]
   });
  }
 }

 return a;
}


// --------------------------------------------------
// NON-MAXIMUM SUPPRESSION
// --------------------------------------------------

function nms(a){

 const keep=[];

 for(
  let cls=0;
  cls<CLASSES.length;
  cls++
 ){

  const q=
   a
    .filter(
     z=>z.cls===cls
    )
    .sort(
     (p,q)=>
      q.conf-p.conf
    );

  while(q.length){

   const b=q.shift();

   keep.push(b);

   for(
    let i=q.length-1;
    i>=0;
    i--
   ){

    if(
     iou(b,q[i])>IOU
    ){
     q.splice(i,1);
    }
   }
  }
 }

 return keep
  .sort(
   (p,q)=>
    q.conf-p.conf
  )
  .slice(0,5);
}


// --------------------------------------------------
// INTERSECTION OVER UNION
// --------------------------------------------------

function iou(a,b){

 const x1=
  Math.max(
   a.x1,
   b.x1
  );

 const y1=
  Math.max(
   a.y1,
   b.y1
  );

 const x2=
  Math.min(
   a.x2,
   b.x2
  );

 const y2=
  Math.min(
   a.y2,
   b.y2
  );

 const inter=
  Math.max(
   0,
   x2-x1
  )*
  Math.max(
   0,
   y2-y1
  );

 const aa=
  (a.x2-a.x1)*
  (a.y2-a.y1);

 const bb=
  (b.x2-b.x1)*
  (b.y2-b.y1);

 return inter/
  Math.max(
   aa+bb-inter,
   1e-6
  );
}


// --------------------------------------------------
// USER-FRIENDLY AUTOVISION LABELS
// --------------------------------------------------

function getDisplayInfo(z){

 switch(z.label){

  case "AIR_FILTER_CLEAN":

   return{
    title:"Engine Air Filter",
    condition:"CLEAN",
    boxLabel:"AIR FILTER — CLEAN"
   };


  case "AIR_FILTER_DIRTY":

   return{
    title:"Engine Air Filter",
    condition:"DIRTY",
    boxLabel:"AIR FILTER — DIRTY"
   };


  case "BRAKE_PAD":

   return{
    title:"Brake Pad",
    condition:"Identified",
    boxLabel:"BRAKE PAD"
   };


  case "SPARK_PLUG":

   return{
    title:"Spark Plug",
    condition:"Identified",
    boxLabel:"SPARK PLUG"
   };


  default:

   return{
    title:"Component",
    condition:"Identified",
    boxLabel:z.label
   };
 }
}


// --------------------------------------------------
// STABLE DETECTION + DISPLAY
// --------------------------------------------------

function drawStable(a){

 const r=
  video.getBoundingClientRect();

 const dpr=
  devicePixelRatio||1;

 overlay.width=
  Math.round(
   r.width*dpr
  );

 overlay.height=
  Math.round(
   r.height*dpr
  );

 ctx.setTransform(
  dpr,
  0,
  0,
  dpr,
  0,
  0
 );

 ctx.clearRect(
  0,
  0,
  r.width,
  r.height
 );


 // NOTHING DETECTED
 if(!a.length){

  candidate=null;
  stableCount=0;

  resultEl.textContent=
   "No supported component detected — move closer if needed.";

  return;
 }


 // USE STRONGEST DETECTION
 // FOR STABILITY CHECK
 const z=a[0];


 if(
  candidate &&
  candidate.cls===z.cls &&
  iou(candidate,z)>=0.30
 ){

  stableCount++;
  candidate=z;

 }else{

  candidate=z;
  stableCount=1;
 }


 if(
  stableCount<
  REQUIRED_STABLE_FRAMES
 ){

  resultEl.textContent=
   `Checking… ${stableCount}/${REQUIRED_STABLE_FRAMES}`;

  return;
 }


 const info=
  getDisplayInfo(z);


 // ------------------------------------------------
 // RESULT TEXT
 // ------------------------------------------------

 if(
  z.label==="AIR_FILTER_CLEAN" ||
  z.label==="AIR_FILTER_DIRTY"
 ){

  resultEl.textContent=
   `${info.title} — ${info.condition} — `+
   `${(z.conf*100).toFixed(1)}% confidence`;

 }else{

  resultEl.textContent=
   `${info.title} — ${info.condition} — `+
   `${(z.conf*100).toFixed(1)}% confidence`;
 }


 // ------------------------------------------------
 // CAMERA DISPLAY SCALING
 // ------------------------------------------------

 const vw=
  video.videoWidth;

 const vh=
  video.videoHeight;

 const sc=
  Math.max(
   r.width/vw,
   r.height/vh
  );

 const ox=
  (r.width-vw*sc)/2;

 const oy=
  (r.height-vh*sc)/2;


 // ------------------------------------------------
 // DRAW ALL DETECTED COMPONENTS
 // ------------------------------------------------

 for(const det of a){

  const detInfo=
   getDisplayInfo(det);

  const x=
   det.x1*sc+ox;

  const y=
   det.y1*sc+oy;

  const w=
   (det.x2-det.x1)*sc;

  const h=
   (det.y2-det.y1)*sc;

  const label=
   `${detInfo.boxLabel} `+
   `${(det.conf*100).toFixed(0)}%`;


  ctx.strokeStyle=
   "#00ff66";

  ctx.lineWidth=4;

  ctx.strokeRect(
   x,
   y,
   w,
   h
  );


  ctx.font=
   "bold 18px Arial";

  const tw=
   ctx.measureText(
    label
   ).width;

  ctx.fillStyle=
   "rgba(0,0,0,.75)";

  ctx.fillRect(
   x,
   Math.max(0,y-28),
   tw+14,
   28
  );

  ctx.fillStyle=
   "#fff";

  ctx.fillText(
   label,
   x+7,
   Math.max(20,y-7)
  );
 }
}
