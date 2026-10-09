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
let stableFinding=null;
let capturedFinding=null;
const STORAGE_KEY="autovision_v64_inspection_records";
const SESSION_KEY="autovision_v65_sessions";
let sessions=[];
let activeId=null;
function id(){return (crypto.randomUUID?crypto.randomUUID():Date.now().toString(36)+Math.random().toString(36).slice(2));}
function activeSession(){return sessions.find(s=>s.id===activeId);}
function storeSessions(nextSessions,nextRecords,nextId){
 try{
  localStorage.setItem(SESSION_KEY,JSON.stringify({sessions:nextSessions,records:nextRecords,activeId:nextId}));
  return true;
 }catch(e){alert("Could not save inspection. Browser storage may be full or unavailable.");return false;}
}
let storageAvailable=true;
let inspectionRecords=[];
function readSavedRecords(){
 try{
  const raw=localStorage.getItem(STORAGE_KEY);
  if(!raw)return [];
  const parsed=JSON.parse(raw);
  if(!Array.isArray(parsed))throw new Error("Invalid stored records");
  return parsed.filter(r=>r && typeof r==="object" &&
   typeof r.component==="string" && typeof r.technicianAssessment==="string");
 }catch(error){
  console.warn("AutoVision storage unavailable or invalid",error);
  storageAvailable=false;
  return [];
 }
}
function persistRecords(next){
 try{
  localStorage.setItem(STORAGE_KEY,JSON.stringify(next));
  storageAvailable=true;
  return true;
 }catch(error){
  console.error("Cannot save inspection records",error);
  alert("Could not save records on this device. Check browser storage settings or available space. Your current records have not been changed.");
  return false;
 }
}
inspectionRecords=readSavedRecords();
try{
 const saved=localStorage.getItem(SESSION_KEY);
 if(saved){
  const state=JSON.parse(saved);
  if(!Array.isArray(state.sessions)||!Array.isArray(state.records))throw Error("Invalid session data");
  sessions=state.sessions;inspectionRecords=state.records;
  activeId=sessions.some(x=>x.id===state.activeId)?state.activeId:(sessions[0]?.id||null);
 }else if(inspectionRecords.length){
  const old={id:id(),registration:"LEGACY",makeModel:"Previous V6.4 findings",mileage:"",technician:"",createdAt:new Date().toISOString()};
  const migrated=inspectionRecords.map(r=>({...r,sessionId:old.id}));
  if(storeSessions([old],migrated,old.id)){sessions=[old];inspectionRecords=migrated;activeId=old.id;}
 }
}catch(e){console.error("Cannot load vehicle sessions",e);alert("Saved sessions could not be loaded. Previous V6.4 records remain in browser storage.");}
const captureBtn=document.getElementById("captureBtn");
const inspectionPanel=document.getElementById("inspectionPanel");
const inspectionSummary=document.getElementById("inspectionSummary");
const decisionEl=document.getElementById("decision");
const assessmentEl=document.getElementById("assessment");
const remarksEl=document.getElementById("remarks");
const recordList=document.getElementById("recordList");
const recordCount=document.getElementById("recordCount");


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
  stableFinding=null;
  captureBtn.disabled=true;

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
 stableFinding=null;
 captureBtn.disabled=true;
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
  stableFinding=null;
  captureBtn.disabled=true;

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
  stableFinding=null;
  captureBtn.disabled=true;

  resultEl.textContent=
   `Checking… ${stableCount}/${REQUIRED_STABLE_FRAMES}`;

  return;
 }


 const info=
  getDisplayInfo(z);
 stableFinding={
  label:z.label,
  component:info.title,
  aiSuggestion:(z.label==="AIR_FILTER_CLEAN"||z.label==="AIR_FILTER_DIRTY")?info.condition:"Identification only",
  confidence:z.conf
 };
 captureBtn.disabled=!activeSession();


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


// --------------------------------------------------
// V6.3 TECHNICIAN REVIEW AND SESSION-ONLY RECORDS
// --------------------------------------------------
function assessmentOptions(label){
 if(label==="AIR_FILTER_CLEAN"||label==="AIR_FILTER_DIRTY")
  return ["CLEAN","DIRTY","Unable to determine"];
 if(label==="BRAKE_PAD")
  return ["Serviceable (visually)","Worn / suspect","Requires measurement","Unable to determine"];
 return ["Serviceable (visually)","Worn / suspect","Requires further testing","Unable to determine"];
}
function setAssessmentOptions(finding){
 assessmentEl.replaceChildren();
 for(const value of assessmentOptions(finding.label)){
  const option=document.createElement("option");
  option.value=value;option.textContent=value;
  assessmentEl.appendChild(option);
 }
 if(finding.label==="AIR_FILTER_CLEAN"||finding.label==="AIR_FILTER_DIRTY")
  assessmentEl.value=finding.aiSuggestion;
 else assessmentEl.value="Unable to determine";
}
captureBtn.onclick=()=>{
 if(!stableFinding||!activeSession()){alert("Create or select a vehicle inspection first.");return;}
 capturedFinding={...stableFinding, capturedAt:new Date().toISOString()};
 inspectionSummary.textContent=
  capturedFinding.component+" — AI: "+capturedFinding.aiSuggestion+
  " ("+(capturedFinding.confidence*100).toFixed(1)+"% confidence)";
 decisionEl.value="confirm";
 remarksEl.value="";
 setAssessmentOptions(capturedFinding);
 updateDecision();
 inspectionPanel.hidden=false;
 inspectionPanel.scrollIntoView({behavior:"smooth",block:"nearest"});
};
function updateDecision(){
 const isAir=capturedFinding && capturedFinding.label.startsWith("AIR_FILTER_");
 const isOverride=decisionEl.value==="override";
 assessmentEl.disabled=decisionEl.value==="review" || (!isOverride && isAir);
 if(decisionEl.value==="review") assessmentEl.value="Unable to determine";
 if(decisionEl.value==="confirm" && isAir) assessmentEl.value=capturedFinding.aiSuggestion;
}
decisionEl.onchange=()=>{
 if(capturedFinding)setAssessmentOptions(capturedFinding);
 updateDecision();
};
document.getElementById("cancelFindingBtn").onclick=()=>{
 capturedFinding=null;inspectionPanel.hidden=true;
};
document.getElementById("saveFindingBtn").onclick=()=>{
 if(!capturedFinding)return;
 const isAir=capturedFinding.label.startsWith("AIR_FILTER_");
 const decision=decisionEl.value;
 const assessment=decision==="review"?"Further inspection required":
  (decision==="confirm"&&!isAir?"Component identification confirmed":assessmentEl.value);
 if(decision==="override" && assessment==="Unable to determine" && !remarksEl.value.trim()){
  alert("Add a remark explaining why the AI finding was overridden.");return;
 }
 if(decision==="override" && isAir && assessment===capturedFinding.aiSuggestion){
  alert("Choose a different assessment for an override.");return;
 }
 const next=[...inspectionRecords,{
  sessionId:activeId,
  ...capturedFinding,
  decision,
  technicianAssessment:assessment,
  remarks:remarksEl.value.trim(),
  savedAt:new Date().toISOString()
 }];
 if(!storeSessions(sessions,next,activeId))return;
 inspectionRecords=next;
 capturedFinding=null;inspectionPanel.hidden=true;
 renderRecords();
};
const CHECKLIST_COMPONENTS=["Engine Air Filter","Brake Pad","Spark Plug"];
function checklistFor(sessionId){
 return CHECKLIST_COMPONENTS.map(component=>{
  const records=inspectionRecords.filter(r=>r.sessionId===sessionId&&r.component===component);
  const last=records[records.length-1];
  return {component,status:!last?"Pending":last.decision==="review"?"Requires Further Inspection":"Completed",basis:!last?"No saved finding":component==="Engine Air Filter"?"Technician: "+last.technicianAssessment:"Identification reviewed; condition not assessed"};
 });
}
function renderChecklist(){
 const container=document.getElementById("checklistItems");
 const progress=document.getElementById("checklistProgress");
 container.replaceChildren();
 if(!activeSession()){progress.textContent="Select a vehicle to view progress.";return;}
 const items=checklistFor(activeId);
 progress.textContent=items.filter(x=>x.status==="Completed").length+" of 3 component reviews completed";
 for(const item of items){
  const row=document.createElement("div");row.className="checklistItem";
  const left=document.createElement("div");
  const title=document.createElement("strong");title.textContent=item.component;
  const note=document.createElement("small");note.textContent=item.basis;
  left.append(title,note);
  const status=document.createElement("span");status.className="checklistStatus "+(item.status==="Completed"?"completed":item.status==="Pending"?"":"review");
  status.textContent=item.status;row.append(left,status);container.appendChild(row);
 }
}
function renderRecords(){
 recordList.replaceChildren();
 const selectedRecords=inspectionRecords.filter(r=>r.sessionId===activeId);
 recordCount.textContent=String(selectedRecords.length);
 renderChecklist();
 document.getElementById("reportBtn").disabled=!activeSession() || selectedRecords.length===0;
 const storageStatus=document.getElementById("storageStatus");
 storageStatus.textContent=storageAvailable?
  "Saved on this device only. Records remain after refresh but may be lost if browser data is cleared.":
  "Browser storage is unavailable. Do not rely on records being retained.";
 for(const r of selectedRecords){
  const li=document.createElement("li");
  li.textContent=r.component+" | AI: "+r.aiSuggestion+
   " | Technician: "+r.technicianAssessment+
   " | "+r.decision+
   (r.remarks?" | "+r.remarks:"");
  recordList.appendChild(li);
 }
}
document.getElementById("clearRecordsBtn").onclick=()=>{
 if(activeSession() && inspectionRecords.some(r=>r.sessionId===activeId) && confirm("Clear findings for this vehicle only?")){
  const next=inspectionRecords.filter(r=>r.sessionId!==activeId);
  if(!storeSessions(sessions,next,activeId))return;
  inspectionRecords=next;renderRecords();
 }
};
renderRecords();

const vehicleSelect=document.getElementById("vehicleSelect");
const vehicleForm=document.getElementById("vehicleForm");
const activeVehicleCard=document.getElementById("activeVehicleCard");
const historyPanel=document.getElementById("historyPanel");
const historyList=document.getElementById("historyList");
const historyBtn=document.getElementById("historyBtn");
function countFindings(sessionId){
 return inspectionRecords.filter(r=>r.sessionId===sessionId).length;
}
function chooseVehicle(nextId){
 if(capturedFinding){alert("Finish or cancel the current review first.");return false;}
 if(!sessions.some(v=>v.id===nextId))return false;
 if(!storeSessions(sessions,inspectionRecords,nextId))return false;
 activeId=nextId;
 renderVehicles();
 return true;
}
historyBtn.onclick=()=>{
 historyPanel.hidden=!historyPanel.hidden;
 historyBtn.setAttribute("aria-expanded",String(!historyPanel.hidden));
 historyBtn.textContent=historyPanel.hidden?"View Previous Inspections":"Hide Inspection History";
};
function renderVehicles(){
 vehicleSelect.replaceChildren();
 historyList.replaceChildren();
 if(!sessions.length){
  const o=document.createElement("option");o.value="";o.textContent="No inspections yet";vehicleSelect.appendChild(o);
 }
 for(const v of sessions){
  const o=document.createElement("option");o.value=v.id;
  o.textContent=v.registration+" — "+v.makeModel+" ("+countFindings(v.id)+" findings)";
  vehicleSelect.appendChild(o);
 }
 vehicleSelect.value=activeId||"";
 const v=activeSession();
 activeVehicleCard.replaceChildren();
 if(v){
  const status=document.createElement("div");status.className="status";status.textContent="● ACTIVE INSPECTION";
  const reg=document.createElement("strong");reg.textContent=v.registration;
  const detail=document.createElement("div");detail.className="muted";
  detail.textContent=v.makeModel+" · "+(v.mileage||"Mileage not entered")+" km · "+countFindings(v.id)+" saved finding(s)";
  activeVehicleCard.append(status,reg,detail);
 }else{
  activeVehicleCard.textContent="No active vehicle. Tap New Vehicle Inspection to begin.";
 }
 const previous=sessions.filter(x=>x.id!==activeId).slice().reverse();
 if(!previous.length){
  const p=document.createElement("div");p.className="vehicleDetails";
  p.textContent="No other inspections saved yet.";
  historyList.appendChild(p);
 }
 for(const old of previous){
  const btn=document.createElement("button");
  btn.type="button";btn.className="historyEntry";
  const title=document.createElement("span");title.textContent=old.registration+" — "+old.makeModel;
  const details=document.createElement("span");details.className="muted";
  details.textContent=countFindings(old.id)+" finding(s) · "+new Date(old.createdAt).toLocaleDateString()+" · Tap to open";
  btn.append(title,details);
  btn.onclick=()=>{if(chooseVehicle(old.id)){historyPanel.hidden=true;historyBtn.textContent="View Previous Inspections";historyBtn.setAttribute("aria-expanded","false");}};
  historyList.appendChild(btn);
 }
 document.getElementById("vehicleInfo").textContent=v?
  "Technician: "+(v.technician||"Not recorded")+" | Date: "+new Date(v.createdAt).toLocaleString()+" | Inspection ID: "+v.id:
  "Vehicle records are stored on this device.";
 captureBtn.disabled=!v||!stableFinding;
 renderRecords();
}
document.getElementById("newVehicleBtn").onclick=()=>{vehicleForm.hidden=!vehicleForm.hidden;};
document.getElementById("cancelVehicleBtn").onclick=()=>{vehicleForm.reset();vehicleForm.hidden=true;};
vehicleForm.onsubmit=e=>{
 e.preventDefault();
 if(capturedFinding){alert("Finish or cancel the current review first.");return;}
 const registration=document.getElementById("vehicleReg").value.trim().toUpperCase();
 const makeModel=document.getElementById("vehicleModel").value.trim();
 const mileage=document.getElementById("vehicleMileage").value.trim();
 const technician=document.getElementById("technicianName").value.trim();
 if(!registration||!makeModel||!technician)return;
 const v={id:id(),registration,makeModel,mileage,technician,createdAt:new Date().toISOString()};
 const next=[...sessions,v];
 if(!storeSessions(next,inspectionRecords,v.id))return;
 sessions=next;activeId=v.id;vehicleForm.reset();vehicleForm.hidden=true;renderVehicles();
};
vehicleSelect.onchange=()=>{
 const nextId=vehicleSelect.value;
 if(!chooseVehicle(nextId))vehicleSelect.value=activeId||"";
};
renderVehicles();

/* V6.6 — build a printable A4 report from the selected vehicle only.
   Browser print/share allows Save as PDF without uploading inspection data. */
function reportCell(row,value){
 const td=document.createElement("td");
 td.textContent=String(value??"");
 row.appendChild(td);
}
function reportField(parent,label,value){
 const div=document.createElement("div");
 const title=document.createElement("b");title.textContent=label;
 const content=document.createElement("span");content.textContent=String(value??"Not recorded");
 div.append(title,content);parent.appendChild(div);
}
document.getElementById("reportBtn").onclick=()=>{
 const vehicle=activeSession();
 if(!vehicle){alert("Select a vehicle inspection first.");return;}
 const findings=inspectionRecords.filter(r=>r.sessionId===vehicle.id);
 if(!findings.length){alert("Save at least one finding before generating a report.");return;}
 const metadata=document.getElementById("reportMetadata");
 const rows=document.getElementById("reportRows");
 const checklistRows=document.getElementById("reportChecklistRows");
 metadata.replaceChildren();rows.replaceChildren();checklistRows.replaceChildren();
 reportField(metadata,"Vehicle Registration",vehicle.registration);
 reportField(metadata,"Make / Model",vehicle.makeModel);
 reportField(metadata,"Mileage",vehicle.mileage?vehicle.mileage+" km":"Not recorded");
 reportField(metadata,"Technician",vehicle.technician||"Not recorded");
 reportField(metadata,"Inspection Date",new Date(vehicle.createdAt).toLocaleString());
 reportField(metadata,"Inspection ID",vehicle.id);
 reportField(metadata,"Report Generated",new Date().toLocaleString());
 reportField(metadata,"Number of Findings",findings.length);
 for(const item of checklistFor(vehicle.id)){
  const tr=document.createElement("tr");
  reportCell(tr,item.component);reportCell(tr,item.status);reportCell(tr,item.basis);
  checklistRows.appendChild(tr);
 }
 for(const finding of findings){
  const tr=document.createElement("tr");
  reportCell(tr,finding.component);
  reportCell(tr,(finding.aiSuggestion||"Not recorded")+(typeof finding.confidence==="number"?" ("+(finding.confidence*100).toFixed(1)+"% AI confidence)":""));
  reportCell(tr,finding.technicianAssessment);
  reportCell(tr,(finding.decision||"Not recorded")+(finding.remarks?" — "+finding.remarks:"")+(finding.savedAt?"\nSaved: "+new Date(finding.savedAt).toLocaleString():""));
  rows.appendChild(tr);
 }
 window.print();
};
