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


if(typeof ort!=="undefined"){
 ort.env.wasm.wasmPaths="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
}else{
 statusEl.textContent="AI engine unavailable — check internet connection or content blocking.";
 resultEl.textContent="Manual checklist and AR guide remain available.";
}


// --------------------------------------------------
// LOAD AUTOVISION MODEL
// --------------------------------------------------

(async()=>{
 try{

  if(typeof ort==="undefined")throw new Error("AI engine library could not load. Check internet connection and reload.");
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
   "Unable to start camera — check browser permissions and whether another camera is active";

  resultEl.textContent=
   e?.message||String(e);
 }
};


// --------------------------------------------------
// STOP CAMERA
// --------------------------------------------------

function stopCamera(){
 if(typeof updateAiGuidance==="function")updateAiGuidance(null);

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

const aiGuidanceText=document.getElementById("aiGuidanceText");
const DETECTION_GUIDANCE={
 AIR_FILTER_CLEAN:"Engine air filter detected. The model suggests clean, but verify the filter visually using the manufacturer's procedure. Check for dust, damage and correct seating before recording.",
 AIR_FILTER_DIRTY:"Engine air filter detected. The model suggests dirty, but verify visually before deciding whether servicing or replacement is required.",
 BRAKE_PAD:"Brake pad detected (identification only). Measure friction material thickness with suitable equipment and compare with the manufacturer's minimum specification. Do not judge thickness from this image.",
 SPARK_PLUG:"Spark plug detected (identification only). With the engine safely off and cool, follow the manufacturer's removal and inspection procedure. Do not diagnose condition from detection alone."
};
const aiDetectionSummary=document.getElementById("aiDetectionSummary");
function updateAiGuidance(label){
 if(aiDetectionSummary)aiDetectionSummary.textContent=label&&DETECTION_GUIDANCE[label]?"AI detected: "+getDisplayInfo({label}).title+" · technician verification required":"No stable AI detection yet";
 if(!aiGuidanceText)return;
 aiGuidanceText.textContent=label&&DETECTION_GUIDANCE[label]
  ?DETECTION_GUIDANCE[label]
  :"No stable supported component detected. Aim at an engine air filter, brake pad or spark plug. Detection boxes appear only when the model identifies a component.";
}
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
  updateAiGuidance(null);

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
  updateAiGuidance(null);

  resultEl.textContent=
   `Checking… ${stableCount}/${REQUIRED_STABLE_FRAMES}`;

  return;
 }


 const info=
  getDisplayInfo(z);
 updateAiGuidance(z.label);
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

  const label=detInfo.title.toUpperCase()+" · "+(det.conf*100).toFixed(0)+"%";
  const left=Math.max(2,Math.min(r.width-2,x));
  const top=Math.max(2,Math.min(r.height-2,y));
  const right=Math.max(left,Math.min(r.width-2,x+w));
  const bottom=Math.max(top,Math.min(r.height-2,y+h));
  ctx.strokeStyle=det===z?"#3bf6a1":"#fbbf24";
  ctx.lineWidth=det===z?3:2;
  ctx.strokeRect(left,top,right-left,bottom-top);
  ctx.font="bold 12px Arial";
  const pad=8;
  const maxWidth=Math.max(20,r.width-8);
  const labelWidth=Math.min(maxWidth,ctx.measureText(label).width+pad*2);
  const labelX=Math.max(4,Math.min(r.width-labelWidth-4,left));
  const labelY=top>=28?top-26:Math.min(r.height-26,bottom+3);
  ctx.fillStyle="rgba(9,23,20,.92)";
  ctx.fillRect(labelX,labelY,labelWidth,24);
  ctx.fillStyle="#b5f5d0";
  ctx.fillText(label,labelX+pad,labelY+16,labelWidth-pad*2);
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
const MANUAL_ITEMS={
 "Engine Oil Level":["Within specified range","Below MIN","Above MAX","Unable to determine","Not inspected","Not applicable"],
 "Engine Oil Appearance":["Normal appearance","Dark/discoloured","Milky/emulsified","Visible contamination","Unable to determine","Not inspected","Not applicable"],
 "Coolant Level":["Within specified range","Below MIN","Above MAX","Unable to determine","Not inspected","Not applicable"],
 "Coolant Appearance":["Normal appearance","Discoloured","Visible contamination","Signs of oil mixing","Unable to determine","Not inspected","Not applicable"],
 "Brake Fluid Level":["Within specified range","Below MIN","Above MAX","Visible contamination","Unable to determine","Not inspected","Not applicable"],
 "Tyre Tread":["Above wear limit","Near wear limit","At/below wear limit","Uneven wear","Not measured","Not inspected","Not applicable"],
 "Tyre Condition":["No visible defects","Cut","Bulge","Cracking","Foreign object embedded","Uneven wear","Not inspected","Not applicable"],
 "Tyre Pressure":["Within vehicle specification","Below specification","Above specification","Not measured","Not inspected","Not applicable"],
 "Battery Visual Condition":["No visible defects","Terminal corrosion","Loose connection","Damaged casing","Not inspected","Not applicable"],
 "Exterior Lights":["Operating normally","Not operating","Intermittent operation","Damaged lens","Not inspected","Not applicable"],
 "Wiper Blades":["Wiping normally","Streaking","Torn rubber","Not operating","Not inspected","Not applicable"],
 "Drive Belt Visual Condition":["No visible defects","Cracked","Frayed","Glazed","Loose","Not inspected","Not applicable"],
 "Visible Fluid Leaks":["No visible leak","Seepage","Active leak","Unable to determine","Not inspected","Not applicable"]
};
const MANUAL_NONFINDINGS=["Not inspected","Not applicable","Not measured"];
function manualChecklistItems(){return Object.keys(MANUAL_ITEMS);}
const CHECKLIST_COMPONENTS=["Engine Air Filter","Brake Pad","Spark Plug",...manualChecklistItems()];
function checklistFor(sessionId){
 return CHECKLIST_COMPONENTS.map(component=>{
  const records=inspectionRecords.filter(r=>r.sessionId===sessionId&&r.component===component);
  const last=records[records.length-1];
  const manual=!!MANUAL_ITEMS[component];
  const status=!last?"Pending":last.decision==="review"?"Requires Further Inspection":last.decision==="not_inspected"?"Not Inspected":last.decision==="not_applicable"?"Not Applicable":"Completed";
  const basis=!last?"No saved finding":manual?(last.technicianAssessment||"Recorded"):(component==="Engine Air Filter"?"Technician: "+last.technicianAssessment:"Identification reviewed; condition not assessed");
  return {component,status,basis};
 });
}
function renderChecklist(){
 const container=document.getElementById("checklistItems");
 const progress=document.getElementById("checklistProgress");
 container.replaceChildren();
 if(!activeSession()){progress.textContent="Select a vehicle to view progress.";return;}
 const items=checklistFor(activeId);
 progress.textContent=items.filter(x=>x.status==="Completed").length+" of "+items.length+" component reviews completed";
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
  li.textContent=r.component+" | "+(r.source==="manual"?"Manual: ":"AI: ")+r.aiSuggestion+
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
 document.getElementById("saveManualBtn").disabled=!v;
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
  reportCell(tr,(finding.source==="manual"?"Manual observation: ":"")+(finding.aiSuggestion||"Not recorded")+(typeof finding.confidence==="number"?" ("+(finding.confidence*100).toFixed(1)+"% AI confidence)":""));
  reportCell(tr,finding.technicianAssessment);
  reportCell(tr,(finding.decision||"Not recorded")+(finding.remarks?" — "+finding.remarks:"")+(finding.savedAt?"\nSaved: "+new Date(finding.savedAt).toLocaleString():""));
  rows.appendChild(tr);
 }
 window.print();
};

/* V6.8 manual servicing: independent of camera; same per-vehicle persistence. */
const manualForm=document.getElementById("manualForm");
const manualComponent=document.getElementById("manualComponent");
const manualConditions=document.getElementById("manualConditions");
for(const name of manualChecklistItems()){
 const opt=document.createElement("option");opt.value=name;opt.textContent=name;manualComponent.appendChild(opt);
}
function renderManualOptions(){
 manualConditions.replaceChildren();
 for(const descriptor of MANUAL_ITEMS[manualComponent.value]||[]){
  const label=document.createElement("label");
  const checkbox=document.createElement("input");checkbox.type="checkbox";checkbox.value=descriptor;
  checkbox.onchange=()=>{
   if(checkbox.checked && (MANUAL_NONFINDINGS.includes(descriptor)||descriptor==="Unable to determine")){
    for(const other of manualConditions.querySelectorAll("input"))if(other!==checkbox)other.checked=false;
   }else if(checkbox.checked){
    for(const other of manualConditions.querySelectorAll("input"))if(MANUAL_NONFINDINGS.includes(other.value)||other.value==="Unable to determine")other.checked=false;
   }
  };
  label.append(checkbox,document.createTextNode(descriptor));manualConditions.appendChild(label);
 }
}
manualComponent.onchange=renderManualOptions;
renderManualOptions();
manualForm.onsubmit=e=>{
 e.preventDefault();
 if(!activeSession()){alert("Create or select a vehicle first.");return;}
 if(capturedFinding){alert("Finish or cancel the current AI review first.");return;}
 const descriptors=Array.from(manualConditions.querySelectorAll("input:checked"),x=>x.value);
 if(!descriptors.length){alert("Select at least one observed condition.");return;}
 const location=document.getElementById("manualLocation").value;
 const measurement=document.getElementById("manualMeasurement").value.trim();
 const remarks=document.getElementById("manualRemarks").value.trim();
 const nonFinding=descriptors[0];
 const decision=nonFinding==="Not inspected"?"not_inspected":nonFinding==="Not applicable"?"not_applicable":nonFinding==="Unable to determine"||nonFinding==="Not measured"?"review":"manual";
 const component=manualComponent.value;
 const finding={
  sessionId:activeId,source:"manual",component,label:"MANUAL",
  aiSuggestion:"Not used",technicianAssessment:descriptors.join("; ")+(measurement?" | Measurement: "+measurement:"")+(location?" | "+location:""),
  decision,remarks,savedAt:new Date().toISOString()
 };
 const next=[...inspectionRecords,finding];
 if(!storeSessions(sessions,next,activeId))return;
 inspectionRecords=next;
 manualForm.reset();renderManualOptions();renderVehicles();
 document.getElementById("manualSavedMessage").textContent="Saved to "+activeSession().registration+".";
};

const aiPage=document.getElementById("aiModePanel");
for(const name of ["viewer","status","resultCard","startBtn","captureBtn","inspectionPanel","supported"]){
 const el=document.getElementById(name);
 if(el)aiPage.appendChild(el);
}
const tabs=document.getElementById("modeTabs");
const extras=document.getElementById("manualExtras");
const extrasBtn=document.getElementById("manualExtrasBtn");
extrasBtn.onclick=()=>{
 extras.hidden=!extras.hidden;
 extrasBtn.setAttribute("aria-expanded",String(!extras.hidden));
 extrasBtn.textContent=extras.hidden?"+ Add location, measurement or remarks":"Hide additional details";
};
function showMode(mode){
 if(capturedFinding&&mode!=="ai"){alert("Complete the AI review first.");return;}
 document.getElementById("manualPanel").hidden=mode!=="manual";
 document.getElementById("checklistPanel").hidden=mode!=="manual";
 aiPage.hidden=mode!=="ai";
 document.getElementById("arGuidePanel").hidden=mode!=="ar";
 if(mode!=="ar")stopArCamera();
 if(mode!=="ai" && running)stopCamera();
 document.getElementById("recordsPanel").hidden=mode!=="results";
 for(const b of tabs.querySelectorAll("button")){
  const active=b.dataset.mode===mode;
  b.classList.toggle("selected",active);
  b.setAttribute("aria-pressed",String(active));
 }
}
for(const b of tabs.querySelectorAll("button"))b.onclick=()=>showMode(b.dataset.mode);
/* V7.0 guided servicing: informational 2D camera overlay, not spatial AR. */
const AR_GUIDES={
 "Engine Oil Level":["Park on level ground, secure the vehicle, and follow the manufacturer procedure for engine temperature and waiting time.","Locate the dipstick using the vehicle manual. Keep clear of hot or moving parts.","Remove and wipe the dipstick, reinsert fully, then remove and read the level against MIN and MAX marks.","Record the observed level. Do not assume a reading from the camera image."],
 "Coolant Level":["Allow the engine to cool fully. Never open a hot pressurised cooling system.","Locate the correct coolant expansion tank using the vehicle manual.","Observe the level against the marked range without opening a hot cap.","Record the level and any visible contamination."],
 "Brake Fluid Level":["Park safely and consult the manufacturer procedure.","Locate the brake fluid reservoir; avoid introducing contamination.","Inspect the level against reservoir markings without assuming the reason for a low level.","Record the observation. Investigate low level, leaks, or brake wear before deciding action."],
 "Tyre Tread":["Secure the vehicle and identify the tyre location.","Inspect the tread across its width for uneven wear and damage.","Use a suitable tread-depth gauge; compare measurements with applicable limits.","Record tread condition, location, and measured depth."],
 "Tyre Condition":["Identify the tyre location and inspect only when safe.","Look for cuts, bulges, cracking, embedded objects, and uneven wear.","Do not remove embedded objects or attempt repairs based only on appearance.","Record every observed defect and tyre location."],
 "Tyre Pressure":["Find the vehicle manufacturer's specified cold tyre pressure.","Measure with a suitable tyre pressure gauge when tyres are cold.","Compare the measured pressure with the vehicle specification, not the sidewall maximum.","Record pressure, units, and tyre location."],
 "Battery Visual Condition":["Turn off ignition and electrical loads before visual inspection.","Look for casing damage, loose terminals, corrosion, or leaks without touching exposed conductors.","Avoid shorting terminals; follow battery and vehicle-specific safety procedures.","Record visual findings. Voltage or health requires suitable test equipment."],
 "Exterior Lights":["Secure the vehicle and activate the relevant light function.","Check operation of headlights, indicators, brake lights and other required lamps.","Note non-operating, intermittent, or damaged lights; obtain assistance where necessary.","Record the specific light and observed fault."],
 "Wiper Blades":["Ensure the windscreen is clear and wet before operating wipers.","Inspect rubber for splits, hardening or damage.","Operate wipers safely and observe streaking or poor clearing.","Record blade condition and affected side."],
 "Drive Belt Visual Condition":["Switch off engine and secure against accidental starting. Keep hands away from moving parts.","Locate the accessory drive belt using the vehicle manual.","Visually check for cracking, fraying, glazing and obvious damage; do not touch moving belts.","Record observations. Belt tension requires the correct specified method."],
 "Visible Fluid Leaks":["Secure the vehicle and avoid hot surfaces or moving components.","Look for fresh drips, wet areas or stains from a safe viewing position.","Do not identify fluid type solely from colour; avoid contact with unknown fluids.","Record location, seepage or active leakage and arrange follow-up inspection."],
 "Engine Oil Appearance":["Follow the vehicle manufacturer's safe oil inspection procedure.","Inspect a suitable oil sample or dipstick under adequate lighting.","Note discolouration, milky appearance or visible contamination without inferring serviceability from colour alone.","Record observed appearance and any follow-up required."],
 "Coolant Appearance":["Ensure the cooling system is cool; never open a hot pressure cap.","Observe coolant through the reservoir where visible.","Note unusual discolouration, oil mixing or contamination without opening a hot system.","Record the observation and arrange further checks if needed."]
};
const arTask=document.getElementById("arTask");
const arVideo=document.getElementById("arVideo");
let arStream=null;
let arStep=0;
for(const item of manualChecklistItems()){
 const opt=document.createElement("option");opt.value=item;opt.textContent=item;arTask.appendChild(opt);
}
function updateArGuide(){
 const steps=AR_GUIDES[arTask.value]||["Refer to the manufacturer's inspection procedure.","Record the observed condition."];
 arStep=Math.max(0,Math.min(arStep,steps.length-1));
 document.getElementById("arStepCount").textContent=arTask.value+" • Step "+(arStep+1)+" of "+steps.length;
 document.getElementById("arStepText").textContent=steps[arStep];
 document.getElementById("arPrevious").disabled=arStep===0;
 document.getElementById("arNext").disabled=arStep===steps.length-1;
 if(window.autoVisionReadStep && document.getElementById("arGuidePanel").hidden===false)window.autoVisionReadStep();
}
function stopArCamera(){
 if(arStream){arStream.getTracks().forEach(track=>track.stop());arStream=null;}
 arVideo.srcObject=null;
 document.getElementById("arCameraBtn").textContent="Start AR Camera";
 document.getElementById("arCameraStatus").textContent="Camera off";
}
arTask.onchange=()=>{arStep=0;updateArGuide();};
document.getElementById("arPrevious").onclick=()=>{arStep--;updateArGuide();};
document.getElementById("arNext").onclick=()=>{arStep++;updateArGuide();};
document.getElementById("arCameraBtn").onclick=async()=>{
 if(arStream){stopArCamera();return;}
 if(running){alert("Stop the AI camera before starting the AR guide.");return;}
 try{
  arStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:"environment"}},audio:false});
  arVideo.srcObject=arStream;await arVideo.play();
  document.getElementById("arCameraBtn").textContent="Stop AR Camera";
  document.getElementById("arCameraStatus").textContent="Live camera — instructions are informational overlays";
 }catch(e){
  stopArCamera();
  document.getElementById("arCameraStatus").textContent="Camera unavailable: "+(e.message||"Permission denied")+" — check browser camera permission and close other camera apps.";
 }
};
const arQuickConditions=document.getElementById("arQuickConditions");
const arQuickStatus=document.getElementById("arQuickStatus");
function renderArConditions(){
 arQuickConditions.replaceChildren();
 arQuickStatus.textContent="";
 for(const descriptor of MANUAL_ITEMS[arTask.value]||[]){
  const label=document.createElement("label");
  const cb=document.createElement("input");cb.type="checkbox";cb.value=descriptor;
  cb.onchange=()=>{
   if(!cb.checked)return;
   const exclusive=MANUAL_NONFINDINGS.includes(descriptor)||descriptor==="Unable to determine";
   for(const other of arQuickConditions.querySelectorAll("input")){
    if(other!==cb&&(exclusive||MANUAL_NONFINDINGS.includes(other.value)||other.value==="Unable to determine"))other.checked=false;
   }
  };
  label.append(cb,document.createTextNode(descriptor));arQuickConditions.appendChild(label);
 }
}
const arMore=document.getElementById("arQuickMore");
const arExtras=document.getElementById("arQuickExtras");
arMore.onclick=()=>{
 arExtras.hidden=!arExtras.hidden;
 arMore.setAttribute("aria-expanded",String(!arExtras.hidden));
 arMore.textContent=arExtras.hidden?"+ Add location, measurement or remarks":"Hide additional details";
};
const priorArTaskChange=arTask.onchange;
arTask.onchange=()=>{
 priorArTaskChange();
 renderArConditions();
};
document.getElementById("arRecord").onclick=()=>{
 if(!activeSession()){alert("Create or select a vehicle inspection first.");return;}
 if(capturedFinding){alert("Finish the current AI review first.");return;}
 const descriptors=Array.from(arQuickConditions.querySelectorAll("input:checked"),x=>x.value);
 if(!descriptors.length){arQuickStatus.textContent="Select at least one observed condition.";return;}
 const location=document.getElementById("arQuickLocation").value;
 const measurement=document.getElementById("arQuickMeasurement").value.trim();
 const remarks=document.getElementById("arQuickRemarks").value.trim();
 const first=descriptors[0];
 const decision=first==="Not inspected"?"not_inspected":first==="Not applicable"?"not_applicable":first==="Unable to determine"||first==="Not measured"?"review":"manual";
 const finding={
  sessionId:activeId,source:"manual",component:arTask.value,label:"MANUAL",
  aiSuggestion:"Not used",technicianAssessment:descriptors.join("; ")+(measurement?" | Measurement: "+measurement:"")+(location?" | "+location:""),
  decision,remarks,savedAt:new Date().toISOString()
 };
 const next=[...inspectionRecords,finding];
 if(!storeSessions(sessions,next,activeId)){arQuickStatus.textContent="Unable to save finding.";return;}
 inspectionRecords=next;
 renderVehicles();
 renderArConditions();
 for(const field of ["arQuickLocation","arQuickMeasurement","arQuickRemarks"])document.getElementById(field).value="";
 arQuickStatus.textContent="Saved "+finding.component+" to "+activeSession().registration+".";
};
renderArConditions();
updateArGuide();
showMode("manual");

/* V7.7: voice camera operation and spoken AR steps. */
const voiceBtn=document.getElementById("voiceCommandBtn");
const voiceStatus=document.getElementById("voiceCommandStatus");
const SpeechAPI=window.SpeechRecognition||window.webkitSpeechRecognition;
const speechSynthesisAvailable="speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
let voiceRecognizer=null;
let handsFreeEnabled=false;
let voiceStarting=false;
let voiceRetryCount=0;
let voiceRetryTimer=null;
let speakingStep=false;
let speechGeneration=0;
const MAX_VOICE_RETRIES=4;
function voiceUi(message){
 voiceBtn.textContent=handsFreeEnabled?"🔴 Stop Listening":"🎤 Start Hands-Free Listening";
 if(message)voiceStatus.textContent=message;
}
function cancelStepSpeech(){
 speechGeneration++;
 speakingStep=false;
 if(speechSynthesisAvailable)window.speechSynthesis.cancel();
}
function stopHandsFree(message){
 handsFreeEnabled=false;
 clearTimeout(voiceRetryTimer);
 voiceRetryTimer=null;
 voiceRetryCount=0;
 cancelStepSpeech();
 const previous=voiceRecognizer;
 voiceRecognizer=null;
 voiceStarting=false;
 if(previous){try{previous.abort();}catch(_){}}
 voiceUi(message||"Microphone off. Tap Start Hands-Free Listening to resume.");
}
function scheduleVoiceRestart(){
 if(!handsFreeEnabled||voiceRetryTimer||document.hidden||speakingStep)return;
 if(voiceRetryCount>=MAX_VOICE_RETRIES){
  stopHandsFree("Speech recognition repeatedly stopped. Tap Start Hands-Free Listening to retry.");
  return;
 }
 voiceRetryTimer=setTimeout(()=>{
  voiceRetryTimer=null;
  if(handsFreeEnabled)startVoiceCycle();
 },Math.min(800+voiceRetryCount*700,3000));
}
function speakTextAloud(text){
 if(!speechSynthesisAvailable){
  voiceStatus.textContent="Speech playback is not supported in this browser.";
  return;
 }
 cancelStepSpeech();
 const token=speechGeneration;
 speakingStep=true;
 clearTimeout(voiceRetryTimer);
 voiceRetryTimer=null;
 const previous=voiceRecognizer;
 voiceRecognizer=null;
 voiceStarting=false;
 if(previous){try{previous.abort();}catch(_){}}
 const utterance=new SpeechSynthesisUtterance(text);
 const voices=window.speechSynthesis.getVoices();
 const english=voices.find(v=>v.lang.toLowerCase()==="en-sg")||
               voices.find(v=>v.lang.toLowerCase().startsWith("en-"))||
               voices.find(v=>v.lang.toLowerCase().startsWith("en"));
 if(english){utterance.voice=english;utterance.lang=english.lang;}
 else utterance.lang="en-US";
 utterance.rate=0.92;
 utterance.volume=1;
 utterance.onstart=()=>{
  if(token===speechGeneration)voiceStatus.textContent="🔊 Speaking now. If silent, check phone media volume and silent mode.";
 };
 utterance.onend=()=>{
  if(token!==speechGeneration)return;
  speakingStep=false;
  voiceStatus.textContent="Speech finished. Listening for your next command.";
  if(handsFreeEnabled)scheduleVoiceRestart();
 };
 utterance.onerror=event=>{
  if(token!==speechGeneration)return;
  speakingStep=false;
  voiceStatus.textContent="Speech playback error: "+(event.error||"unknown")+". Tap Test Speaker / Enable Audio.";
  if(handsFreeEnabled)scheduleVoiceRestart();
 };
 voiceStatus.textContent="Preparing spoken instruction…";
 try{
  window.speechSynthesis.cancel();
  window.speechSynthesis.resume();
  window.speechSynthesis.speak(utterance);
 }catch(e){
  speakingStep=false;
  voiceStatus.textContent="Speech playback failed: "+e.message;
  if(handsFreeEnabled)scheduleVoiceRestart();
 }
}
function speakArStep(){
 speakTextAloud(document.getElementById("arStepText").textContent);
}
const testSpeechBtn=document.getElementById("testSpeechBtn");
if(testSpeechBtn){
 testSpeechBtn.onclick=()=>{
  speakTextAloud("AutoVision audio test. If you can hear this message, spoken inspection guidance is enabled.");
 };
 if(!speechSynthesisAvailable)testSpeechBtn.disabled=true;
}
window.autoVisionReadStep=()=>{if(handsFreeEnabled)speakArStep();};
function openVoiceCamera(mode){
 if(capturedFinding&&mode!=="ai"){
  voiceStatus.textContent="Complete the pending AI finding review before switching modes.";
  return;
 }
 showMode(mode);
 if(mode==="ai"){
  if(!running){
   if(startBtn.disabled){voiceStatus.textContent="AI model is still loading; please wait.";return;}
   startBtn.click();
  }
 }else if(mode==="ar"){
  const cameraBtn=document.getElementById("arCameraBtn");
  if(!arStream)cameraBtn.click();
  if(handsFreeEnabled)speakArStep();
 }
}
function handleVoiceCommand(raw){
 const phrase=raw.toLowerCase().trim().replace(/[.!?]/g,"");
 voiceStatus.textContent="Heard: "+phrase;
 if(/\b(stop listening|microphone off|disable voice)\b/.test(phrase)){
  stopHandsFree("Hands-free listening stopped by voice command.");
 }else if(phrase.includes("stop camera")){
  if(running)stopCamera();
  if(arStream)stopArCamera();
 }else if(phrase.includes("start ar camera")||phrase.includes("start ar guide")||phrase.includes("ar guide")){
  openVoiceCamera("ar");
 }else if(phrase.includes("start ai camera")||phrase.includes("ai camera")||phrase.includes("start camera")){
  openVoiceCamera("ai");
 }else if(phrase.includes("capture finding")){
  if(!running){voiceStatus.textContent="Start the AI camera before capturing.";return;}
  if(!captureBtn.disabled)captureBtn.click();
  else voiceStatus.textContent="No stable AI detection to capture.";
 }else if(phrase.includes("next step")){
  if(capturedFinding){voiceStatus.textContent="Finish reviewing the captured finding first.";return;}
  if(document.getElementById("arGuidePanel").hidden){openVoiceCamera("ar");return;}
  const next=document.getElementById("arNext");
  if(!next.disabled)next.click();
  else voiceStatus.textContent="Already at the final step.";
 }else if(phrase.includes("previous step")||phrase.includes("back step")){
  if(capturedFinding){voiceStatus.textContent="Finish reviewing the captured finding first.";return;}
  if(document.getElementById("arGuidePanel").hidden){openVoiceCamera("ar");return;}
  const previous=document.getElementById("arPrevious");
  if(!previous.disabled)previous.click();
  else voiceStatus.textContent="Already at the first step.";
 }else if(phrase.includes("repeat step")||phrase.includes("read step")){
  if(document.getElementById("arGuidePanel").hidden){openVoiceCamera("ar");return;}
  speakArStep();
 }else if(phrase.includes("stop reading")||phrase.includes("be quiet")){
  cancelStepSpeech();
  scheduleVoiceRestart();
 }else if(phrase.includes("service checklist")){
  if(capturedFinding){voiceStatus.textContent="Finish reviewing the captured finding first.";return;}
  cancelStepSpeech();showMode("manual");
 }else if(phrase.includes("results")){
  if(capturedFinding){voiceStatus.textContent="Finish reviewing the captured finding first.";return;}
  cancelStepSpeech();showMode("results");
 }else if(phrase.includes("save finding")){
  voiceStatus.textContent="Review the technician assessment and tap Save Finding to confirm.";
 }else{
  voiceStatus.textContent="Command not recognised: "+phrase;
 }
}
function startVoiceCycle(){
 if(!handsFreeEnabled||voiceRecognizer||voiceStarting||document.hidden||speakingStep)return;
 voiceStarting=true;
 const recognition=new SpeechAPI();
 voiceRecognizer=recognition;
 recognition.lang="en-SG";
 recognition.continuous=true;
 recognition.interimResults=false;
 recognition.onstart=()=>{
  voiceStarting=false;
  voiceRetryCount=0;
  voiceUi("Listening. Say 'AR guide', 'AI camera', or 'next step'.");
 };
 recognition.onresult=event=>{
  for(let i=event.resultIndex;i<event.results.length;i++){
   if(event.results[i].isFinal){
    handleVoiceCommand(event.results[i][0].transcript||"");
    if(!handsFreeEnabled||speakingStep)break;
   }
  }
 };
 recognition.onerror=e=>{
  voiceStatus.textContent="Microphone: "+e.error;
  if(["not-allowed","service-not-allowed","audio-capture"].includes(e.error)){
   stopHandsFree("Microphone unavailable ("+e.error+"). Check permissions or use buttons.");
  }else if(!speakingStep)voiceRetryCount++;
 };
 recognition.onend=()=>{
  if(voiceRecognizer===recognition)voiceRecognizer=null;
  voiceStarting=false;
  if(handsFreeEnabled&&!speakingStep)scheduleVoiceRestart();
 };
 try{recognition.start();}
 catch(e){voiceRecognizer=null;voiceStarting=false;voiceRetryCount++;voiceStatus.textContent="Voice could not start: "+e.message;scheduleVoiceRestart();}
}
if(!SpeechAPI){
 voiceBtn.disabled=true;
 voiceStatus.textContent="Voice recognition is unavailable in this browser. Use the existing buttons.";
}else{
 voiceBtn.onclick=()=>{
  if(handsFreeEnabled){stopHandsFree();return;}
  handsFreeEnabled=true;
  voiceRetryCount=0;
  voiceUi("Starting microphone…");
  startVoiceCycle();
 };
 document.addEventListener("visibilitychange",()=>{
  if(document.hidden){
   if(handsFreeEnabled){
    clearTimeout(voiceRetryTimer);voiceRetryTimer=null;
    cancelStepSpeech();
    if(voiceRecognizer){try{voiceRecognizer.abort();}catch(_){}}
    voiceStatus.textContent="Listening paused while browser is in the background.";
   }
  }else if(handsFreeEnabled){voiceStatus.textContent="Resuming listening…";scheduleVoiceRestart();}
 });
 window.addEventListener("pagehide",()=>stopHandsFree());
}
