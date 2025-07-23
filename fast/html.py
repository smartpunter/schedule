"""HTML report generation for schedules."""

from typing import Any, Dict
import json

def generate_html(
    schedule: Dict[str, Any],
    cfg: Dict[str, Any],
    path: str = "schedule.html",
    generated: str | None = None,
    include_config: bool = False,
) -> None:
    """Create interactive HTML overview of the schedule."""
    schedule_json = json.dumps(schedule, ensure_ascii=False)
    cfg_json = json.dumps(cfg, ensure_ascii=False)
    html = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Schedule Overview</title>
<style>
body { font-family: Arial, sans-serif; }
.schedule-grid { border-collapse: collapse; width: 100%; display:grid; }
.schedule-grid .cell, .schedule-grid .header { border:1px solid #999; padding:4px; vertical-align:top; }
.schedule-grid .cell { display:flex; flex-direction:column; }
.schedule-grid .header { background:#f0f0f0; text-align:center; }
.mini-grid { margin-top:10px; }
.class-block { display:flex; flex-direction:column; margin-bottom:4px; }
.class-line { display:flex; gap:4px; width:100%; }
.class-line span { flex:1; }
.cls-subj { flex:0 0 50%; text-align:left; }
.cls-room { flex:0 0 30%; text-align:right; }
.cls-part { flex:0 0 20%; text-align:right; }
.slot-info { display:flex; gap:4px; justify-content:space-between; font-size:0.9em; color:#555; cursor:pointer; margin-top:auto; }
.slot-info span { flex:1 1 20%; text-align:center; }
.modal { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); }
.modal-content { background:#fff; position:fixed; top:5vh; bottom:5vh; left:10vw; right:10vw; margin:0; padding:20px; overflow:auto; }
.modal-header { position:relative; text-align:center; margin-bottom:10px; }
.close { position:absolute; right:0; top:0; cursor:pointer; font-size:20px; }
.history { display:inline-flex; gap:6px; justify-content:center; flex-wrap:wrap; }
.hist-item { font-size:0.9em; color:#888; cursor:pointer; }
.hist-item.active { font-weight:bold; color:#000; }
.clickable { color:#0066cc; cursor:pointer; text-decoration:underline; }
.slot-detail .slot-class{border-top:1px solid #ddd;padding:2px 0;}
.slot-detail .slot-class:first-child{border-top:none;}
.detail-line{display:flex;gap:6px;}
.detail-line span{flex:1;}
.detail-subj{flex:0 0 40%;text-align:left;}
.detail-teacher{flex:0 0 20%;text-align:left;}
.detail-room{flex:0 0 15%;text-align:right;}
.detail-size{flex:0 0 10%;text-align:right;}
.detail-part{flex:0 0 15%;text-align:right;}
.detail-students{font-size:0.9em;margin-left:1em;}
.info-table{border-collapse:collapse;width:100%;margin-top:6px;}
.info-table th,.info-table td{border:1px solid #999;padding:4px;vertical-align:top;}
.info-table th{background:#f0f0f0;}
.info-table td.num{text-align:right;}
/* overview tables below the schedule */
.overview-section{margin-top:20px;}
.overview-table{border:1px solid #999;border-collapse:collapse;width:70%;margin:0 auto;}
.overview-header,.overview-row{display:flex;align-items:center;}
.overview-header span,.overview-row span{border-right:1px solid #999;text-align:center;}
.overview-header span:last-child,.overview-row span:last-child{border-right:none;}
.overview-row{border-top:1px solid #999;}
.overview-row:first-child{border-top:none;}
.overview-header{background:#f0f0f0;font-weight:bold;}
.person-name{flex:0 0 25%;text-align:left;}
.person-info{flex:0 0 6%;text-align:center;}
.person-pen{flex:0 0 6%;text-align:center;}
.person-hours{flex:0 0 6%;text-align:center;}
.person-time{flex:0 0 6%;text-align:center;}
.person-subjects{flex:0 0 51%;text-align:left;padding:0;}
.subject-list{display:flex;flex-direction:column;}
.subject-line{display:flex;gap:6px;border-top:1px solid #ddd;padding:2px 4px;}
.subject-line:first-child{border-top:none;}
.subject-name{flex:0 0 60%;text-align:left;}
.subject-count{flex:0 0 20%;text-align:center;}
.subject-extra{flex:0 0 20%;text-align:center;}
/* parameter blocks */
.param-table{--gap:8px;display:flex;flex-wrap:wrap;gap:var(--gap);justify-content:center;margin-top:10px;}
.param-block{flex:0 0 calc((100% - (6*var(--gap)))/6);border:1px solid #999;text-align:center;}
.param-block div{margin:0 0 5px 0; padding: 4px 0;}
.param-name{background:#f0f0f0;}
.meta{font-size:0.9em;margin-bottom:10px;}
pre{white-space:pre-wrap;word-break:break-all;background:#f8f8f8;padding:10px;}
</style>
</head>
<body>
<h1>Schedule Overview</h1>
__META__
<div id="table" class="schedule-grid"></div>
<h2 class="overview-section">Teachers Overview</h2>
<div id="teachers" class="overview-table"></div>
<h2 class="overview-section">Students Overview</h2>
<div id="students" class="overview-table"></div>
<div id="modal" class="modal"><div class="modal-content"><div class="modal-header"><div id="history" class="history"></div><span id="close" class="close">&#10006;</span></div><div id="modal-body"></div></div></div>
<script>
const scheduleData = __SCHEDULE__;
const configData = __CONFIG__;
</script>
<script>
(function(){
const modal=document.getElementById('modal');
const close=document.getElementById('close');
const historyBox=document.getElementById('history');
const modalBody=document.getElementById('modal-body');
const cfgLink=document.getElementById('show-config');
close.onclick=()=>{modal.style.display='none';};
window.onclick=e=>{if(e.target==modal)modal.style.display='none';};
if(cfgLink){cfgLink.onclick=()=>{showConfig();};}
const studentSize={};
(configData.students||[]).forEach((s,i)=>{studentSize[s.name]=s.group||1;});
const teacherIndex={};
const teacherDisplay={};
(configData.teachers||[]).forEach((t,i)=>{teacherIndex[t.name]=i;teacherDisplay[t.name]=t.printName||t.name;});
const studentIndex={};
(configData.students||[]).forEach((s,i)=>{studentIndex[s.name]=i;});
const subjectDisplay={};
Object.keys(configData.subjects||{}).forEach(id=>{const info=configData.subjects[id]||{};subjectDisplay[id]=info.printName||info.name||id;});
const cabinetDisplay={};
Object.keys(configData.cabinets||{}).forEach(id=>{const info=configData.cabinets[id]||{};cabinetDisplay[id]=info.printName||id;});
const teacherSet=new Set(Object.keys(teacherIndex));
const studentSet=new Set(Object.keys(studentIndex));
function personLink(name,role){
  if(role==='teacher' || (!role && teacherSet.has(name) && !studentSet.has(name))){
    const id=teacherIndex[name];
    return '<span class="clickable teacher" data-id="'+id+'">'+(teacherDisplay[name]||name)+'</span>';
  }
  if(role==='student' || (!role && studentSet.has(name) && !teacherSet.has(name))){
    const id=studentIndex[name];
    return '<span class="clickable student" data-id="'+id+'">'+name+'</span>';
  }
  return name;
}
function teacherSpan(name,subj){
  const id=teacherIndex[name];
  const prim=(configData.subjects[subj]||{}).primaryTeachers||[];
  const disp=teacherDisplay[name]||name;
  const inner=prim.includes(name)?'<strong>'+disp+'</strong>':disp;
  return '<span class="clickable teacher" data-id="'+id+'">'+inner+'</span>';
}
function cabinetSpan(name){
  return '<span class="clickable cabinet" data-id="'+name+'">'+(cabinetDisplay[name]||name)+'</span>';
}
function makeParamTable(list){
  let html='<div class="param-table">';
  list.forEach(item=>{
    const name = Array.isArray(item) ? item[0] : item.name;
    const val = Array.isArray(item) ? item[1] : item.value;
    const bold = Array.isArray(item) ? item[2] : item.bold;
    const valueHtml = bold ? '<strong>'+val+'</strong>' : val;
    html += '<div class="param-block"><div class="param-name">'+name+'</div><div>'+valueHtml+'</div></div>';
  });
  html+='</div>';return html;
}
function countStudents(list){return(list||[]).reduce((a,n)=>a+(studentSize[n]||1),0);}
let historyStack=[];
let historyTitles=[];
let historyIndex=-1;
function getTitle(html){const m=html.match(/<h2[^>]*>(.*?)<\/h2>/i);return m?m[1]:'';}
function renderModal(){
 modalBody.innerHTML=historyStack[historyIndex]||'';
 modal.style.display='block';
 historyBox.innerHTML='';
 const start=Math.max(0,historyTitles.length-3);
 for(let i=start;i<historyTitles.length;i++){
   const item=document.createElement('span');
   item.className='hist-item'+(i===historyIndex?' active':'');
   item.textContent=historyTitles[i];
   item.onclick=()=>{historyIndex=i;renderModal();};
   historyBox.appendChild(item);
   if(i<historyTitles.length-1){
     const sep=document.createElement('span');
     sep.textContent='>';
     historyBox.appendChild(sep);
   }
 }
}
function openModal(html,reset=true){
 const title=getTitle(html);
 if(reset){
   historyStack=[html];
   historyTitles=[title];
 }else{
   const idx=historyTitles.indexOf(title);
   if(idx>=0){historyStack.splice(idx,1);historyTitles.splice(idx,1);}
   historyStack.push(html);historyTitles.push(title);
   if(historyStack.length>3){
     historyStack=historyStack.slice(-3);
     historyTitles=historyTitles.slice(-3);
   }
}
historyIndex=historyStack.length-1;
renderModal();
}
function showConfig(){
 const json=JSON.stringify(configData,null,2);
 const html='<h2>Configuration</h2><button id="cfg-copy">Copy</button><pre id="cfg-pre"></pre>';
 openModal(html);
 document.getElementById('cfg-pre').textContent=json;
 const btn=document.getElementById('cfg-copy');
 if(btn){btn.onclick=()=>{navigator.clipboard.writeText(json);};}
}
const COLOR_MIN=[220,255,220];
const COLOR_MID=[255,255,255];
const COLOR_MAX=[255,220,220];

function buildTable(){
 const container=document.getElementById('table');
 container.style.gridTemplateColumns=`auto repeat(${scheduleData.days.length},1fr)`;
 const maxSlots=Math.max(...scheduleData.days.map(d=>d.slots.length?Math.max(...d.slots.map(s=>s.slotIndex)):0))+1;
 const cells=[];
 let minP=Infinity,maxP=-Infinity;
 container.innerHTML='';
 container.appendChild(document.createElement('div'));
 scheduleData.days.forEach(d=>{const h=document.createElement('div');h.className='header';h.textContent=d.name;container.appendChild(h);});
  for(let i=0;i<maxSlots;i++){
   const hdr=document.createElement('div');hdr.className='header';hdr.textContent='Lesson '+(i+1);container.appendChild(hdr);
   scheduleData.days.forEach(day=>{
     const slot=day.slots.find(s=>s.slotIndex==i) || {classes:[],gaps:{students:[],teachers:[]},home:{students:[],teachers:[]},penalty:{}};
     const cell=document.createElement('div');
     cell.className='cell';
     const pVal=Object.values(slot.penalty||{}).reduce((a,b)=>a+b,0);
     minP=Math.min(minP,pVal);maxP=Math.max(maxP,pVal);
     slot.classes.forEach(cls=>{
       const block=document.createElement('div');
       block.className='class-block';
       const subj=subjectDisplay[cls.subject]||cls.subject;
       const part=(cls.length>1)?((i-cls.start+1)+'/'+cls.length):'1/1';
       const l1=document.createElement('div');
       l1.className='class-line';
        const rooms=(cls.cabinets||[]).map(c=>cabinetSpan(c)).join(', ');
        l1.innerHTML='<span class="cls-subj clickable subject" data-id="'+cls.subject+'">'+subj+'</span>'+
         '<span class="cls-room">'+rooms+'</span>'+
         '<span class="cls-part">'+part+'</span>';
       const l2=document.createElement('div');
       l2.className='class-line';
      const tNames=(cls.teachers||[]).map(t=>teacherSpan(t,cls.subject)).join(', ');
       l2.innerHTML='<span class="cls-teach">'+tNames+'</span>'+
        '<span class="cls-size">'+cls.size+'</span>';
       block.appendChild(l1);block.appendChild(l2);
       cell.appendChild(block);
     });
     const info=document.createElement('div');
     info.className='slot-info';
     info.dataset.day=day.name;info.dataset.slot=i;
    function makeSpan(val,title){const s=document.createElement('span');s.textContent=val;s.title=title;return s;}
    const detail=(slot.penaltyDetails||[]).map(p=>p.name+' '+p.type+': '+p.amount.toFixed(1)).join('\n');
    info.appendChild(makeSpan(pVal.toFixed(1),detail||'Penalty'));
     info.appendChild(makeSpan(countStudents(slot.home.students),'Students at home: '+(slot.home.students.join(', ')||'-')));
     info.appendChild(makeSpan(slot.home.teachers.length,'Teachers at home: '+(slot.home.teachers.join(', ')||'-')));
     info.appendChild(makeSpan(countStudents(slot.gaps.students),'Students waiting for class: '+(slot.gaps.students.join(', ')||'-')));
     info.appendChild(makeSpan(slot.gaps.teachers.length,'Teachers waiting for class: '+(slot.gaps.teachers.join(', ')||'-')));
     cell.appendChild(info);
     container.appendChild(cell);
     cells.push({el:cell,val:pVal});
   });
 }

 const mid=(minP+maxP)/2;
 function mix(a,b,f){return a+(b-a)*f;}
 function colorFor(v){
   if(maxP===minP)return `rgb(${COLOR_MID.join(',')})`;
   if(v<=mid){
     const f=(v-minP)/(mid-minP||1);
     const r=Math.round(mix(COLOR_MIN[0],COLOR_MID[0],f));
     const g=Math.round(mix(COLOR_MIN[1],COLOR_MID[1],f));
     const b=Math.round(mix(COLOR_MIN[2],COLOR_MID[2],f));
     return `rgb(${r},${g},${b})`;
   }
   const f=(v-mid)/(maxP-mid||1);
   const r=Math.round(mix(COLOR_MID[0],COLOR_MAX[0],f));
   const g=Math.round(mix(COLOR_MID[1],COLOR_MAX[1],f));
   const b=Math.round(mix(COLOR_MID[2],COLOR_MAX[2],f));
   return `rgb(${r},${g},${b})`;
 }
cells.forEach(c=>{c.el.style.background=colorFor(c.val);});
}

function makeGrid(filterFn){
 const maxSlots=Math.max(...scheduleData.days.map(d=>d.slots.length?Math.max(...d.slots.map(s=>s.slotIndex)):0))+1;
 let html='<div class="schedule-grid mini-grid" style="grid-template-columns:auto repeat('+scheduleData.days.length+',1fr)">';
 html+='<div></div>';
 scheduleData.days.forEach(d=>{html+='<div class="header">'+d.name+'</div>';});
 for(let i=0;i<maxSlots;i++){
  html+='<div class="header">Lesson '+(i+1)+'</div>';
   scheduleData.days.forEach(day=>{
     const slot=day.slots.find(s=>s.slotIndex==i)||{classes:[]};
     html+='<div class="cell">';
     slot.classes.filter(filterFn).forEach(cls=>{
       const subj=subjectDisplay[cls.subject]||cls.subject;
       const part=(cls.length>1)?((i-cls.start+1)+'/'+cls.length):'1/1';
       const tNames=(cls.teachers||[]).map(t=>teacherSpan(t,cls.subject)).join(', ');
        const rooms=(cls.cabinets||[]).map(c=>cabinetSpan(c)).join(', ');
        html+='<div class="class-block">'+
         '<div class="class-line">'+
          '<span class="cls-subj clickable subject" data-id="'+cls.subject+'">'+subj+'</span>'+
          '<span class="cls-room">'+rooms+'</span>'+
          '<span class="cls-part">'+part+'</span>'+
        '</div>'+
        '<div class="class-line">'+
          '<span class="cls-teach">'+tNames+'</span>'+
          '<span class="cls-size">'+cls.size+'</span>'+
        '</div>'+
       '</div>';
     });
     html+='</div>';
   });
 }
 html+='</div>';
 return html;
}

function showSlot(day,idx,fromModal=false){
 const d=scheduleData.days.find(x=>x.name===day);if(!d)return;
 const slot=d.slots.find(s=>s.slotIndex==idx);if(!slot)return;
 const total=Object.values(slot.penalty||{}).reduce((a,b)=>a+b,0);
 let html='<h2>'+day+' lesson '+(idx+1)+'</h2><p>Total penalty: '+total.toFixed(1)+'</p>';
 html+='<div class="slot-detail">';
 slot.classes.forEach((cls)=>{
   const subj=subjectDisplay[cls.subject]||cls.subject;
   const part=(cls.length>1)?((idx-cls.start+1)+'/'+cls.length):'1/1';
   html+='<div class="slot-class">'+
     '<div class="detail-line">'+
       '<span class="detail-subj clickable subject" data-id="'+cls.subject+'">'+subj+'</span>'+
      '<span class="detail-teacher">'+(cls.teachers||[]).map(t=>teacherSpan(t,cls.subject)).join(', ')+'</span>'+
       '<span class="detail-room">'+(cls.cabinets||[]).map(c=>cabinetSpan(c)).join(', ')+'</span>'+
       '<span class="detail-size">'+cls.size+'</span>'+
       '<span class="detail-part">'+part+'</span>'+
     '</div>';
   const studs=cls.students.map(n=>personLink(n,'student')).join(', ');
   if(studs)html+='<div class="detail-students">'+studs+'</div>';
   html+='</div>';
 });
 html+='</div>';
 const homeStu=slot.home.students.map(n=>personLink(n,'student')).join(', ');
 const homeTeach=slot.home.teachers.map(n=>personLink(n,'teacher')).join(', ');
 const waitStu=slot.gaps.students.map(n=>personLink(n,'student')).join(', ');
 const waitTeach=slot.gaps.teachers.map(n=>personLink(n,'teacher')).join(', ');
 html+='<h3>Presence</h3>';
 html+='<table class="info-table"><tr><th></th><th>Students</th><th>Teachers</th></tr>'+
  '<tr><td>At home</td><td>'+ (homeStu||'-') +'</td><td>'+ (homeTeach||'-') +'</td></tr>'+
  '<tr><td>Waiting</td><td>'+ (waitStu||'-') +'</td><td>'+ (waitTeach||'-') +'</td></tr>'+
  '</table>';
 const penGrouped={};
 (slot.penaltyDetails||[]).filter(p=>p.amount>0).forEach(p=>{(penGrouped[p.type]=penGrouped[p.type]||[]).push(p);});
 const types=Object.keys(penGrouped);
 if(types.length){
   html+='<h3>Penalties</h3><table class="info-table"><tr><th>Type</th><th>Amount</th><th>Who</th></tr>';
   types.forEach(t=>{
     const list=penGrouped[t];
     const amount=list.reduce((a,x)=>a+x.amount,0);
     if(amount>0){
      const names=list.map(p=>{
        const isTeach=p.type==='gapTeacher'||(p.type==='unoptimalSlot'&&teacherSet.has(p.name))||(p.type==='consecutiveClass'&&teacherSet.has(p.name));
        const role=isTeach?'teacher':'student';
        return personLink(p.name,role)+' ('+p.amount.toFixed(1)+')';
      }).join(', ');
       html+='<tr><td>'+t+'</td><td class="num">'+amount.toFixed(1)+'</td><td>'+names+'</td></tr>';
     }
   });
   html+='</table>';
 }
 openModal(html,!fromModal);
}

function computeTeacherStats(name){
 const info=(configData.teachers||[]).find(t=>t.name===name)||{};
 const defArr=(configData.defaults.teacherArriveEarly||[false])[0];
 const arrive=info.arriveEarly!==undefined?info.arriveEarly:defArr;
 let sizes=[],total=0,gap=0,time=0;
 scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const dayStart=slots.length?slots[0].slotIndex:0;
   const teachSlots=slots.filter(sl=>sl.classes.some(c=>(c.teachers||[]).includes(name)));
   if(teachSlots.length){
     const firstClass=teachSlots[0].slotIndex;
     const first=arrive?dayStart:firstClass;
     const last=teachSlots[teachSlots.length-1].slotIndex;
     time+=last-first+1;
     teachSlots.forEach(sl=>{const c=sl.classes.find(x=>(x.teachers||[]).includes(name));sizes.push(c.size);total++;});
     for(const sl of slots){if(sl.slotIndex>=first&&sl.slotIndex<=last){if(sl.gaps.teachers.includes(name))gap++;}}
   }
  });
 const avg=sizes.reduce((a,b)=>a+b,0)/(sizes.length||1);
 return{totalClasses:total,avgSize:avg.toFixed(1),gap:gap,time:time};
}

function computeStudentStats(name){
 const info=(configData.students||[]).find(s=>s.name===name)||{};
 const defArr=(configData.defaults.studentArriveEarly||[true])[0];
 const arrive=info.arriveEarly!==undefined?info.arriveEarly:defArr;
 let gap=0,time=0;
 scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const dayStart=slots.length?slots[0].slotIndex:0;
   const stSlots=slots.filter(sl=>sl.classes.some(c=>c.students.includes(name)));
   if(stSlots.length){
     const firstClass=stSlots[0].slotIndex;
     const first=arrive?dayStart:firstClass;
     const last=stSlots[stSlots.length-1].slotIndex;
     time+=last-first+1;
     for(const sl of slots){if(sl.slotIndex>=first&&sl.slotIndex<=last){if(sl.gaps.students.includes(name))gap++;}}
   }
 });
return{gap:gap,time:time};
}

function computeTeacherInfo(name){
 const info=(configData.teachers||[]).find(t=>t.name===name)||{};
 const defArr=(configData.defaults.teacherArriveEarly||[false])[0];
 const defImp=(configData.defaults.teacherImportance||[1])[0];
 const arrive=info.arriveEarly!==undefined?info.arriveEarly:defArr;
 const imp=info.importance!==undefined?info.importance:defImp;
 let hours=0,gap=0,time=0,subjects={},pen=0;
 scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const dayStart=slots.length?slots[0].slotIndex:0;
   const teachSlots=slots.filter(sl=>sl.classes.some(c=>(c.teachers||[]).includes(name)));
   if(teachSlots.length){
     const first=arrive?dayStart:teachSlots[0].slotIndex;
     const last=teachSlots[teachSlots.length-1].slotIndex;
     time+=last-first+1;
     for(const sl of slots){if(sl.slotIndex>=first&&sl.slotIndex<=last){if(sl.gaps.teachers.includes(name))gap++;}}
     teachSlots.forEach(sl=>{
       const cls=sl.classes.find(c=>(c.teachers||[]).includes(name));
       hours++;
       const stat=subjects[cls.subject]||{count:0,size:0};
       stat.count++;stat.size+=cls.size;subjects[cls.subject]=stat;
     });
   }
  slots.forEach(sl=>{(sl.penaltyDetails||[]).forEach(p=>{if(p.name===name)pen+=p.amount;});});
 });
 for(const k in subjects){subjects[k].avg=(subjects[k].size/subjects[k].count).toFixed(1);}
 return{arrive,imp,penalty:pen/imp,hours:hours,time:time,subjects};
}

function computeStudentInfo(name){
 const info=(configData.students||[]).find(s=>s.name===name)||{};
 const defArr=(configData.defaults.studentArriveEarly||[true])[0];
 const defImp=(configData.defaults.studentImportance||[0])[0];
 const arrive=info.arriveEarly!==undefined?info.arriveEarly:defArr;
 const imp=info.importance!==undefined?info.importance:defImp;
 let hours=0,gap=0,time=0,subjects={},pen=0;
 scheduleData.days.forEach(day=>{
   const slots=day.slots;
   const dayStart=slots.length?slots[0].slotIndex:0;
   const stSlots=slots.filter(sl=>sl.classes.some(c=>c.students.includes(name)));
   if(stSlots.length){
     const first=arrive?dayStart:stSlots[0].slotIndex;
     const last=stSlots[stSlots.length-1].slotIndex;
     time+=last-first+1;
     for(const sl of slots){if(sl.slotIndex>=first&&sl.slotIndex<=last){if(sl.gaps.students.includes(name))gap++;}}
     stSlots.forEach(sl=>{
       const cls=sl.classes.find(c=>c.students.includes(name));
       hours++;
       const stat=subjects[cls.subject]||{count:0,penalty:0};
       stat.count++;subjects[cls.subject]=stat;
     });
   }
   slots.forEach(sl=>{(sl.penaltyDetails||[]).forEach(p=>{
    if(p.name===name){
      pen+=p.amount;
      if(p.type==='unoptimalSlot'){
        const cls=sl.classes.find(c=>c.students.includes(name));
        if(cls){
          subjects[cls.subject]=subjects[cls.subject]||{count:0,penalty:0};
          subjects[cls.subject].penalty+=(p.amount/imp);
        }
      }
    }
  });});
});
return{arrive,imp,penalty:pen/imp,hours:hours,time:time,subjects};
}

function buildTeachers(){
 const cont=document.getElementById('teachers');
 cont.innerHTML='';
 const header=document.createElement('div');
 header.className='overview-header';
  header.innerHTML='<span class="person-name">Teacher</span><span class="person-info">Priority<br>Arrive</span><span class="person-pen">Penalty</span><span class="person-hours">Hours</span><span class="person-time">At school</span><span class="person-subjects">Subject<br>Cls | Avg</span>';
 cont.appendChild(header);
 const infos=(configData.teachers||[]).map(t=>{return{info:t,stat:computeTeacherInfo(t.name)}});
 infos.sort((a,b)=>b.stat.penalty-a.stat.penalty);
  infos.forEach(item=>{
   const row=document.createElement('div');
   row.className='overview-row';
   const arr=item.stat.arrive?"yes":"no";
   const pr=item.info.importance!==undefined?item.info.importance:(configData.defaults.teacherImportance||[1])[0];
   let subjHtml='';
   Object.keys(item.stat.subjects).forEach(sid=>{
     const s=item.stat.subjects[sid];
    const name=subjectDisplay[sid]||sid;
     subjHtml+='<div class="subject-line"><span class="subject-name clickable subject" data-id="'+sid+'">'+name+'</span>'+
       '<span class="subject-count">'+s.count+'</span>'+
       '<span class="subject-extra">'+s.avg+'</span></div>';
   });
   row.innerHTML='<span class="person-name clickable teacher" data-id="'+teacherIndex[item.info.name]+'">'+(teacherDisplay[item.info.name]||item.info.name)+'</span>'+
     '<span class="person-info">'+pr+'<br>'+arr+'</span>'+
     '<span class="person-pen">'+item.stat.penalty.toFixed(1)+'</span>'+
     '<span class="person-hours">'+item.stat.hours+'</span>'+
     '<span class="person-time">'+item.stat.time+'</span>'+
     '<span class="person-subjects"><div class="subject-list">'+subjHtml+'</div></span>';
   cont.appendChild(row);
  });
}

function buildStudents(){
 const cont=document.getElementById('students');
 cont.innerHTML='';
 const header=document.createElement('div');
 header.className='overview-header';
  header.innerHTML='<span class="person-name">Student</span><span class="person-info">Priority<br>Arrive</span><span class="person-pen">Penalty</span><span class="person-hours">Hours</span><span class="person-time">At school</span><span class="person-subjects">Subject<br>Cls | Pen</span>';
 cont.appendChild(header);
 const infos=(configData.students||[]).map(s=>{return{info:s,stat:computeStudentInfo(s.name)}});
 infos.sort((a,b)=>b.stat.penalty-a.stat.penalty);
  infos.forEach(item=>{
   const row=document.createElement('div');
   row.className='overview-row';
   const arr=item.stat.arrive?"yes":"no";
   const pr=item.info.importance!==undefined?item.info.importance:(configData.defaults.studentImportance||[0])[0];
   let subjHtml='';
   Object.keys(item.stat.subjects).forEach(sid=>{
     const s=item.stat.subjects[sid];
    const name=subjectDisplay[sid]||sid;
     subjHtml+='<div class="subject-line"><span class="subject-name clickable subject" data-id="'+sid+'">'+name+'</span>'+
       '<span class="subject-count">'+s.count+'</span>'+
       '<span class="subject-extra">'+(s.penalty||0).toFixed(1)+'</span></div>';
   });
   row.innerHTML='<span class="person-name clickable student" data-id="'+studentIndex[item.info.name]+'">'+item.info.name+'</span>'+
     '<span class="person-info">'+pr+'<br>'+arr+'</span>'+
     '<span class="person-pen">'+item.stat.penalty.toFixed(1)+'</span>'+
     '<span class="person-hours">'+item.stat.hours+'</span>'+
     '<span class="person-time">'+item.stat.time+'</span>'+
     '<span class="person-subjects"><div class="subject-list">'+subjHtml+'</div></span>';
   cont.appendChild(row);
  });
}

function showTeacher(idx,fromModal=false){
 const info=(configData.teachers||[])[idx]||{};
 const name=info.name||'';
 const display=teacherDisplay[name]||name;
 const defImp=(configData.defaults.teacherImportance||[1])[0];
 const imp=info.importance!==undefined?info.importance:defImp;
 const boldImp=imp!==defImp;
 const stats=computeTeacherStats(name);
 const full=computeTeacherInfo(name);
 const defArr=(configData.defaults.teacherArriveEarly||[false])[0];
 const boldArr=full.arrive!==defArr;
 let html='<h2>Teacher: '+display+'</h2>'+makeGrid(cls=>cls.teachers.includes(name));
 html+='<h3>Subjects</h3><table class="info-table"><tr><th>Subject</th><th>Classes</th><th>Avg size</th></tr>';
 Object.keys(full.subjects).forEach(sid=>{
   const s=full.subjects[sid];
   const sname=subjectDisplay[sid]||sid;
   html+='<tr><td><span class="clickable subject" data-id="'+sid+'">'+sname+'</span></td><td class="num">'+s.count+'</td><td class="num">'+s.avg+'</td></tr>';});
 html+='</table>';
 const params=[
 ['Priority',imp,boldImp],
 ['Arrive early',full.arrive?'yes':'no',boldArr],
 ['Gap hours',stats.gap],
 ['At school',stats.time],
 ['Total classes',stats.totalClasses],
 ['Average size',stats.avgSize],
 ['Penalty',full.penalty.toFixed(1)]
 ];
 html+='<h3>Configuration</h3>'+makeParamTable(params);
 openModal(html,!fromModal);
}

function showStudent(idx,fromModal=false){
 const info=(configData.students||[])[idx]||{};
 const name=info.name||'';
 const defImp=(configData.defaults.studentImportance||[0])[0];
 const imp=info.importance!==undefined?info.importance:defImp;
 const boldImp=imp!==defImp;
 const group=studentSize[name]||1;
 const boldGroup=group!==1;
 const stats=computeStudentStats(name);
 const full=computeStudentInfo(name);
 const defArr=(configData.defaults.studentArriveEarly||[true])[0];
 const boldArr=full.arrive!==defArr;
let html='<h2>Student: '+name+'</h2>'+makeGrid(cls=>cls.students.includes(name));
 html+='<h3>Subjects</h3><table class="info-table"><tr><th>Subject</th><th>Classes</th><th>Penalty</th></tr>';
Object.keys(full.subjects).forEach(sid=>{const s=full.subjects[sid];const sn=subjectDisplay[sid]||sid;html+='<tr><td><span class="clickable subject" data-id="'+sid+'">'+sn+'</span></td><td class="num">'+s.count+'</td><td class="num">'+(s.penalty||0).toFixed(1)+'</td></tr>';});
 html+='</table>';
 const params=[
 ['Group size',group,boldGroup],
 ['Priority',imp,boldImp],
 ['Arrive early',full.arrive?'yes':'no',boldArr],
 ['Gap hours',stats.gap],
 ['At school',stats.time],
 ['Penalty',full.penalty.toFixed(1)]
 ];
 html+='<h3>Configuration</h3>'+makeParamTable(params);
 openModal(html,!fromModal);
}

function showCabinet(name,fromModal=false){
 const info=configData.cabinets[name]||{};
  const disp=cabinetDisplay[name]||name;
  let html='<h2>Room: '+disp+'</h2>'+makeGrid(cls=>(cls.cabinets||[]).includes(name));
 const params=[
  ['Capacity',info.capacity||'-'],
  ['Allowed subjects',(info.allowedSubjects||[]).map(s=>subjectDisplay[s]||s).join(', ')||'-']
 ];
 html+='<h3>Configuration</h3>'+makeParamTable(params);
 openModal(html,!fromModal);
}

function showSubject(id,fromModal=false){
 const subj=configData.subjects[id]||{};
 const defOpt=(configData.defaults.optimalSlot||[0])[0];
 const disp=subjectDisplay[id]||id;
 let html='<h2>Subject: '+disp+'</h2>'+makeGrid(cls=>cls.subject===id);
html+='<h3>Teachers</h3><table class="info-table"><tr><th>Name</th></tr>';
(configData.teachers||[]).forEach((t,i)=>{if((t.subjects||[]).includes(id)){const bold=(subj.primaryTeachers||[]).includes(t.name);const nm=bold?'<strong>'+(teacherDisplay[t.name]||t.name)+'</strong>':(teacherDisplay[t.name]||t.name);html+='<tr><td><span class="clickable teacher" data-id="'+i+'">'+nm+'</span></td></tr>';}});
 html+='</table><h3>Students</h3><table class="info-table"><tr><th>Name</th><th>Group</th></tr>';
 (configData.students||[]).forEach((s,i)=>{if((s.subjects||[]).includes(id)){html+='<tr><td><span class="clickable student" data-id="'+i+'">'+s.name+'</span></td><td class="num">'+(studentSize[s.name]||1)+'</td></tr>';}});
 html+='</table>';
 const defPerm=(configData.defaults.permutations||[true])[0];
 const defAvoid=(configData.defaults.avoidConsecutive||[true])[0];
 const opt=subj.optimalSlot!==undefined?subj.optimalSlot:defOpt;
 const boldOpt=opt!==defOpt;
 const perm=subj.allowPermutations!==undefined?subj.allowPermutations:defPerm;
 const boldPerm=perm!==defPerm;
 const avoid=subj.avoidConsecutive!==undefined?subj.avoidConsecutive:defAvoid;
 const boldAvoid=avoid!==defAvoid;
  const req=subj.requiredTeachers!==undefined?subj.requiredTeachers:1;
  const boldReq=req!==1;
  const reqCab=subj.requiredCabinets!==undefined?subj.requiredCabinets:1;
  const boldReqCab=reqCab!==1;
  const params=[
  ['Classes',(subj.classes||[]).join(', ')||'-'],
  ['Optimal lesson',opt+1,boldOpt],
  ['Allow permutations',perm?'yes':'no',boldPerm],
  ['Avoid consecutive',avoid?'yes':'no',boldAvoid],
   ['Required teachers',req,boldReq],
   ['Required cabinets',reqCab,boldReqCab],
   ['Cabinets',(subj.cabinets||[]).map(c=>cabinetDisplay[c]||c).join(', ')||'-'],
  ['Primary teachers',(subj.primaryTeachers||[]).map(t=>teacherDisplay[t]||t).join(', ')||'-']
 ];
 html+='<h3>Configuration</h3>'+makeParamTable(params);
 openModal(html,!fromModal);
}

document.addEventListener('click',e=>{
 const fromModal=modal.contains(e.target);
 const slotElem=e.target.closest('.slot-info');
 if(slotElem){
   showSlot(slotElem.dataset.day,parseInt(slotElem.dataset.slot),fromModal);
   return;
 }
 const target=e.target.closest('.subject,.teacher,.student,.cabinet');
 if(!target)return;
 if(target.classList.contains('subject')){showSubject(target.dataset.id,fromModal);}
 else if(target.classList.contains('teacher')){showTeacher(parseInt(target.dataset.id),fromModal);}
 else if(target.classList.contains('student')){showStudent(parseInt(target.dataset.id),fromModal);}
 else if(target.classList.contains('cabinet')){showCabinet(target.dataset.id,fromModal);}
});

buildTable();
buildTeachers();
buildStudents();
})();
</script>
</body>
</html>
"""
    meta = ""
    if generated:
        meta = f"<div class=\"meta\">Generated: {generated}"
        if include_config:
            meta += " | <span id=\"show-config\" class=\"clickable\">View config</span>"
        meta += "</div>"
    html = (
        html.replace("__SCHEDULE__", schedule_json)
        .replace("__CONFIG__", cfg_json)
        .replace("__META__", meta)
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)

