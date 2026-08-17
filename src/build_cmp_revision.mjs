import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile, Workbook } from "@oai/artifact-tool";

const ROOT = "C:/Users/user/Documents/Codex/2026-08-01/s";
const DATA = path.join(ROOT, "outputs/cmp_mrr_analysis_20260809/revision");
const OUT = path.join(ROOT, "outputs/cmp_mrr_analysis_20260809/CMP_MRR_예측모델_및_Sensor조건_개정본.pptx");
const RENDER = path.join(ROOT, "work/cmp_mrr_analysis/revision/rendered");
const C = { navy:"#0B1F3A", navy2:"#17365D", coral:"#F26B4E", teal:"#2A7F8E", gray:"#F3F5F7", grid:"#D7DEE7", text:"#202B3C", muted:"#657386", white:"#FFFFFF", paleCoral:"#FFF2EE", paleTeal:"#EAF5F6", red:"#B83A2F", green:"#2E7D5B" };
const FONT = "Pretendard";
const W=1280, H=720;

async function csv(name){
  const raw=await fs.readFile(path.join(DATA,name),"utf8");
  const wb=await Workbook.fromCSV(raw.replace(/^\uFEFF/,""),{sheetName:"Import"});
  const v=wb.worksheets.getItem("Import").getUsedRange(true).values;
  const h=v[0]; return v.slice(1).map(r=>Object.fromEntries(h.map((x,i)=>[String(x),r[i]])));
}
function rect(slide,name,x,y,w,h,fill="none",line="none",radius="rounded-md"){
  const o={geometry:radius.startsWith("rounded")?"roundRect":"rect",name,position:{left:x,top:y,width:w,height:h},fill,line:{style:"solid",fill:line,width:line==="none"?0:1}};
  if(radius.startsWith("rounded")) o.borderRadius=radius;
  return slide.shapes.add(o);
}
function tx(slide,name,value,x,y,w,h,size=20,color=C.text,bold=false,align="left"){
  const s=slide.shapes.add({geometry:"textbox",name,position:{left:x,top:y,width:w,height:h},fill:"none",line:{style:"solid",fill:"none",width:0}});
  s.text=String(value); s.text.style={fontFamily:FONT,fontSize:size,color,bold,alignment:align,verticalAlignment:"middle"}; return s;
}
function header(slide,title,takeaway,page){
  tx(slide,`title-${page}`,title,64,35,950,52,38,C.navy,true);
  tx(slide,`take-${page}`,takeaway,64,88,1120,32,18,C.coral,true);
  rect(slide,`rule-${page}`,64,132,1152,2,C.grid,"none","rect");
  tx(slide,`page-${page}`,String(page).padStart(2,"0"),1155,43,60,24,14,C.muted,true,"right");
}
function footer(slide,note="분석값은 공개 CMP 데이터의 관측 결과이며 장비 Setpoint를 의미하지 않음"){
  tx(slide,"footer",note,64,686,920,18,12,C.muted,false);
  tx(slide,"brand","CMP MRR ANALYSIS",1010,686,206,18,12,C.navy,true,"right");
}
function notes(slide,sources){ slide.speakerNotes.textFrame.setText(`[Sources]\n${sources.map(s=>`- ${s}`).join("\n")}`); }
function bullet(slide,items,x,y,w,size=18,gap=48){ items.forEach((v,i)=>{rect(slide,`dot-${x}-${y}-${i}`,x,y+i*gap+14,8,8,i===0?C.coral:C.teal,"none","rect");tx(slide,`bul-${x}-${y}-${i}`,v,x+22,y+i*gap,w-22,gap-4,size,C.text,false);}); }
function metric(slide,x,y,w,label,value,color=C.navy){rect(slide,`m-${label}`,x,y,w,112,C.white,C.grid);tx(slide,`ml-${label}`,label,x+16,y+14,w-32,24,15,C.muted,true);tx(slide,`mv-${label}`,value,x+16,y+44,w-32,48,30,color,true);}
function sectionLabel(slide,label,x,y,w=250){tx(slide,`sect-${label}-${x}`,label,x,y,w,28,18,C.navy,true);rect(slide,`sectline-${label}-${x}`,x,y+34,w,2,C.teal,"none","rect");}
function labelBox(slide,x,y,w,label,value,fill=C.gray){rect(slide,`lb-${label}-${x}`,x,y,w,74,fill,"none");tx(slide,`lbl-${label}-${x}`,label,x+14,y+8,w-28,22,14,C.muted,true);tx(slide,`lbv-${label}-${x}`,value,x+14,y+31,w-28,30,22,C.navy,true);}
function drawBoxPlot(slide,name,x,y,w,h,rows,labelKey,title){
  rect(slide,`${name}-frame`,x,y,w,h,C.white,C.grid);
  tx(slide,`${name}-title`,title,x,y-29,w,26,16,C.navy,true,"center");
  const vals=rows.flatMap(r=>[Number(r.LOWER_WHISKER),Number(r.UPPER_WHISKER)]).filter(Number.isFinite);
  const rawMin=Math.min(...vals); let min=rawMin, max=Math.max(...vals); if(max<=min){max=min+1;} const pad=(max-min)*0.06; min=rawMin>=0?Math.max(0,min-pad):min-pad; max+=pad;
  const left=x+82,right=x+w-18,plotW=right-left; const sx=v=>left+(Number(v)-min)/(max-min)*plotW;
  const rowH=(h-56)/rows.length;
  rows.forEach((r,i)=>{
    const cy=y+26+i*rowH+rowH/2; const q1=sx(r.Q1),med=sx(r.MEDIAN),q3=sx(r.Q3),lo=sx(r.LOWER_WHISKER),hi=sx(r.UPPER_WHISKER);
    tx(slide,`${name}-lab-${i}`,r[labelKey],x+7,cy-12,70,24,11,C.muted,true,"right");
    rect(slide,`${name}-wh-${i}`,lo,cy-1,Math.max(1,hi-lo),2,C.muted,"none","rect");
    rect(slide,`${name}-capl-${i}`,lo-1,cy-10,2,20,C.muted,"none","rect");rect(slide,`${name}-caph-${i}`,hi-1,cy-10,2,20,C.muted,"none","rect");
    rect(slide,`${name}-box-${i}`,q1,cy-14,Math.max(3,q3-q1),28,i%2?C.paleTeal:C.paleCoral,C.teal,"none");
    rect(slide,`${name}-med-${i}`,med-1,cy-14,3,28,C.coral,"none","rect");
    if(Number(r.OUTLIER_COUNT)>0) tx(slide,`${name}-out-${i}`,`후보 ${r.OUTLIER_COUNT}`,Math.min(hi+3,right-42),cy-10,42,20,9,C.red,true,"right");
  });
  tx(slide,`${name}-min`,min.toFixed(Math.abs(max)<20?1:0),left,y+h-25,55,18,10,C.muted,false,"left");tx(slide,`${name}-max`,max.toFixed(Math.abs(max)<20?1:0),right-55,y+h-25,55,18,10,C.muted,false,"right");
}
function groupedBar(slide,name,x,y,w,h,categories,series,yTitle,min,max){
  slide.charts.add("bar",{name,position:{left:x,top:y,width:w,height:h},categories,series,barOptions:{direction:"column",grouping:"clustered",gapWidth:75},hasLegend:true,legend:{position:"bottom",overlay:false,textStyle:{fontSize:12,fill:C.muted}},xAxis:{textStyle:{fontSize:12,fill:C.muted},line:{style:"solid",fill:C.grid,width:1}},yAxis:{title:yTitle,min,max,numberFormatCode:"0.0",textStyle:{fontSize:11,fill:C.muted},majorGridlines:{style:"solid",fill:C.grid,width:1}},chartFill:C.white,plotAreaFill:C.white,chartLine:{style:"solid",fill:C.grid,width:1}});
}
function tableRow(slide,y,cells,widths,styles={}){let x=styles.x??64; cells.forEach((v,i)=>{const fill=styles.header?C.navy:(styles.fill??C.white);rect(slide,`cell-${y}-${i}-${x}`,x,y,widths[i],styles.h??42,fill,styles.header?C.navy:C.grid,"none");tx(slide,`ct-${y}-${i}-${x}`,v,x+8,y+3,widths[i]-16,(styles.h??42)-6,styles.size??15,styles.header?C.white:(styles.color??C.text),styles.header||styles.bold,(styles.aligns?.[i]??"left"));x+=widths[i];});}

async function main(){
  await fs.mkdir(RENDER,{recursive:true});
  const [scope,targets,models,tests,importance,sensorStats,candidates,preds,mrrStats]=await Promise.all([
    csv("sample_scope.csv"),csv("target_mrr_by_regime.csv"),csv("model_comparison_by_regime.csv"),csv("test_metrics_by_regime.csv"),csv("sensor_importance_by_regime.csv"),csv("sensor_boxplot_stats.csv"),csv("sensor_condition_candidates.csv"),csv("test_predictions_by_regime.csv"),csv("mrr_boxplot_stats.csv")]);
  const p=Presentation.create({slideSize:{width:W,height:H}});
  const local="Local analysis outputs under outputs/cmp_mrr_analysis_20260809/revision";
  const official="PHM Society, 2016 PHM Data Challenge Call for Participation (PHM16DataChallengeCFP.pdf)";

  // 1. White title
  {const s=p.slides.add();s.background.fill=C.white;rect(s,"accent",64,76,86,5,C.coral,"none","rect");tx(s,"kicker","2016 PHM CMP DATA CHALLENGE",64,94,420,28,14,C.muted,true);tx(s,"deck-title","CMP MRR(Material Removal Rate, 연마 제거율)\n예측 모델 개발 및 목표 Sensor 조건 도출",64,180,990,170,52,C.navy,true);tx(s,"sub","Stage–Chamber별 MRR 목표·영향 Sensor·예측 성능을 분리 평가",64,378,1000,42,22,C.teal,true);tx(s,"date","2026.08.10",64,650,200,24,14,C.muted,false);notes(s,[official,local]);}

  // 2. Goal
  {const s=p.slides.add();s.background.fill=C.white;header(s,"분석 목표","Target MRR 설정 → 예측 모델 비교 → Sensor 조건 후보 도출",2);
    const xs=[64,443,822]; const nums=["01","02","03"]; const heads=["Target MRR 설정","MRR 예측 모델 개발","Sensor 조건 후보 도출"]; const bod=["Stage–Chamber별 과거 MRR 중앙값을 분석 목표로 사용","동일 공정 조건 안에서 여러 회귀모델의 오차와 설명력을 비교","목표 MRR 구간에서 실제 관측된 Sensor 분포를 후보 범위로 제시"];
    for(let i=0;i<3;i++){tx(s,`num-${i}`,nums[i],xs[i],190,100,54,36,C.coral,true);tx(s,`head-${i}`,heads[i],xs[i],252,320,42,24,C.navy,true);tx(s,`body-${i}`,bod[i],xs[i],310,315,125,18,C.text,false);if(i<2){rect(s,`arr-${i}`,xs[i]+330,274,30,3,C.grid,"none","rect");}}
    rect(s,"scope",64,515,1138,98,C.paleTeal,"none");tx(s,"scope-h","분석 원칙",86,531,150,25,17,C.teal,true);tx(s,"scope-t","Stage와 Chamber가 다른 데이터를 섞지 않고, 각 조건 안에서 Sensor–MRR 관계를 검증합니다. 관측 범위는 최적 Setpoint가 아니며 DOE 확인이 필요합니다.",86,560,1080,39,18,C.text,false);footer(s);notes(s,[official,local]);}

  // 3. Agenda
  {const s=p.slides.add();s.background.fill=C.white;header(s,"목차","데이터 처리부터 적용 한계까지 7개 항목으로 구성",3);const items=["1. 데이터 전처리","2. Target MRR 설정","3. Sensor 영향 분석","4. 예측모델 비교","5. 최종 예측 성능","6. 목표 Sensor 조건","7. 결론 및 적용 한계"];items.forEach((v,i)=>{const y=166+i*66;tx(s,`agenda-${i}`,v,112,y,500,42,23,i<3?C.navy:C.text,true);rect(s,`aline-${i}`,650,y+21,510,1,C.grid,"none","rect");tx(s,`apage-${i}`,String(i+1).padStart(2,"0"),1160,y,40,42,14,C.muted,true,"right");});footer(s);notes(s,[local]);}

  // 4. preprocessing
  {const s=p.slides.add();s.background.fill=C.white;header(s,"1. 데이터 전처리","동일 Wafer와 Stage의 시계열 Sensor 값을 한 행으로 요약",4);
    sectionLabel(s,"원본 데이터",64,166,330);bullet(s,["식별 정보: WAFER_ID, STAGE, CHAMBER","공정 Sensor: 압력, Slurry 유량, 회전수","사용량: Pad·Dresser 누적 사용량","정답값: AVG_REMOVAL_RATE(MRR)"],64,216,480,17,52);
    sectionLabel(s,"처리 방법",640,166,330);bullet(s,["같은 WAFER_ID와 STAGE의 연속 측정값을 통합","Sensor별 평균·표준편차·최댓값−최솟값 계산","결측값 0건: 별도 대체 없이 원자료 유지","수치형 MRR 예측이므로 회귀모델 적용"],640,216,530,17,52);
    rect(s,"why",64,475,1106,128,C.paleCoral,"none");tx(s,"why-h","왜 평균만 사용하지 않는가",86,492,260,27,18,C.coral,true);tx(s,"why-t","평균은 공정의 대표 수준, 표준편차와 최댓값−최솟값은 공정 중 흔들림을 나타냅니다. 따라서 Sensor 수준과 변동을 함께 모델 입력값으로 사용했습니다.",86,527,1035,56,18,C.text,false);footer(s);notes(s,[official,local]);}

  // 5 scope
  {const s=p.slides.add();s.background.fill=C.white;header(s,"1-1. 분석 대상 선정","표본이 충분한 3개 조건만 본 분석에 포함",5);
    tableRow(s,170,["공정 조건","표본 수","본 분석","판단"],[330,170,170,440],{header:true,h:45,size:16,aligns:["left","center","center","left"]});
    const inc=scope.filter(r=>String(r.USE_IN_MAIN_ANALYSIS).toLowerCase()==="true");const exc=scope.filter(r=>String(r.USE_IN_MAIN_ANALYSIS).toLowerCase()!=="true");
    inc.forEach((r,i)=>tableRow(s,215+i*47,[`Stage ${r.STAGE}–Chamber ${Number(r.CHAMBER)}`,String(r.N),"포함","조건별 모델·Target·Sensor 분석"],[330,170,170,440],{h:47,size:16,aligns:["left","center","center","left"],bold:i===0}));
    tableRow(s,356,["기타 Stage–Chamber",String(exc.reduce((a,r)=>a+Number(r.N),0)),"제외","표본 부족: 일반화 가능한 비교가 어려움"],[330,170,170,440],{h:47,size:16,aligns:["left","center","center","left"],fill:C.paleCoral,color:C.red,bold:true});
    rect(s,"review",64,442,1110,140,C.gray,"none");tx(s,"review-h","추가 검토",86,458,200,25,18,C.navy,true);tx(s,"review-t","MRR 4건은 전체 분포에서 매우 멀리 떨어져 통계적 검토 대상으로 분리했습니다. 측정 오류로 확정할 근거는 없으므로 원자료에서는 삭제하지 않았으며, 본 모델 비교에서만 민감도 확인을 위해 제외했습니다.",86,493,1040,66,18,C.text,false);footer(s,"표본 부족 조건 9개는 본 분석에서 제외; MRR 검토 후보 4개는 오류 확정이 아님");notes(s,[local]);}

  // 6 target
  {const s=p.slides.add();s.background.fill=C.white;header(s,"2. Target MRR 설정","조건별 과거 중앙값을 분석용 Target으로 적용",6);
    drawBoxPlot(s,"mrr-box",64,180,700,350,mrrStats.map(r=>({...r,SHORT:r.REGIME.replace("Stage A–Chamber 1","A–Ch1").replace("Stage A–Chamber 4","A–Ch4").replace("Stage B–Chamber 4","B–Ch4")})),"SHORT","조건별 MRR 분포");
    targets.forEach((r,i)=>labelBox(s,800,174+i*95,350,`${r.REGIME}`,`${Number(r.TARGET_MRR).toFixed(2)}  (Q1–Q3 ${Number(r.TARGET_BAND_LOWER).toFixed(2)}–${Number(r.TARGET_BAND_UPPER).toFixed(2)})`,i===0?C.paleTeal:C.gray));
    rect(s,"box-rule",800,478,350,126,C.paleCoral,"none");tx(s,"br-h","Box Plot 판독 기준",818,490,310,24,16,C.coral,true);tx(s,"br-t","상자: Q1–Q3 / 선: 중앙값\n수염: 1.5×IQR 안의 최솟값·최댓값\n수염 밖 점: 이상치 후보",818,520,310,70,15,C.text,false);footer(s,"Target은 제품 Spec이 아닌 과거 데이터 기반 분석 기준");notes(s,[local]);}

  // 7 method
  {const s=p.slides.add();s.background.fill=C.white;header(s,"3. Sensor 영향 분석","Stage–Chamber를 고정한 뒤 Sensor와 MRR의 관계를 비교",7);
    const cols=[64,430,796];const heads=["독립변수(X)","종속변수(Y)","검증 방법"];const bod=["압력·유량·회전수의 평균, 표준편차, 범위","각 Wafer–Stage의 평균 MRR","순열중요도 + MRR 사분위별 Box Plot"];
    for(let i=0;i<3;i++){tx(s,`mh-${i}`,heads[i],cols[i],185,300,30,22,C.navy,true);rect(s,`mline-${i}`,cols[i],225,300,3,i===1?C.coral:C.teal,"none","rect");tx(s,`mb-${i}`,bod[i],cols[i],252,300,95,18,C.text,false);}
    rect(s,"perm",64,400,1100,154,C.gray,"none");tx(s,"perm-h","순열중요성(Permutation Importance)",86,416,430,28,18,C.navy,true);tx(s,"perm-t","한 Sensor 값을 임의로 섞었을 때 예측 성능이 얼마나 악화되는지 측정합니다. 성능 감소가 클수록 모델이 해당 Sensor에 더 의존했다는 뜻입니다.",86,450,1025,48,18,C.text,false);tx(s,"perm-c","주의: 중요도는 인과관계가 아니며, 상관된 Sensor 사이에서는 중요도가 나뉠 수 있습니다.",86,510,1025,28,16,C.red,true);footer(s);notes(s,[local]);}

  // 8-10 condition sensor boxes
  const topSensors={"Stage A–Chamber 1":["HEAD_ROTATION","RETAINER_RING_PRESSURE","CENTER_AIR_BAG_PRESSURE"],"Stage A–Chamber 4":["EDGE_AIR_BAG_PRESSURE","SLURRY_FLOW_LINE_A","SLURRY_FLOW_LINE_C"],"Stage B–Chamber 4":["SLURRY_FLOW_LINE_A","RETAINER_RING_PRESSURE","CENTER_AIR_BAG_PRESSURE"]};
  const takeaways={"Stage A–Chamber 1":"HEAD_ROTATION의 영향이 가장 크지만 최종 R²가 낮아 해석에 주의","Stage A–Chamber 4":"공기압과 Slurry 유량이 MRR 차이를 설명하는 주요 변수","Stage B–Chamber 4":"Slurry 유량과 압력 변수가 MRR 예측에 가장 크게 기여"};
  let page=8;
  for(const reg of Object.keys(topSensors)){
    const s=p.slides.add();s.background.fill=C.white;header(s,`3-${page-7}. ${reg.replace("Stage ","Stage ")}`,takeaways[reg],page);
    const sens=topSensors[reg]; sens.forEach((sn,i)=>drawBoxPlot(s,`box-${page}-${i}`,64+i*382,205,350,330,sensorStats.filter(r=>r.REGIME===reg&&r.SENSOR===sn),"MRR_QUARTILE",sn));
    tx(s,`quart-${page}`,"MRR 그룹: Q1 낮음 → Q4 높음",64,551,500,24,15,C.muted,true);
    const msg=reg==="Stage A–Chamber 1"?"압력 평균값에 0이 많이 포함되어 단순 평균 범위만으로 운전 조건을 정하기 어렵습니다. A–Ch1은 HEAD_ROTATION을 우선 검토하되 추가 데이터가 필요합니다.":"Box Plot의 그룹 차이는 관측상 연관성을 보여주며, 최적 조건 확정 전 동일 조건 DOE가 필요합니다.";
    rect(s,`cond-${page}`,64,590,1114,66,reg==="Stage A–Chamber 1"?C.paleCoral:C.paleTeal,"none");tx(s,`condt-${page}`,msg,84,602,1070,42,16,reg==="Stage A–Chamber 1"?C.red:C.text,true);footer(s);notes(s,[local]);page++;
  }

  // 11 model comparison
  {const s=p.slides.add();s.background.fill=C.white;header(s,"4. 예측모델 비교","RMSE는 낮을수록, R²는 1에 가까울수록 우수",11);
    const order=["Ridge","Random Forest","Extra Trees","HistGradientBoosting","XGBoost","CatBoost"];const regs=["Stage A–Chamber 1","Stage A–Chamber 4","Stage B–Chamber 4"];
    const rm=regs.map((r,i)=>({name:r.replace("Stage ",""),values:order.map(m=>Number(models.find(x=>x.REGIME===r&&x.MODEL===m)?.VALID_RMSE)),fill:[C.navy2,C.teal,C.coral][i]}));
    const rr=regs.map((r,i)=>({name:r.replace("Stage ",""),values:order.map(m=>Number(models.find(x=>x.REGIME===r&&x.MODEL===m)?.VALID_R2)),fill:[C.navy2,C.teal,C.coral][i]}));
    groupedBar(s,"rmse",64,176,555,350,order,rm,"검증 RMSE",0,8);groupedBar(s,"r2",650,176,555,350,order,rr,"검증 R²",0,1);
    rect(s,"metric-def",64,550,1140,88,C.gray,"none");tx(s,"metric-def-t","RMSE: 큰 오차에 더 큰 불이익을 주는 평균 예측오차  |  MAE: 평균 절대오차  |  R²: 실제 MRR 변동 중 모델이 설명한 비율",84,566,1100,30,16,C.text,true);tx(s,"metric-pick","선정: A–Ch1 Random Forest / A–Ch4 Random Forest / B–Ch4 XGBoost",84,602,1100,25,17,C.coral,true);footer(s);notes(s,[local]);}

  // 12 hyperparameters
  {const s=p.slides.add();s.background.fill=C.white;header(s,"4-1. 하이퍼파라미터 결정","3-Fold 교차검증의 평균 RMSE가 가장 낮은 조합을 선택",12);
    tx(s,"cv-h","교차검증 절차",64,168,280,30,22,C.navy,true);bullet(s,["학습 데이터를 3개 묶음으로 분할","2개 묶음으로 학습하고 1개 묶음으로 확인","확인 묶음을 바꿔 총 3회 반복","평균 RMSE가 가장 낮은 후보 조합 선택"],64,216,480,17,52);
    tableRow(s,170,["공정 조건","선정 모델","선정 파라미터"],[170,130,290],{x:610,header:true,h:45,size:14});
    const selected=tests.map(r=>[r.REGIME,r.MODEL,r.REGIME==="Stage B–Chamber 4"?"트리 200 / 최대 깊이 4 / 학습률 0.1 / 행 표본 80%":"트리 300 / 최대 깊이 10 / 사용 변수 90% / 말단 최소 표본 1"]);
    selected.forEach((r,i)=>tableRow(s,215+i*75,r,[170,130,290],{x:610,h:75,size:13,fill:i%2?C.gray:C.white}));
    rect(s,"hp-limit",610,465,590,112,C.paleCoral,"none");tx(s,"hp-limit-h","해석 범위",632,480,170,24,17,C.coral,true);tx(s,"hp-limit-t","계산 가능한 후보 조합 안에서 최적값을 선택했습니다. 모든 가능한 조합을 탐색한 전역 최적값은 아닙니다.",632,512,540,47,16,C.text,false);footer(s);notes(s,[local]);}

  // 13 independent test
  {const s=p.slides.add();s.background.fill=C.white;header(s,"5. 최종 예측 성능","독립 시험 데이터에서 A–Ch4와 B–Ch4의 설명력이 높음",13);
    const regs=["Stage A–Chamber 1","Stage A–Chamber 4","Stage B–Chamber 4"];
    regs.forEach((reg,i)=>{const rows=preds.filter(r=>r.REGIME===reg);const xs=rows.map(r=>Number(r.AVG_REMOVAL_RATE));const ys=rows.map(r=>Number(r.PREDICTED_MRR));const lo=Math.min(...xs,...ys),hi=Math.max(...xs,...ys);s.charts.add("scatter",{name:`sc-${i}`,position:{left:64+i*382,top:176,width:350,height:300},series:[{name:"예측",xValues:xs,values:ys,marker:{symbol:"circle",size:4},fill:[C.navy2,C.teal,C.coral][i]},{name:"일치선",xValues:[lo,hi],values:[lo,hi],line:{style:"solid",fill:C.muted,width:2},marker:{symbol:"none",size:2}}],scatterOptions:{style:"marker"},hasLegend:false,xAxis:{title:"실제 MRR",textStyle:{fontSize:10,fill:C.muted},majorGridlines:{style:"solid",fill:C.grid,width:1}},yAxis:{title:"예측 MRR",textStyle:{fontSize:10,fill:C.muted},majorGridlines:{style:"solid",fill:C.grid,width:1}},chartFill:C.white,plotAreaFill:C.white,chartLine:{style:"solid",fill:C.grid,width:1}});tx(s,`sc-title-${i}`,reg,64+i*382,145,350,28,17,C.navy,true,"center");const t=tests.find(r=>r.REGIME===reg);labelBox(s,64+i*382,495,350,`${t.MODEL} · 시험 ${t.N_TEST}개`,`RMSE ${Number(t.RMSE).toFixed(2)} | MAE ${Number(t.MAE).toFixed(2)} | R² ${Number(t.R2).toFixed(3)}`,i===0?C.paleCoral:C.paleTeal);});
    tx(s,"test-note","A–Ch1은 R² 0.301로 예측 설명력이 제한적이므로 추가 Sensor·Recipe 정보 확보 전 운전 판단에 직접 사용하지 않습니다.",64,600,1120,38,17,C.red,true);footer(s);notes(s,[local]);}

  // 14 conditions
  {const s=p.slides.add();s.background.fill=C.white;header(s,"6. 목표 Sensor 조건 후보","목표 MRR 중심 50%에서 관측된 Sensor 범위를 조건 후보로 제시",14);
    tableRow(s,165,["공정 조건","우선 Sensor","관측 Q1","중앙값","관측 Q3"],[270,340,160,160,160],{header:true,h:44,size:15,aligns:["left","left","right","right","right"]});
    const chosen=[candidates.find(r=>r.REGIME==="Stage A–Chamber 1"&&r.SENSOR==="HEAD_ROTATION"),...candidates.filter(r=>r.REGIME==="Stage A–Chamber 4").slice(0,3),...candidates.filter(r=>r.REGIME==="Stage B–Chamber 4").slice(0,3)].filter(Boolean);
    chosen.forEach((r,i)=>tableRow(s,209+i*45,[r.REGIME,r.SENSOR,Number(r.SENSOR_CANDIDATE_LOWER_Q1).toFixed(2),Number(r.SENSOR_CANDIDATE_MEDIAN).toFixed(2),Number(r.SENSOR_CANDIDATE_UPPER_Q3).toFixed(2)],[270,340,160,160,160],{h:45,size:14,fill:i%2?C.gray:C.white,aligns:["left","left","right","right","right"]}));
    rect(s,"cond-limit",64,550,1090,95,C.paleCoral,"none");tx(s,"cond-limit-h","적용 전 확인",84,563,180,24,17,C.coral,true);tx(s,"cond-limit-t","공개 데이터는 Sensor 단위와 Recipe 설정값을 공개하지 않습니다. 위 값은 목표 MRR 부근에서 관측된 운전 범위이며, 최적 Setpoint는 DOE로 확인해야 합니다.",84,594,1035,39,16,C.text,false);footer(s);notes(s,[official,local]);}

  // 15 white close
  {const s=p.slides.add();s.background.fill=C.white;tx(s,"last-sec","7. 결론 및 적용 한계",64,67,560,44,36,C.navy,true);rect(s,"last-accent",64,126,90,5,C.coral,"none","rect");
    tx(s,"last-main","공정 조건별 Target·모델·Sensor 후보를\n분리하여 평가해야 합니다.",64,177,850,110,42,C.navy,true);
    bullet(s,["Target MRR: A–Ch1 151.07 / A–Ch4 74.12 / B–Ch4 81.51","선정 모델: A–Ch1·A–Ch4 Random Forest / B–Ch4 XGBoost","A–Ch1은 최종 R² 0.301로 추가 데이터 없이 적용하기 어려움","Sensor 조건은 관측 후보이며 DOE·공정 Spec 확인 후 확정"],64,330,1030,19,58);
    tx(s,"last-limit","한계: 단일 Machine, 비공개 Sensor 단위·Recipe, 관측자료 기반 분석으로 인과관계 확정 불가",64,606,1100,34,17,C.red,true);tx(s,"last-date","CMP MRR ANALYSIS · 2026.08.10",64,662,360,20,13,C.muted,true);notes(s,[official,local]);}

  // Per-slide renders, layout exports, montage and PPTX
  for(const [i,s] of p.slides.items.entries()){
    const stem=`slide-${String(i+1).padStart(2,"0")}`;
    const png=await p.export({slide:s,format:"png",scale:1});await fs.writeFile(path.join(RENDER,`${stem}.png`),new Uint8Array(await png.arrayBuffer()));
    const lay=await s.export({format:"layout"});await fs.writeFile(path.join(RENDER,`${stem}.layout.json`),await lay.text());
  }
  const montage=await p.export({format:"webp",montage:true,scale:1});await fs.writeFile(path.join(RENDER,"montage.webp"),new Uint8Array(await montage.arrayBuffer()));
  const pptx=await PresentationFile.exportPptx(p);await pptx.save(OUT);
  const inspect=await p.inspect({kind:"slide,textbox,shape,chart,notes",maxChars:200000});await fs.writeFile(path.join(RENDER,"final.inspect.ndjson"),inspect.ndjson);
  console.log(OUT);
}
main().catch(e=>{console.error(e);process.exitCode=1;});
