import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const run=path.dirname(fileURLToPath(import.meta.url));
const cfg=JSON.parse(await fs.readFile(path.join(run,'config.json'),'utf8'));
const data=JSON.parse(await fs.readFile(path.join(run,'workbook_input.json'),'utf8'));
const out=cfg.figures_dir;
const wb=Workbook.create();
const sheets=Object.fromEntries(['Summary','Tests','BR','Cells'].map(n=>[n,wb.worksheets.add(n)]));
const colors={header:'#203748',band:'#EDF3F7',ink:'#172E40',muted:'#576A78'};
const porder=cfg.proteins, corder=cfg.conditions;
const sortGroup=(a,b)=>porder.indexOf(a.protein)-porder.indexOf(b.protein)||corder.indexOf(a.condition)-corder.indexOf(b.condition)||String(a.BR||'').localeCompare(String(b.BR||''),undefined,{numeric:true})||String(a.cell||'').localeCompare(String(b.cell||''),undefined,{numeric:true});
const cells=data.cells.sort(sortGroup);
const brs=data.br_metrics.filter(x=>x.endpoint==='on_percentage').sort(sortGroup);
const conditions=data.condition_metrics.filter(x=>x.endpoint==='on_percentage').sort(sortGroup);
function metric(rows,p,c,b,e){return rows.find(r=>r.protein===p&&r.condition===c&&(!b||r.BR===b)&&r.endpoint===e);}
function style(sheet,lastRow,lastCol){
  sheet.showGridLines=false;
  sheet.getRangeByIndexes(0,0,lastRow,lastCol).format.font={name:'Arial',size:11,color:colors.ink};
  sheet.getRangeByIndexes(0,0,lastRow,lastCol).format.rowHeight=23;
  sheet.getRangeByIndexes(0,0,lastRow,lastCol).format.verticalAlignment='center';
}
function table(sheet,row,headers,values,widths){
  sheet.getRangeByIndexes(row-1,0,1,headers.length).values=[headers];
  if(values.length)sheet.getRangeByIndexes(row,0,values.length,headers.length).values=values;
  const header=sheet.getRangeByIndexes(row-1,0,1,headers.length);
  header.format={fill:colors.header,font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:43};
  for(let i=0;i<values.length;i++)if(i%2===1)sheet.getRangeByIndexes(row+i,0,1,headers.length).format.fill=colors.band;
  for(let j=0;j<widths.length;j++)sheet.getRangeByIndexes(0,j,row+values.length+2,1).format.columnWidth=widths[j];
  if(values.length)sheet.getRangeByIndexes(row,0,values.length,headers.length).format.numberFormat='0.00';
}
function title(sheet,text,note1,note2){
  sheet.getRange('A2').values=[[text]];sheet.getRange('A2').format.font={name:'Arial',size:16,bold:true,color:colors.ink};
  if(note1)sheet.getRange('A3').values=[[note1]];
  if(note2)sheet.getRange('A4').values=[[note2]];
  sheet.getRange('A3:N4').format.font={name:'Arial',size:10,color:colors.muted};
  sheet.getRange('A3:N4').format.wrapText=false;
}
function writeFormula(sheet,row,col,value){sheet.getCell(row-1,col-1).formulas=[[value]];}

const cellValues=cells.map(r=>[r.protein,r.condition,r.BR,r.cell,r.On,r.Not_on,r.unknown,r.QC_excluded,r.selected,null,r.first_mask_ROI_area_um2,r.mean_ROI_area_um2]);
const cs=sheets.Cells;style(cs,cells.length+5,12);
table(cs,1,['Protein','Condition','BR','Cell','On','Not-on','Unknown','QC excluded','Selected localizations','On %','First mask area (µm²)','Mean mask area (µm²)'],cellValues,[14,13,10,10,15,15,13,15,20,15,21,22]);
const endCell=cells.length+1;
for(let i=0;i<cells.length;i++){const r=i+2;writeFormula(cs,r,10,'=IF(E'+r+'+F'+r+'=0,"n.a.",E'+r+'/(E'+r+'+F'+r+'))');}
cs.getRange('E2:I'+endCell).setNumberFormat('#,##0');cs.getRange('J2:J'+endCell).setNumberFormat('0.00%');cs.getRange('K2:L'+endCell).setNumberFormat('0.00');
cs.freezePanes.freezeRows(1);cs.freezePanes.freezeColumns(4);
cs.getRange('A'+(endCell+3)).values=[['Sources: saved TRAK1 and TRAK2 cell metrics; full source records are retained with the analysis.']];

const bs=sheets.BR;style(bs,brs.length+8,13);title(bs,'Biological replicate measurements','Equal cell weighting within each BR. Pooled count On% is shown separately.','Area values use only the original cell ROI. First mask means the 0–49 frame support block.');
table(bs,6,['Protein','Condition','BR','Cells','Mean cell On %','Cell On SD (pp)','Pooled count On %','Mean On per cell','Mean Not-on per cell','First mask mean (µm²)','First mask SD (µm²)','Mean mask mean (µm²)','Mean mask SD (µm²)'],brs.map(r=>[r.protein,r.condition,r.BR,null,null,r.sample_sd,null,null,null,null,metric(data.br_metrics,r.protein,r.condition,r.BR,'first_mask_ROI_area_um2').sample_sd,null,metric(data.br_metrics,r.protein,r.condition,r.BR,'mean_ROI_area_um2').sample_sd]),[14,13,10,9,16,17,19,18,20,22,21,24,22]);
const cellCol=letter=>"'Cells'!$"+letter+"$2:$"+letter+'$'+endCell;
for(let i=0;i<brs.length;i++){
  const r=i+7, criteria=cellCol('A')+',$A'+r+','+cellCol('B')+',$B'+r+','+cellCol('C')+',$C'+r;
  writeFormula(bs,r,4,'=COUNTIFS('+criteria+')');
  writeFormula(bs,r,5,'=AVERAGEIFS('+cellCol('J')+','+criteria+')');
  const on='SUMIFS('+cellCol('E')+','+criteria+')',off='SUMIFS('+cellCol('F')+','+criteria+')';
  writeFormula(bs,r,7,'='+on+'/('+on+'+'+off+')');
  for(const [outCol,inCol]of[[8,'E'],[9,'F'],[10,'K'],[12,'L']])writeFormula(bs,r,outCol,'=AVERAGEIFS('+cellCol(inCol)+','+criteria+')');
}
const endBR=brs.length+6;
bs.getRange('D7:D'+endBR).setNumberFormat('#,##0');
bs.getRange('E7:E'+endBR).setNumberFormat('0.00%');bs.getRange('G7:G'+endBR).setNumberFormat('0.00%');
bs.getRange('H7:I'+endBR).setNumberFormat('#,##0.00');bs.freezePanes.freezeRows(6);bs.freezePanes.freezeColumns(3);

const ss=sheets.Summary;style(ss,29,10);title(ss,'TRAK1 and TRAK2 across conditions','Cell On% → equal-cell BR mean → equal-BR condition mean. SD below is between BR means.','BRs are provisionally treated as independent; pairing is unconfirmed. Each group contains three BRs.');
table(ss,6,['Protein','Condition','Cells','BRs','Mean On %','On SD (pp)','First mask mean (µm²)','First mask SD (µm²)','Mean mask area (µm²)','Mean mask SD (µm²)'],conditions.map(r=>[r.protein,r.condition,null,null,null,null,null,null,null,null]),[15,14,14,15,17,17,22,21,24,22]);
const brCol=letter=>"'BR'!$"+letter+'$7:$'+letter+'$'+endBR;
for(let i=0;i<conditions.length;i++){
 const r=i+7,conditionsExpr=brCol('A')+',$A'+r+','+brCol('B')+',$B'+r;
 writeFormula(ss,r,3,'=SUMIFS('+brCol('D')+','+conditionsExpr+')');writeFormula(ss,r,4,'=COUNTIFS('+conditionsExpr+')');
 for(const [col,source]of[[5,'E'],[7,'J'],[9,'L']])writeFormula(ss,r,col,'=AVERAGEIFS('+brCol(source)+','+conditionsExpr+')');
 const ids=brs.map((b,j)=>b.protein===conditions[i].protein&&b.condition===conditions[i].condition?j+7:null).filter(x=>x!==null);
 for(const [col,source,multiplier]of[[6,'E',100],[8,'J',1],[10,'L',1]])writeFormula(ss,r,col,'=STDEV.S('+ids.map(n=>"'BR'!"+source+n).join(',')+')'+(multiplier===100?'*100':''));
}
ss.getRange('C7:D12').setNumberFormat('#,##0');ss.getRange('E7:E12').setNumberFormat('0.00%');
ss.getRange('A15').values=[['Why dividing mean counts gives a different On%']];ss.getRange('A15').format.font={name:'Arial',size:14,bold:true};
ss.getRange('A16').values=[['Count-pooled percentages weight cells by their localization counts. The primary estimate gives cells and then BRs equal weight.']];ss.getRange('A16').format.font={name:'Arial',size:10,color:colors.muted};
const audits=data.aggregation_audit.filter(r=>r.BR==='ALL').sort(sortGroup);
table(ss,18,['Protein','Condition','Mean On per cell','Mean Not-on per cell','Count-pooled On %','All-cell mean On %','Equal-BR mean On %','BR mean − pooled (pp)'],audits.map(r=>[r.protein,r.condition,r.On_per_cell_mean,r.Not_on_per_cell_mean,null,r.mean_of_cell_ratios/100,r.equal_BR_mean_of_cell_ratios/100,r.equal_BR_mean_of_cell_ratios-r.pooled_On_counts_ratio]),[15,14,14,15,17,17,22,21]);
for(let i=0;i<audits.length;i++){const r=i+19;writeFormula(ss,r,5,'=C'+r+'/(C'+r+'+D'+r+')');}
ss.getRange('C19:D24').setNumberFormat('#,##0.00');ss.getRange('E19:G24').setNumberFormat('0.00%');
ss.getRange('A27').values=[['Tests: exact two-sided independent-BR permutation; Holm correction across 27 planned comparisons. See Tests for effect sizes and p values.']];
ss.getRange('A28').values=[['With 3 vs 3 BRs there are 20 allocations; the minimum exact two-sided raw p is 0.10. Non-rejection does not establish equivalence.']];
ss.getRange('A27:J28').format.font={name:'Arial',size:10,color:colors.muted};

const ts=sheets.Tests;style(ts,38,14);title(ts,'Independent-BR exploratory comparisons','Two-sided exact permutation of BR means. Effects are group A minus group B.','Holm FWER correction across all 27 comparisons. Independence is assumed because pairing is unconfirmed.');
const tests=data.comparisons;
const epnames={on_percentage:'On percentage',mean_ROI_area_um2:'Mean ROI mask area',first_mask_ROI_area_um2:'First-block ROI mask area'};
table(ts,6,['Metric','Group A','Group B','BRs A','BRs B','Cells A','Cells B','Mean A','Mean B','Difference A − B','Difference unit','Raw p','Holm p','Reject at 0.05'],tests.map(r=>[epnames[r.endpoint],r.protein_a+' '+r.condition_a,r.protein_b+' '+r.condition_b,r.n_BR_a,r.n_BR_b,r.n_cells_a,r.n_cells_b,r.mean_a/(r.endpoint==='on_percentage'?100:1),r.mean_b/(r.endpoint==='on_percentage'?100:1),r.difference_a_minus_b,r.endpoint==='on_percentage'?'pp':'µm²',r.p_raw,r.p_holm,r.significant_holm?'Yes':'No']),[28,17,17,9,9,10,10,16,16,21,16,12,12,17]);
for(let i=0;i<tests.length;i++)if(tests[i].endpoint==='on_percentage')ts.getRange('H'+(i+7)+':I'+(i+7)).setNumberFormat('0.00%');
ts.getRange('D7:G33').setNumberFormat('#,##0');ts.getRange('L7:M33').setNumberFormat('0.000');ts.freezePanes.freezeRows(6);ts.freezePanes.freezeColumns(3);
ts.getRange('A36').values=[['Statistical tests are saved external-analysis results. Recompute the analysis after changing source measurements.']];
ts.getRange('A37').values=[['Permutation reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html']];
for(const n of ['Summary','Tests'])sheets[n].tabColor=colors.header;

wb.recalculate();
const inspect=await wb.inspect({kind:'region',sheetId:'Summary',range:'A6:J24',maxChars:5000,tableMaxRows:25,tableMaxCols:10});
await fs.writeFile(path.join(run,'workbook_inspection.json'),inspect.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},summary:'Workbook formula errors'});
await fs.writeFile(path.join(run,'workbook_error_scan.json'),errors.ndjson);
const preview=await wb.render({sheetName:'Summary',range:'A1:J29',scale:1.4,format:'png'});
await fs.writeFile(path.join(run,'workbook_preview.png'),new Uint8Array(await preview.arrayBuffer()));
const xlsx=await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(out,'Protein_Comparison.xlsx'));
try { await fs.rename(path.join(out,'Protein_Comparison.xlsx.inspect.ndjson'),path.join(run,'workbook_export_inspect.ndjson')); } catch(e) { if(e.code!=='ENOENT') throw e; }
await fs.writeFile(path.join(run,'workbook_values.json'),JSON.stringify({summary:ss.getRange('A7:J12').values,audit:ss.getRange('A19:H24').values,br:bs.getRange('A7:M24').values,cells:cs.getRange('A2:L212').values,tests:ts.getRange('A7:N33').values}));
console.log(JSON.stringify({output:path.join(out,'Protein_Comparison.xlsx'),cells:cells.length,BRs:brs.length,tests:tests.length,errors:errors.ndjson},null,2));
