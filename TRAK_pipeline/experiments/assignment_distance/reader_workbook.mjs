import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const argument = process.argv[2];
if (!argument) throw new Error('Usage: node reader_workbook.mjs <config.json>');
const configPath = path.resolve(argument);
const cfg = JSON.parse(await fs.readFile(configPath, 'utf8'));
const outputDir = path.resolve(cfg.output_directory || cfg.output_dir || cfg.output);
const portableRoot = '/Volumes/My Book/Xinran_FullsetTRAK2';
if (!outputDir.startsWith(portableRoot + path.sep)) throw new Error('Workbook output must remain on My Book');
const summaryPath = path.resolve(cfg.summary_json || path.join(outputDir, 'summary.json'));
const transitionsPath = path.resolve(cfg.transitions_csv || path.join(outputDir, 'transitions.csv'));
const validationDir = path.join(outputDir, 'validation');
const protectedPaths = [configPath, summaryPath, transitionsPath];
async function digest(p) {
  const bytes = await fs.readFile(p);
  return { path: p, bytes: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') };
}
const before = await Promise.all(protectedPaths.map(digest));
const summary = JSON.parse(await fs.readFile(summaryPath, 'utf8'));
const methods = summary.methods;
const flips = summary.flips;
if (!Array.isArray(methods) || !Array.isArray(flips)) throw new Error('Expected summary.methods and summary.flips arrays');

function parseCSV(text) {
  const rows = []; let row = [], cell = '', quoted = false;
  for (let i=0; i<text.length; i++) {
    const ch=text[i];
    if (ch==='"') {
      if (quoted && text[i+1]==='"') { cell+='"'; i++; }
      else quoted=!quoted;
    } else if (ch===',' && !quoted) { row.push(cell); cell=''; }
    else if ((ch==='\n' || ch==='\r') && !quoted) {
      if (ch==='\r' && text[i+1]==='\n') i++;
      row.push(cell); cell=''; if (row.some(v=>v!=='')) rows.push(row); row=[];
    } else cell+=ch;
  }
  if (cell!=='' || row.length) { row.push(cell); rows.push(row); }
  if (quoted) throw new Error('Unterminated CSV quote');
  const headers=rows.shift() || [];
  return rows.map(values=>Object.fromEntries(headers.map((h,i)=>[h,values[i] ?? ''])));
}
const transitions = parseCSV(await fs.readFile(transitionsPath,'utf8'));
for (const col of ['from_method','to_method','from_label','to_label','count'])
  if (transitions.length && !(col in transitions[0])) throw new Error('Missing transition column '+col);
const labels = new Map(methods.map(m=>[m.key,m.label]));
const integer=(v,label)=>{ const n=Number(v); if (!Number.isSafeInteger(n)||n<0) throw new Error('Invalid count '+label); return n; };
const number=v=>v==null ? null : Number(v);
const percent=v=>v==null ? null : Number(v)/100;
const prettyLabel=v=>({'mito-compatible':'On','non-mito-compatible':'Not-on','unknown':'Unknown','QC excluded':'QC excluded','excluded':'QC excluded'}[v] || v);
const prettyMethod=v=>labels.get(v) || v;
const sample=summary.sample_id || cfg.sample_id || 'WT_TRAK2_BR1_T5';
const sampleTitle=sample.replaceAll('_',' ');

const workbook=Workbook.create();
const main=workbook.worksheets.add('Summary');
const change=workbook.worksheets.add('Classification_changes');
function col(n) { let out=''; while(n) { n--; out=String.fromCharCode(65+n%26)+out; n=Math.floor(n/26); } return out; }
function base(sheet,widths,lastRow,title,note) {
  const last=col(widths.length);
  sheet.showGridLines=false; sheet.tabColor='#203746';
  const used=sheet.getRange('A1:'+last+lastRow);
  used.format.font={name:'Arial',size:11,color:'#17212B'}; used.format.rowHeight=24;
  for(let c=0;c<widths.length;c++) sheet.getRange(col(c+1)+'1:'+col(c+1)+lastRow).format.columnWidth=widths[c];
  sheet.mergeCells('A2:'+last+'2'); sheet.getRange('A2').values=[[title]];
  sheet.getRange('A2:'+last+'2').format.font={name:'Arial',size:16,bold:true,color:'#203746'};
  sheet.getRange('A2:'+last+'2').format.rowHeight=32;
  sheet.mergeCells('A3:'+last+'4'); sheet.getRange('A3').values=[[note]];
  sheet.getRange('A3:'+last+'4').format={font:{name:'Arial',size:10,color:'#52616B'},wrapText:true,verticalAlignment:'center'};
  sheet.getRange('A5:'+last+'5').format.rowHeight=12;
}
function header(sheet,row,headers) {
  const r=sheet.getRange('A'+row+':'+col(headers.length)+row);r.values=[headers];
  r.format={fill:'#203746',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:48,verticalAlignment:'center',horizontalAlignment:'center'};
  r.format.borders={preset:'inside',style:'thin',color:'#FFFFFF'};
}
function body(sheet,start,rows,types) {
  if(!rows.length) return;
  const end=start+rows.length-1,last=col(types.length);
  sheet.getRange('A'+start+':'+last+end).values=rows;
  for(let c=0;c<types.length;c++) {
    const range=sheet.getRange(col(c+1)+start+':'+col(c+1)+end);
    range.setNumberFormat(types[c]==='percent'?'0.00%':types[c]==='integer'?'#,##0':'General');
    range.format.horizontalAlignment=types[c]==='text'?'left':'right';
  }
  for(let row=start;row<=end;row++) if((row-start)%2) sheet.getRange('A'+row+':'+last+row).format.fill='#F0F4F7';
}
const mainHeaders=['Assignment method','On count','Not-on count','Unknown count','QC excluded count','Selected count','On percentage','Common-known On percentage','Common-known count'];
const mainRows=methods.map(m=>[
  m.label,integer(m.On,m.key+' On'),integer(m.Not_on,m.key+' Not-on'),integer(m.unknown,m.key+' unknown'),integer(m.QC_excluded,m.key+' QC excluded'),integer(m.selected,m.key+' selected'),
  percent(m.On_percentage),percent(m.common_known_On_percentage),integer(m.common_known_count,m.key+' common known')
]);
base(main,[42,17,17,17,19,18,20,25,22],Math.max(8,mainRows.length+6),sampleTitle+' assignment-distance comparison',
  'On percentage = On / (On + Not-on). Unknown and QC-excluded localizations are reported separately. Common-known results use the same localizations with known classifications under every compared method. Saved localization standard errors are unchanged.');
header(main,6,mainHeaders);body(main,7,mainRows,['text','integer','integer','integer','integer','integer','percent','percent','integer']);
for(let i=0;i<methods.length;i++) main.getRange('G'+(7+i)).formulas=[['=IF(B'+(7+i)+'+C'+(7+i)+'=0,"",B'+(7+i)+'/(B'+(7+i)+'+C'+(7+i)+'))']];
if (mainRows.length) main.tables.add('A6:I'+(mainRows.length+6),true,'AssignmentMethods');

const flipRows=flips.map(f=>[
  f.comparison,integer(f.On_to_Not_on,'On to Not-on'),integer(f.Not_on_to_On,'Not-on to On'),integer(f.other_transition_count,'Other transitions'),integer(f.common_known_count,'Common-known count')
]);
const detailedRows=transitions.map(t=>[prettyMethod(t.from_method),prettyMethod(t.to_method),prettyLabel(t.from_label),prettyLabel(t.to_label),integer(t.count,'Transition count')]);
const detailTitleRow=10+flipRows.length,detailHeaderRow=detailTitleRow+1,detailStart=detailTitleRow+2;
base(change,[43,43,23,23,25],Math.max(detailStart,detailStart+detailedRows.length-1),sampleTitle+' classification changes',
  'Directional counts compare the same localizations. On to Not-on and Not-on to On count label changes. Other transitions include changes involving Unknown. The detailed table below retains every saved transition count.');
header(change,6,['Comparison','On to Not-on','Not-on to On','Other transitions','Common-known count']);
body(change,7,flipRows,['text','integer','integer','integer','integer']);
if(flipRows.length) { change.getRange('A7:A'+(flipRows.length+6)).format.wrapText=true; change.getRange('A7:E'+(flipRows.length+6)).format.rowHeight=42; }
if(flipRows.length) change.tables.add('A6:E'+(flipRows.length+6),true,'DirectionalChanges');
change.mergeCells('A'+detailTitleRow+':E'+detailTitleRow);
change.getRange('A'+detailTitleRow).values=[['Detailed classification counts']];
change.getRange('A'+detailTitleRow+':E'+detailTitleRow).format.font={name:'Arial',size:13,bold:true,color:'#203746'};
header(change,detailHeaderRow,['Method before','Method after','Classification before','Classification after','Count']);
body(change,detailStart,detailedRows,['text','text','text','text','integer']);
if(detailedRows.length) change.tables.add('A'+detailHeaderRow+':E'+(detailStart+detailedRows.length-1),true,'ClassificationTransitions');
change.freezePanes.freezeRows(6);

workbook.recalculate();
const expectedPercent=methods.map(m=>m.On+m.Not_on===0 ? null : Number(m.On)/(Number(m.On)+Number(m.Not_on)));
const calculated=main.getRange('G7:G'+(methods.length+6)).values.map(row=>row[0]);
for(let i=0;i<methods.length;i++) {
  if(expectedPercent[i]==null) { if(calculated[i]!=='' && calculated[i]!=null) throw new Error('Zero denominator should be unavailable'); }
  else if(typeof calculated[i]!=='number'||Math.abs(calculated[i]-expectedPercent[i])>1e-12) throw new Error('On percentage formula mismatch');
  const saved=number(methods[i].On_percentage);
  if(saved!=null && expectedPercent[i]!=null && Math.abs(saved/100-expectedPercent[i])>1e-12) throw new Error('Saved percentage disagrees with saved counts');
}
await fs.mkdir(validationDir,{recursive:true});
const inspections=[];
for(const [sheetName,lastRow,lastCol] of [['Summary',methods.length+6,'I'],['Classification_changes',detailStart+detailedRows.length-1,'E']]) {
  const inspection=await workbook.inspect({kind:'region',sheetId:sheetName,range:'A1:'+lastCol+lastRow,maxChars:6000,tableMaxRows:Math.min(lastRow,24),tableMaxCols:9});
  inspections.push({sheet:sheetName,inspection});
  const preview=await workbook.render({sheetName,range:'A1:'+lastCol+lastRow,scale:1,format:'png'});
  await fs.writeFile(path.join(validationDir,sheetName+'_preview.png'),new Uint8Array(await preview.arrayBuffer()));
}
const errors=await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},summary:'Final formula error scan'});
const outputPath=path.join(outputDir,'Distance_Comparison.xlsx');
const output=await SpreadsheetFile.exportXlsx(workbook);await output.save(outputPath);
const after=await Promise.all(protectedPaths.map(digest));
if(JSON.stringify(before)!==JSON.stringify(after)) throw new Error('Protected workbook inputs changed');
const report={all_passed:true,output:outputPath,sheets:['Summary','Classification_changes'],method_rows:methods.length,comparison_rows:flips.length,detailed_transition_rows:transitions.length,percentage_formulas_verified:true,common_known_percentages_stored_as_fractions:true,saved_standard_errors_unchanged:true,source_inputs_unchanged:true,source_fingerprints:before,inspections,formula_error_scan:errors,native_Excel_executed:false};
await fs.writeFile(path.join(validationDir,'workbook_validation.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({all_passed:true,output:outputPath,method_rows:methods.length,transition_rows:transitions.length,previews:validationDir}));
