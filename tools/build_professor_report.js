const fs = require('fs');

const input = 'docs/final_report.md';
const output = 'docs/Healthcare_Data_Imputation_Final_Report.pdf';
const W = 595.28, H = 841.89, M = 54, TOP = 770, BOTTOM = 58;

const clean = s => s
  .replace(/\*\*/g, '').replace(/\*/g, '').replace(/`/g, '')
  .replace(/[–—]/g, '-').replace(/[“”]/g, '"').replace(/[’]/g, "'")
  .replace(/×/g, 'x').replace(/±/g, '+/-').replace(/≥/g, '>=').replace(/≤/g, '<=')
  .replace(/→/g, '->').replace(/−/g, '-').replace(/·/g, '-');

function wrap(text, max) {
  const words = clean(text).split(/\s+/).filter(Boolean), lines = [];
  let line = '';
  for (const word of words) {
    if ((line + ' ' + word).trim().length > max && line) { lines.push(line); line = word; }
    else line = (line + ' ' + word).trim();
  }
  if (line) lines.push(line);
  return lines;
}

function esc(s) { return s.replace(/\\/g, '\\\\').replace(/\(/g, '\\(').replace(/\)/g, '\\)'); }
function textOp(text, x, y, size, font = 'F1', color = '0.12 0.16 0.22') {
  return `BT ${color} rg /${font} ${size} Tf 1 0 0 1 ${x.toFixed(1)} ${y.toFixed(1)} Tm (${esc(text)}) Tj ET`;
}

const pages = [];
function addPage(lines, cover = false) {
  const ops = [];
  if (!cover) {
    ops.push(textOp('HEALTHCARE DATA IMPUTATION | FINAL RESEARCH REPORT', M, 810, 7.5, 'F2', '0.32 0.42 0.55'));
    ops.push(`0.72 0.78 0.84 RG 0.6 w ${M} 800 m ${W - M} 800 l S`);
  }
  let y = cover ? 0 : TOP;
  for (const item of lines) {
    if (item.type === 'rule') { ops.push(`0.72 0.78 0.84 RG 0.6 w ${M} ${y + 5} m ${W - M} ${y + 5} l S`); y -= 14; continue; }
    const h = item.type === 'h1' ? 20 : item.type === 'h2' ? 16 : 13;
    if (item.type === 'h1' || item.type === 'h2') {
      y -= item.type === 'h1' ? 8 : 5;
      ops.push(textOp(item.text, M, y, item.type === 'h1' ? 16 : 12, 'F2', item.type === 'h1' ? '0.08 0.25 0.45' : '0.16 0.34 0.52'));
      y -= h;
    } else {
      ops.push(textOp(item.text, M + (item.indent || 0), y, item.size || 9.3, item.font || 'F1', item.color || '0.12 0.16 0.22'));
      y -= item.leading || 13;
    }
  }
  pages.push({ops, cover});
}

// Human-readable cover page.
addPage([
  {type:'h1', text:'Healthcare Data Imputation'},
  {type:'h2', text:'Using Machine Learning with Genetic Algorithm-Based Feature Selection'},
  {type:'rule'},
  {type:'p', text:'Final Research Report', size:13, font:'F2', color:'0.16 0.34 0.52'},
  {type:'p', text:'Heart Failure Clinical Records Dataset | UCI Machine Learning Repository', size:10.5},
  {type:'p', text:'Prepared for academic review | 5 September 2026', size:10},
  {type:'p', text:'', leading:24},
  {type:'h2', text:'Executive result'},
  ...wrap('Missingness itself measurably reduced downstream prediction performance in 4 of 9 complete-data comparisons. No imputation method or missingness mechanism was statistically distinguishable at five seeds. Genetic Algorithm selection produced a stable, clinically plausible reduced feature set, but did not reliably improve prediction after replication.', 78).map(text => ({type:'p', text})),
  {type:'p', text:'', leading:20},
  {type:'p', text:'Repository: Data_imputation_in_health_care', size:9, color:'0.35 0.40 0.46'},
], true);

const lines = fs.readFileSync(input, 'utf8').split(/\r?\n/);
let blocks = [], para = [];
function flush() {
  if (!para.length) return;
  for (const line of wrap(para.join(' '), 102)) blocks.push({type:'p', text:line});
  blocks.push({type:'p', text:'', leading:4}); para = [];
}
for (const raw of lines) {
  const line = raw.trim();
  if (!line) { flush(); continue; }
  if (/^# /.test(line)) continue;
  if (/^## /.test(line)) { flush(); blocks.push({type:'h1', text:line.slice(3)}); continue; }
  if (/^### /.test(line)) { flush(); blocks.push({type:'h2', text:line.slice(4)}); continue; }
  if (/^\|?\s*:?-{3,}/.test(line)) continue;
  if (/^[-*] /.test(line)) {
    flush(); const parts = wrap(line.slice(2), 96);
    blocks.push({type:'p', text:'- ' + parts[0], indent:8});
    for (const part of parts.slice(1)) blocks.push({type:'p', text:part, indent:16});
    continue;
  }
  if (line.startsWith('|')) {
    flush(); blocks.push({type:'p', text:line.replace(/\|/g, ' | '), size:7.2, font:'F3', color:'0.18 0.25 0.32'}); continue;
  }
  para.push(line);
}
flush();

let page = [], y = TOP;
for (const block of blocks) {
  const need = block.type === 'h1' ? 45 : block.type === 'h2' ? 31 : (block.leading || 13);
  if (y - need < BOTTOM && page.length) { addPage(page); page = []; y = TOP; }
  page.push(block); y -= need;
}
if (page.length) addPage(page);

function pdfObject(n, body) { return `${n} 0 obj\n${body}\nendobj\n`; }
const objects = [];
objects[1] = pdfObject(1, '<< /Type /Catalog /Pages 2 0 R >>');
const fontRefs = {F1:3, F2:4, F3:5};
objects[3] = pdfObject(3, '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>');
objects[4] = pdfObject(4, '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>');
objects[5] = pdfObject(5, '<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>');
const pageRefs = [];
let next = 6;
for (let i = 0; i < pages.length; i++) {
  const p = pages[i];
  if (!p.cover) p.ops.push(textOp(`Page ${i} of ${pages.length - 1}`, W - 108, 34, 7.5, 'F1', '0.38 0.43 0.49'));
  const stream = p.ops.join('\n');
  const content = next++, pageObj = next++;
  objects[content] = pdfObject(content, `<< /Length ${Buffer.byteLength(stream)} >>\nstream\n${stream}\nendstream`);
  objects[pageObj] = pdfObject(pageObj, `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${W} ${H}] /Resources << /Font << /F1 ${fontRefs.F1} 0 R /F2 ${fontRefs.F2} 0 R /F3 ${fontRefs.F3} 0 R >> >> /Contents ${content} 0 R >>`);
  pageRefs.push(`${pageObj} 0 R`);
}
objects[2] = pdfObject(2, `<< /Type /Pages /Kids [${pageRefs.join(' ')}] /Count ${pageRefs.length} >>`);
let pdf = '%PDF-1.4\n%\xE2\xE3\xCF\xD3\n', offsets = [0];
for (let i = 1; i < objects.length; i++) { offsets[i] = Buffer.byteLength(pdf, 'binary'); pdf += objects[i]; }
const xref = Buffer.byteLength(pdf, 'binary');
pdf += `xref\n0 ${objects.length}\n0000000000 65535 f \n`;
for (let i = 1; i < objects.length; i++) pdf += `${String(offsets[i]).padStart(10, '0')} 00000 n \n`;
pdf += `trailer\n<< /Size ${objects.length} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
fs.mkdirSync('docs', {recursive:true}); fs.writeFileSync(output, pdf, 'binary');
console.log(`Wrote ${output} (${pages.length} pages)`);
