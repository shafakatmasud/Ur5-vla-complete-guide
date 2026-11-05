#!/usr/bin/python3
"""Build a source-linked evaluation report and workbook; never alter trial data."""
import argparse
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import json
import math
import statistics
import zipfile
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.chart import BarChart, Reference
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table as PDFTable, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4
from docx import Document
from docx.shared import Inches, Pt

ORDER=['shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3']
COLORS={'Complete success':'17806D','Partial completion':'D19518','Failure':'C14953'}
SHORT={1:'Closed behind cube; no pickup.',2:'Closed behind cube; no pickup.',3:'Held cube over bowl; released only by next episode preparation.',4:'Pickup, then movement away from bowl.',5:'Pickup; bowl not reached.',6:'Pickup; bowl not reached.',7:'Complete task reported.',8:'Failure; stage unspecified.',9:'Pickup and transport; released outside bowl.',10:'Complete task reported.',11:'Complete task reported.',12:'Failure; stage unspecified; long capture contains multiple launches.',13:'Complete task reported with temporary pickup reposition.',14:'Complete failure; stage unspecified.',15:'Failure; stage unspecified.',16:'Failure; stage unspecified.',17:'Complete task reported.',18:'Complete task reported.',19:'Retest succeeded after two separately archived software aborts.',20:'Orange-cube comparison run; physical pick-and-place completion reported.'}


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def outcome(t): return 'Complete success' if t['operator_outcome']=='pass' else 'Partial completion' if t.get('pickup') is True else 'Failure'
def stage(v): return 'Yes' if v is True else 'No' if v is False else 'Unknown'
def local(utc): return datetime.fromisoformat(utc).astimezone(ZoneInfo('America/Toronto')).strftime('%Y-%m-%d %H:%M:%S %Z')
def condition(a): return 'A: policy carry' if a<=6 else 'C: planned carry + pickup shift' if a in (12,13) else 'D: planned carry + queued diagnostics' if a>=19 else 'B: planned carry'
def scalar_rows(value,path=''):
    if isinstance(value,(dict,list)) and not value:
        yield path,value
        return
    if isinstance(value,dict):
        for k,v in value.items(): yield from scalar_rows(v,f'{path}.{k}' if path else k)
    elif isinstance(value,list):
        for i,v in enumerate(value): yield from scalar_rows(v,f'{path}[{i}]')
    else: yield path,value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene-confirmation',type=Path)
    args=parser.parse_args();root=args.session.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assets=out/'assets';assets.mkdir(exist_ok=True)
    data=[]
    for a in range(1,21):
        p=root/f'cross_01/attempt_{a:02d}'
        t=read(p/'trial.json');s=read(p/'summary.json');m=read(p/'report_frames/source_manifest.json')
        assert t['attempt']==a and t['status']=='finished'
        assert all(t[k]==s[k] for k in ['operator_outcome','pickup','carry','placement'])
        data.append(dict(a=a,p=p,t=t,s=s,m=m))
    counts=Counter(outcome(d['t']) for d in data)
    assert [counts[k] for k in ['Complete success','Partial completion','Failure']]==[8,5,7]
    stage_counts={key:{label:sum(stage(d['t'].get(key))==label for d in data) for label in ['Yes','No','Unknown']} for key in ['pickup','carry','placement']}
    confirmation=read(args.scene_confirmation) if args.scene_confirmation else None
    color_note=('The operator confirmed that attempts 1-19 used the blue cube and only attempt 20 used the orange cube. Camera frames agree. The attempt-20 runtime snapshot still records the blue-cube instruction, so the final run demonstrates physical pick-and-place completion, not compliance with the requested object colour.') if confirmation else 'Representative base-camera frames show a blue cube in attempts 1-19 and an orange/red cube in attempt 20. The scene-colour change is awaiting operator confirmation. This is a sampled image audit, not continuous object tracking.'
    summary={'session':root.name,'date':'2026-10-19','timezone':'America/Toronto','selected_attempts':20,'counts':dict(counts),'overall_reported_full_completion_rate':.4,'complete_success_attempts':[d['a'] for d in data if d['t']['operator_outcome']=='pass'],'stage_counts':stage_counts,'blue_cube_attempts':19 if confirmation else None,'blue_cube_complete_successes':7 if confirmation else None,'blue_cube_completion_proportion':7/19 if confirmation else None,'orange_cube_attempts':1 if confirmation else None,'orange_cube_complete_successes':1 if confirmation else None,'software_aborted_attempt19_captures_excluded':2,'scene_audit':color_note,'timing_warning':'Capture duration includes setup, pauses, reporting delay and sometimes multiple launcher runs. It is not verified task completion time.','scope_warning':'Mixed execution conditions; full passes use planned transport. 40% is an operator-reported system completion proportion, not an autonomous frozen-policy generalization benchmark.'}
    (out/'results_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Outcome figures use measured/operator-reported records only.
    fig,ax=plt.subplots(figsize=(8,3.1));labels=['Complete success','Partial completion','Failure'];vals=[counts[k] for k in labels]
    bars=ax.barh(labels,vals,color=['#'+COLORS[k] for k in labels]);ax.invert_yaxis();ax.set_xlim(0,10);ax.set_xlabel('Selected attempts (n = 20)')
    for b,v in zip(bars,vals):ax.text(v+.12,b.get_y()+b.get_height()/2,f'{v} ({v/20:.0%})',va='center')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(assets/'outcomes.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,2.6));xs=list(range(1,21));ys=[1 if outcome(d['t'])=='Complete success' else .6 if outcome(d['t'])=='Partial completion' else .25 for d in data]
    ax.bar(xs,ys,color=['#'+COLORS[outcome(d['t'])] for d in data]);ax.set_xticks(xs);ax.set_yticks([.25,.6,1],['Failure','Partial','Complete']);ax.set_xlabel('Attempt number');ax.set_ylim(0,1.2);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(assets/'attempt_sequence.png',dpi=180);plt.close(fig)
    # Spreadsheet: normalized result rows, full summary fields, events and links.
    wb=Workbook();wb.remove(wb.active)
    def sheet(name,headers,rows,widths=None):
        ws=wb.create_sheet(name);ws.append(headers)
        for row in rows:ws.append([json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for v in row])
        ws.freeze_panes='B2';ws.auto_filter.ref=ws.dimensions
        for c in ws[1]:c.fill=PatternFill('solid',fgColor='17344A');c.font=Font(color='FFFFFF',bold=True);c.alignment=Alignment(wrap_text=True,vertical='center')
        ws.row_dimensions[1].height=32
        for row in ws.iter_rows(min_row=2):
            for c in row:c.alignment=Alignment(vertical='top',wrap_text=True)
        for i in range(1,ws.max_column+1):ws.column_dimensions[ws.cell(1,i).column_letter].width=(widths or {}).get(i,22)
        if rows:
            tab=Table(displayName=name.replace('_','')+'Table',ref=ws.dimensions);tab.tableStyleInfo=TableStyleInfo(name='TableStyleMedium2',showRowStripes=True);ws.add_table(tab)
        ws.sheet_properties.pageSetUpPr.fitToPage=True;ws.page_setup.orientation='landscape';ws.page_setup.paperSize=ws.PAPERSIZE_A4;ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=0
        ws.print_title_rows='1:1'
        return ws
    notes=[
        ['Selected attempts',20,'Numbered attempts only; two software-aborted attempt-19 captures excluded.'],
        ['Complete successes',8,'Operator-confirmed pickup, transport and placement in bowl.'],
        ['Partial completions',5,'Pickup succeeded but complete placement criterion was not met.'],
        ['Other failures',7,'Includes five failures with unreported individual stage outcomes.'],
        ['Reported system completion proportion',.4,'Mixed execution conditions; not an autonomous policy success rate.'],
        ['Confirmed pickup successes',stage_counts['pickup']['Yes'],'Unknown pickup outcomes stay unknown.'],
        ['Confirmed carry successes',stage_counts['carry']['Yes'],'Carry means transport toward/over bowl, not successful release.'],
        ['Confirmed placement successes',stage_counts['placement']['Yes'],'Operator report is the result authority.'],
        ['Software-aborted captures excluded',2,'Retained separately; see Excluded_runs.'],
        ['Images exported',sum(len(d['m']['panels']) for d in data),'10 base + 10 wrist candidate frames per selected trial.'],
        ['Scene verification',color_note,'Do not call the entire session blue-only without resolving the scene change.'],
        ['Timing',summary['timing_warning'],'The 300 s field is a recording-window reference, not a validated task deadline.'],
        ['Methods',summary['scope_warning'],'See Conditions and Report PDF.'],
        ['Physical placement codes','Internal cross_01 is a recording series','Physical cross IDs and calibrated cube positions are not established.'],
        ['Raw evidence',str(root),'Workbook contains full stored summary fields and links; full per-message traces remain in JSONL, not embedded in Excel.'],
    ]
    if confirmation:
        notes += [['Blue-cube attempts',19,'Attempts 1-19, operator-confirmed.'],['Blue-cube complete successes',7,'Attempts 7,10,11,13,17,18,19; later successes use planned transport.'],['Blue-cube completion proportion',7/19,'7/19 = 36.8%; mixed execution conditions.'],['Orange-cube comparison', '1 complete physical completion / 1 attempt','Attempt 20; recorded language instruction remained blue. Not a blue-generalization trial.']]
    ws=sheet('Summary',['Metric','Value','Interpretation'],notes,{1:38,2:65,3:90});ws['B6'].number_format='0%'
    for row in ws.iter_rows(min_row=2):
        if row[0].value=='Blue-cube completion proportion':row[1].number_format='0.0%'
    chart=BarChart();chart.type='bar';chart.title='Reported outcomes / 20 selected attempts';chart.add_data(Reference(ws,min_col=2,min_row=3,max_row=5),titles_from_data=False);chart.set_categories(Reference(ws,min_col=1,min_row=3,max_row=5));chart.legend=None;ws.add_chart(chart,'E2')
    rows=[]
    for d in data:
        t=d['t'];a=d['a'];s=d['s']
        rows.append([a,t['operator_outcome'],outcome(t),stage(t['pickup']),stage(t['carry']),stage(t['placement']),condition(a),'Orange/red visible; confirmation pending' if a==20 and not confirmation else 'Blue: operator confirmed + reviewed image' if a<20 and confirmation else 'Blue visible in reviewed sample' if a<20 else 'Orange: operator confirmed; instruction remained blue',local(t['started_utc']),local(t['ended_utc']),t['duration_s'],t['timeout_s'],t['timeout_reached'],t['reported_pass_within_limit'],SHORT[a],t['notes'],str(d['p'])])
    ws=sheet('Attempts',['Attempt','Stored full-task result','Report category','Pickup','Carry','Placement','Execution condition','Scene colour audit','Capture start (Toronto)','Capture end (Toronto)','Capture duration (s)','Reference window (s)','Capture >= reference','Legacy pass-within-window flag','Concise interpretation','Full recorded notes','Evidence folder'],rows,{1:12,3:25,7:44,8:46,9:28,10:28,15:65,16:115,17:70})
    for row in ws.iter_rows(min_row=2):
        color=COLORS[row[2].value];row[2].fill=PatternFill('solid',fgColor=color);row[2].font=Font(color='FFFFFF',bold=True);row[10].number_format='0.00';row[16].hyperlink=Path(row[16].value).as_uri();row[16].style='Hyperlink'
    metric_headers=['Attempt','Capture duration (s)','First policy command (s from marker)','First arm command (s from marker)','TCP path length (m; includes noise)','TCP samples','Invalid action count','Warning/error messages','Maximum consecutive policy jump (rad)']
    metric_headers += [f'Max policy-measured {j} (rad)' for j in ORDER]+[f'Max command-measured {j} (rad)' for j in ORDER]
    topics=sorted(set().union(*(set(d['s'].get('rates_hz',{})) for d in data)))
    metric_headers += [f'Mean received rate {t} (Hz)' for t in topics]+['Base video frames','Wrist video frames','Max base source age (s)','Max wrist source age (s)']
    metric_rows=[]
    for d in data:
        s=d['s'];row=[d['a']]+[s.get(k) for k in ['duration_s','first_policy_command_s','first_arm_command_s','tcp_path_length_m','tcp_samples','invalid_action_count','warnings_errors','max_consecutive_policy_jump_rad']]
        row+=s.get('max_policy_measured_gap_rad',[None]*6)+s.get('max_command_measured_gap_rad',[None]*6)
        row += [s.get('rates_hz',{}).get(t) for t in topics]+[s.get('video_frames',{}).get(c) for c in ['base','wrist']]+[s.get('max_color_source_age_s',{}).get(c) for c in ['base','wrist']]
        metric_rows.append(row)
    sheet('Telemetry',metric_headers,metric_rows)
    full=[];events=[];evidence=[];figrows=[]
    for d in data:
        for key,value in scalar_rows(d['s']):full.append([d['a'],key,value,type(value).__name__,str(d['p']/'summary.json')])
        for name in ['notable_logs','gripper_prediction_transitions','digital_output_transitions']:
            for event in d['s'].get(name,[]):events.append([d['a'],name,event.get('t_s'),event.get('node'),event.get('level'),event.get('message'),json.dumps(event,ensure_ascii=False)])
        for f in [d['p']/'trial.json',d['p']/'summary.json',d['p']/'observations.jsonl',d['p']/'report_frames/source_manifest.json',*sorted((d['p']/'video').glob('*.avi'))]:
            assert f.exists();evidence.append([d['a'],f.name,str(f),f.stat().st_size,sha(f) if f.suffix=='.json' else 'Not hashed; retained original large stream'])
        for panel in d['m']['panels']:
            f=d['p']/'report_frames'/panel['file'];assert f.exists()
            figrows.append([d['a'],panel['camera'],panel['step'],panel['capture_time_s'],panel['source_stamp_s'],panel.get('source_age_s'),panel['file'],str(f),'Candidate frame; not independent proof of grasp/placement'])
    for filename in ['scene_confirmation.json','session.json']:
        f=root/filename
        if f.exists():evidence.append([None,filename,str(f),f.stat().st_size,sha(f)])
    (out/'source_records.json').write_text(json.dumps({'scene_confirmation':confirmation,'selected_attempts':[{'attempt':d['a'],'trial':d['t'],'summary':d['s'],'camera_manifest':d['m']} for d in data]},indent=2)+'\n')
    sheet('All_summary_fields',['Attempt','JSON field path','Stored value','Value type','Summary source'],full,{2:65,3:110,5:85})
    sheet('Events',['Attempt','Event family','Capture-relative time (s)','Node','Log level','Message','Original event JSON'],events,{2:35,4:30,6:120,7:120})
    conditions=[['A','1-6','Frozen SmolVLA pickup and policy-generated transport',6,0,4,2,'No full-task pass. Four pickups reported; release/transport limitations.'],['B','7-11, 14-18','Policy pickup + checked planned bowl transport + release gate',10,5,1,4,'Adaptive Cartesian subdivision. No temporary 20 mm pickup shift. Software/camera scene were not randomized.'],['C','12-13','Condition B + planned OPEN pre-close 20 mm forward/left reposition',2,1,0,1,'Existing small alignment offset retained. Attempt 12 capture contains multiple launches.'],['D','19-20 (selected retests)','Condition B + isolated publication and queued diagnostics',2,2,0,0,'Two software aborts precede selected attempt 19. Attempt 20 used orange while its recorded instruction still said blue.']]
    sheet('Conditions',['Code','Selected attempts','Execution method','N','Full successes','Partial completions','Other failures','Interpretation'],conditions,{2:26,3:85,8:110})
    audit=[]
    for p in sorted((root/'cross_01').glob('aborted_attempt_19_*')):
        t=read(p/'trial.json');audit.append([p.name,'Software-aborted capture',t['operator_outcome'],False,t['duration_s'],t['notes'],str(p)])
    for name,desc in [('attempt_12_prestart_rejection.json','Preflight blocked before episode preparation; not a task attempt'),('attempt_19_retake_audit.json','Retake relocation/provenance'),('attempt_19_retake_audit_02.json','Second retake relocation/provenance')]:
        p=root/name
        if p.exists():audit.append([name,desc,'Excluded',False,None,p.read_text(),str(p)])
    for p in (root/'cross_01').glob('unused_prestart*'):
        audit.append([p.name,'Premature start marker before operator start','Excluded',False,None,'Preserved and excluded from selected attempts',str(p)])
    audit += [['Unrecorded repeat of 13','Operator-requested demonstration','Not scored',False,None,'Separate from formal attempt 13; no trial marker created.',''],['Calibration exercise','Stopped before solution or motion integration','Not applied',False,None,'Seven wrist-only references retained; not part of selected evaluation.','']]
    sheet('Excluded_runs',['Record','Classification','Outcome','Included in 20','Capture duration (s)','Notes / original audit','Source'],audit,{1:44,2:55,6:115,7:90})
    ws=sheet('Evidence',['Attempt','File','Absolute path','Bytes','SHA-256 (JSON only)'],evidence,{2:35,3:110,5:75})
    for row in ws.iter_rows(min_row=2):row[2].hyperlink=Path(row[2].value).as_uri();row[2].style='Hyperlink'
    ws=sheet('Camera_frames',['Attempt','Camera','Step','Capture-relative time (s)','Camera source epoch (s)','Reported source age (s)','Image file','Absolute image path','Interpretation'],figrows,{7:40,8:110,9:85})
    for row in ws.iter_rows(min_row=2):row[7].hyperlink=Path(row[7].value).as_uri();row[7].style='Hyperlink'
    definitions=[['Full success','Operator confirmed pickup, transport and release into bowl. All full passes used planned transport.'],['Partial completion','Full-task failure with reported successful pickup; five trials (3,4,5,6,9).'],['Unknown stage','Not reported; never converted to a failed pickup or fabricated sensor measurement.'],['Capture duration','Time from recording marker to finish marker. Includes waiting and reporting; attempt 12 includes multiple runs.'],['Legacy pass-within-window flag','Copied from source; evaluates capture window, not independently validated action completion time.'],['Joint tracking gap','Commanded joint target minus contemporaneous measured joints, max absolute per joint. Not cube/grasp error.'],['Policy gap','Policy target minus measured joint state. Includes normal smoothed-controller lag.'],['TCP path length','Sum of received TCP movements; includes sensor noise, idle time and repeated motion.'],['Warnings/errors','Count of ROS warning/error messages; startup waits and other node logs are included. Not a task-failure count.'],['Camera source age','Recorder-computed wall/source timestamp difference; source clocks and callback arrival age differ.'],['IO state','Acknowledged gripper command history; does not prove force, contact or retention.'],['Raw telemetry','Original per-message JSONL streams linked in Evidence, not all rows embedded in XLSX. All stored summary fields are included.'],['Cross encoding','cross_01 denotes this 20-attempt recording series, not a calibrated physical table cross.'],['Sampled colour audit',color_note],['Selected retakes','Two software-aborted captures excluded from 20; exclusion is explicit and does not estimate an all-execution success rate.'],['Event time','Seconds relative to each capture start marker; cameras not asserted hardware synchronized.']]
    sheet('Definitions',['Field / concept','Meaning and limitations'],definitions,{1:38,2:140})
    wb.calculation.fullCalcOnLoad=True
    workbook_path=out/'UR5_20_Attempts_Results.xlsx';wb.save(workbook_path)
    # PDF report. Shared sections also feed an editable DOCX.
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='SmallBody',fontName='Helvetica',fontSize=8,leading=10));styles['BodyText'].fontSize=10;styles['BodyText'].leading=14
    story=[];docx=Document();docx.styles['Normal'].font.name='Calibri';docx.styles['Normal'].font.size=Pt(10)
    def para(text,style='BodyText'):
        story.append(Paragraph(escape(str(text)).replace('\n','<br/>'),styles[style]));story.append(Spacer(1,3*mm));docx.add_paragraph(str(text))
    def title(text,level=1):story.append(Paragraph(escape(text),styles['Heading'+str(level)]));docx.add_heading(text,level)
    def table(headers,rows,widths,font=8):
        cells=[[Paragraph(escape(str(v)),styles['SmallBody']) for v in headers]]+[[Paragraph(escape('' if v is None else str(v)),styles['SmallBody']) for v in row] for row in rows]
        tbl=PDFTable(cells,colWidths=[w*mm for w in widths],repeatRows=1,hAlign='LEFT');tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DFEAF0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F4F7F9')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(tbl);story.append(Spacer(1,4*mm))
        tab=docx.add_table(rows=1, cols=len(headers));tab.style='Light Shading Accent 1'
        for cell,text in zip(tab.rows[0].cells,headers):cell.text=str(text)
        for row in rows:
            for cell,text in zip(tab.add_row().cells,row):cell.text='' if text is None else str(text)
    def page():story.append(PageBreak());docx.add_page_break()
    def picture(path,width=178,height=None):
        if height is None:
            from PIL import Image as PI
            with PI.open(path) as im:height=width*im.height/im.width
        story.append(Image(str(path),width=width*mm,height=height*mm));docx.add_picture(str(path),width=Inches(width/25.4))
    title('UR5 pick-and-place evaluation',1);para('20 selected attempts | Blue-cube instruction | 19 October 2026 | America/Toronto')
    title('Executive findings',2)
    para('The selected numbered trials contain 8 operator-confirmed physical pick-and-place completions (40%), 5 partial completions (25%) and 7 other failures (35%). Successful attempt IDs are 7, 10, 11, 13, 17, 18, 19 and 20. Two software-aborted captures preceding the selected attempt-19 retest are retained separately and excluded from these totals.')
    if confirmation:
        para('Object-specific result: 7 complete successes among 19 blue-cube attempts (36.8%). Attempt 20 used an orange cube and had one reported physical completion. That one orange run is a comparison observation, not evidence of blue-cube generalization or a reliable population success estimate.')
    picture(assets/'outcomes.png',178)
    para('The model checkpoint remained frozen, but the execution procedure changed. Complete passes used a checked planned transport route and calibrated bowl-release gate after policy pickup. Attempts 12 and 13 also used a planned pickup reposition. Therefore, 40% describes this assisted physical system across mixed execution conditions and object colours; it does not establish a 40% autonomous policy-generalization rate.')
    para(color_note)
    para('No synthetic attempts, assumed telemetry, or donor-replayed observations are included. This report uses trial metadata, saved telemetry summaries, runtime configurations, camera-frame manifests and selected real images.')
    page();title('Methods and execution conditions')
    para('Hardware: physical UR5, Robotiq electric gripper and base/wrist RealSense cameras. Policy: the local 110-episode SmolVLA checkpoint trained for 18,000 steps. Commanded instruction: Pick up the blue cube and place it in the bowl. Operating settings were speed scale 0.5 and 50 actions per prediction; individual runtime snapshots remain authoritative where historical session defaults differ.')
    table(['Condition','Attempts','Method','Full passes'],[[v[0],v[1],v[2],f'{v[4]}/{v[3]}'] for v in conditions],[18,32,103,25])
    para('Conditions occurred sequentially and were not randomized. Camera viewpoint, object placement, software revisions and scene contents were not controlled as one fixed benchmark. Group differences are descriptive; they cannot isolate the effect of one intervention.')
    title('Result and timing definitions',2)
    para('Complete success means operator-confirmed pickup, carry and release into the bowl. Partial completion means a successful pickup was reported but the full placement criterion was not met. Unknown stages remain unknown. Attempt 3 was corrected after review: release occurred only when the next episode preparation opened the gripper. Attempt 9 released outside the bowl and remains a partial completion.')
    para('The inherited 300-second reference is stored as a capture-window field. Recording continues beyond it; late reports, setup pauses and repeated launcher runs can inflate capture duration. Verified task completion times are not established for all trials, so this report does not present mean task time or time efficiency as a measured result.')
    table(['Stage','Reported yes','Reported no','Unknown'],[[k.capitalize(),v['Yes'],v['No'],v['Unknown']] for k,v in stage_counts.items()],[85,30,30,33])
    page();title('Attempt-by-attempt results')
    picture(assets/'attempt_sequence.png',178)
    table(['No.','Result','Pickup','Carry','Place','Condition','Capture s'],[[d['a'],outcome(d['t']),stage(d['t']['pickup']),stage(d['t']['carry']),stage(d['t']['placement']),condition(d['a'])[0],f"{d['t']['duration_s']:.1f}"] for d in data],[10,45,23,23,23,22,32])
    para('Capture seconds are recording durations, not measured task completion times. All complete outcomes are operator reports; raw IO cannot independently verify contact or placement. Result categories distinguish partial completions from other full-task failures.')
    page();title('Failure observations and software interruptions')
    table(['Attempts','Observation','Evidence / interpretation'],[
        ['1-2','Gripper closed behind cube','Operator-reported pickup misses; no calibrated Cartesian miss distance.'],
        ['3','Cube held above bowl; release on next preparation','Operator confirmation and saved timing review; not an independent full-task pass.'],
        ['4-6','Pickup achieved; bowl transport failed','One report described travel opposite the bowl. This motivated planned transport.'],
        ['9','Release outside bowl','Successful pickup/carry; full placement criterion failed.'],
        ['8,12,14-16','Failure, stage not specified','No invented pickup/transport outcome. Attempt 12 includes multiple launcher runs.'],
        ['19 aborted run 1','718 ms command gap; engine watchdog fired','500 ms watchdog stopped arm first; later carry-following error was secondary.'],
        ['19 aborted run 2','736 ms gap despite command-executor isolation','Same primary timeout; isolated executor alone was insufficient.'],
        ['19 selected retest','Operator confirmed complete success','After queued diagnostics and handoff-lock changes; one successful retest does not prove all scheduling risks removed.'],
    ],[25,61,92])
    para('The fault mitigations retained the 0.5-second watchdog and existing feedback/target checks. Command publication was isolated from inference and service callbacks; synchronous handoff logging was then removed from the command path. Read-only scheduling and regression checks are software evidence, not certified real-time guarantees. An unrecorded repeat of attempt 13 and calibration exercises are excluded.')
    para('Warning/error counts in the workbook include startup waiting messages and logs from multiple nodes. They are not counts of collisions or failed grasps. The MoveIt reports listed no world objects or attached cube; registered-scene checks do not prove physical fixture clearance.')
    page();title('Telemetry overview')
    table(['No.','Capture s','TCP path m','Max joint tracking rad','Max target jump rad','Warn/error msgs'],[[d['a'],f"{d['s']['duration_s']:.1f}",f"{d['s'].get('tcp_path_length_m',0):.3f}",f"{max(d['s'].get('max_command_measured_gap_rad') or [0]):.5f}",f"{d['s'].get('max_consecutive_policy_jump_rad',0):.5f}",d['s'].get('warnings_errors')] for d in data],[12,29,30,40,38,29])
    maxgap=max(max(d['s'].get('max_command_measured_gap_rad') or [0]) for d in data)
    para(f'The largest stored selected-run command-versus-measured joint gap was {maxgap:.5f} rad. These joint-space errors quantify controller tracking; they are not cube-centre errors, grasp accuracy or contact-force measurements. Good tracking can coexist with a wrong visual target or a missed grasp.')
    para('TCP path length is accumulated from received pose samples and includes noise and repeated motion. Source camera age uses image header clocks and cannot be equated directly with executor input freshness. All stored summary fields, per-joint maxima, received rates, message counts and gripper/IO events are included in the workbook. Full per-message telemetry remains in the linked JSONL sources.')
    page();title('Real camera evidence: selected attempt 19')
    para('Base and wrist frames below are taken directly from the saved trial. Columns are camera views, not a claim of hardware synchronization. Times are relative to the capture start. Still images near closure or release do not independently prove contact or landing in the bowl.',style='SmallBody')
    manifest=data[18]['m'];folder=data[18]['p']/'report_frames'
    for step,label in [(3,'Pickup region / near close'),(6,'Cube carried above the table'),(9,'Cube over bowl / near release command')]:
        panels=[next(p for p in manifest['panels'] if p['camera']==c and p['step']==step) for c in ['base','wrist']]
        imagecells=[Image(str(folder/p['file']),width=80*mm,height=60*mm) for p in panels]
        captions=[Paragraph(escape(f"{p['camera'].capitalize()} | t={p['capture_time_s']:.3f} s"),styles['SmallBody']) for p in panels]
        story.append(PDFTable([imagecells,captions],colWidths=[89*mm,89*mm]));story.append(Paragraph(escape(label),styles['SmallBody']));story.append(Spacer(1,3*mm))
        docx.add_heading(label,2)
        for panel in panels:
            docx.add_picture(str(folder/panel['file']),width=Inches(3.1));docx.add_paragraph(f"{panel['camera']} | {panel['capture_time_s']:.3f} s")
    page();title('Scene audit, limitations and interpretation')
    para(color_note)
    para('Attempt 13 also contains a reviewed sample with a human hand near the object. That frame alone does not establish whether help occurred before or during the autonomous portion, but it warrants intervention review before making a no-human-assistance claim. The report preserves the operator outcome and marks this as a scene-audit caveat rather than inventing a corrected result.')
    table(['Evidence limit','Implication'],[
        ['Mixed execution revisions','Do not attribute completion changes solely to the frozen model, pickup shift or one software fix.'],
        ['Scene/object changes','Command text is not proof of which coloured object was manipulated.'],
        ['Unknown stages in five failures','Pickup count is a lower bound: 13 confirmed, 2 explicit failures and 5 unknown.'],
        ['Unverified task times','Do not infer mean execution speed, deadline success or improvement from capture duration.'],
        ['Retakes excluded explicitly','Selected-trial completion is 8/20. This is not a success rate over every launch, calibration move or informal run.'],
        ['Uncalibrated object positions / contact','No millimetre grasp accuracy, force, robustness over physical crosses or visual localization accuracy is established.'],
    ],[55,123])
    para('The strongest supported finding is that the frozen policy achieved reported pickups at some placements, while reliable full completion often depended on planned transport and calibrated release logic. The original six policy-carry attempts had no complete passes. The selected later trials show assisted completion, but scene changes, unknown stages and nonrandomized software revisions limit conclusions about unseen-task generalization.')
    title('Workbook and source provenance',2)
    para('Workbook tabs: Summary, Attempts, Telemetry, All_summary_fields, Events, Conditions, Excluded_runs, Evidence, Camera_frames and Definitions. Evidence paths link to original local files. All_summary_fields preserves every scalar leaf of the stored summaries, including nulls and indexed arrays. Camera_frames lists 400 real exported candidate images. This report does not modify source trial outcomes or raw observations.')
    para('Session source: '+str(root))
    para('Prepared '+datetime.now(ZoneInfo('America/Toronto')).strftime('%Y-%m-%d %H:%M %Z')+'. No synthetic telemetry included.')
    def footer(canvas,document):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#506070'));canvas.drawString(16*mm,10*mm,'UR5 | 20 selected attempts | operator-reported outcomes');canvas.drawRightString(194*mm,10*mm,str(document.page))
    pdf=out/'UR5_20_Attempts_Report.pdf';SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=15*mm,bottomMargin=18*mm,title='UR5 20-attempt evaluation report',author='UR5 experiment analysis').build(story,onFirstPage=footer,onLaterPages=footer)
    docx.save(out/'UR5_20_Attempts_Report.docx')
    # Round-trip audit verifies source counts and workbook preservation.
    check=load_workbook(workbook_path,data_only=False)
    assert check['Attempts'].max_row==21 and check['Camera_frames'].max_row==401
    assert sum(check['Attempts'].cell(row,3).value=='Complete success' for row in range(2,22))==8
    assert check['All_summary_fields'].max_row==len(full)+1
    validation={'selected_trials':20,'report_categories':dict(counts),'workbook_roundtrip_passed':True,'summary_scalar_leaves':len(full),'camera_frames':len(figrows),'evidence_files':len(evidence),'source_summary_sha256':{str(d['a']):sha(d['p']/'summary.json') for d in data},'scene_colour_confirmation_received':confirmation is not None,'source_data_modified':False}
    (out/'validation_manifest.json').write_text(json.dumps(validation,indent=2)+'\n')
    archive=out.parent/'UR5_20_Attempts_Report_Package.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.is_file():z.write(f,str(Path(out.name)/f.relative_to(out)))
    print(json.dumps({'pdf':str(pdf),'xlsx':str(workbook_path),'docx':str(out/'UR5_20_Attempts_Report.docx'),'zip':str(archive),'validation':validation},indent=2))


if __name__=='__main__':main()
