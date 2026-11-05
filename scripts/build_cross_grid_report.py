#!/usr/bin/python3
"""Build an evidence-labelled evaluation report; never write predicted robot results."""
import argparse,base64,collections,copy,csv,datetime,hashlib,html,json,math,random,statistics,sys
from pathlib import Path
try:
    import orjson
    load_line = orjson.loads
except ImportError:
    load_line = json.loads


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def numeric(value,prefix=''):
    if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
        yield prefix,float(value)
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from numeric(v,f'{prefix}[{i}]')
    elif isinstance(value,dict):
        for k,v in value.items():
            if k not in ('stamp_s','source_stamp_s'):yield from numeric(v,f'{prefix}.{k}' if prefix else k)


def read_telemetry(folder):
    path=folder/'observations.jsonl'
    if not path.exists():return {'available':False,'limitation':'No recording; operator report only.'}
    counts=collections.Counter();stats={};events=[];first={};last={};max_gap={};tcp=[];errors=[];sha=hashlib.sha256();bad=0
    selected={'/joint_states','/control_abstraction_layer','/forward_position_controller/commands',
              '/control_engine/gripper_state','/speed_scaling_state_broadcaster/speed_scaling','tcp_pose'}
    with path.open('rb') as stream:
        for raw in stream:
            sha.update(raw)
            try:d=load_line(raw)
            except (ValueError,UnicodeDecodeError):bad+=1;continue
            topic=d['topic'];t=d['t_s'];v=d['value'];counts[topic]+=1
            if topic in last:max_gap[topic]=max(max_gap.get(topic,0.),t-last[topic])
            first.setdefault(topic,t);last[topic]=t
            if topic in selected:
                for key,value in numeric(v):
                    # Comparison targets can remain cached after executor stop.
                    if topic=='tcp_pose' and key.startswith(('policy_fk','command_fk')):continue
                    k=topic+' :: '+key
                    s=stats.setdefault(k,[0,0.,0.,value,value]);s[0]+=1;s[1]+=value;s[2]+=value*value;s[3]=min(s[3],value);s[4]=max(s[4],value)
            if topic=='tcp_pose':
                if not tcp:tcp.append({'t_s':t,**v})
                if len(tcp)==1:tcp.append({'t_s':t,**v})
                else:tcp[-1]={'t_s':t,**v}
            if topic=='/rosout':
                message=v.get('message','')
                kind=next((k for k,text in [('automatic_release','BOWL RELEASE:'),('carry_lift','CARRY LIFT:'),('gripper_io','Gripper changed to'),('episode_reset','Software execution state reset'),('policy_start','CONTINUOUS OFFSET ENABLED:')] if text in message),None)
                if kind:events.append({'kind':kind,'t_s':t,**v})
                if v.get('level',0)>=40:errors.append({'t_s':t,**v})
    scalars={k:{'count':s[0],'mean':s[1]/s[0],'rms':math.sqrt(s[2]/s[0]),'minimum':s[3],'maximum':s[4]} for k,s in stats.items()}
    return {'available':True,'source_jsonl':str(path),'bytes':path.stat().st_size,'sha256':sha.hexdigest(),'malformed_lines':bad,
            'topic_message_counts':dict(counts),'first_received_s':first,'last_received_s':last,'maximum_receive_gap_s':max_gap,
            'scalar_statistics':scalars,'first_tcp':tcp[0] if tcp else None,'last_tcp':tcp[-1] if tcp else None,
            'events':events,'errors':errors,'limitations':['Receive gaps include time after a publisher is stopped or restarted.','Joint vector indices retain each message name order; names are in raw JSONL.','TCP statistics are not cube localization or grasp error.','Cached FK comparison fields are excluded from statistics; source-age checks are required to use them.']}


def beta_cdf(x,a,b):
    n=a+b-1
    if x<=0:return 0.
    if x>=1:return 1.
    return sum(math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)+k*math.log(x)+(n-k)*math.log1p(-x)) for k in range(a,n+1))


def beta_quantile(q,a,b):
    lo,hi=0.,1.
    for _ in range(65):
        mid=(lo+hi)/2
        if beta_cdf(mid,a,b)<q:lo=mid
        else:hi=mid
    return (lo+hi)/2


def predictive(n,a,b):
    lb=lambda x,y:math.lgamma(x)+math.lgamma(y)-math.lgamma(x+y)
    pmf=[math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)+lb(a+k,b+n-k)-lb(a,b)) for k in range(n+1)]
    def quantile(q):
        total=0.
        for k,p in enumerate(pmf):
            total+=p
            if total>=q:return k
        return n
    return {'n':n,'expected_successes':n*a/(a+b),'predictive_95_interval_successes':[quantile(.025),quantile(.975)],'pmf_success_count':pmf}


def nfmt(value,digits=1):return 'unknown' if value is None else f'{value:.{digits}f}'


def add_telemetry_analysis(report):
    """Summarize capture-level metrics without claiming task accuracy or causality."""
    metrics = collections.defaultdict(list)
    error_categories = collections.Counter()
    for attempt in report['observed_attempts']:
        summary = attempt['summary']
        if not attempt['derived_telemetry']['available']:
            continue
        for key in ('duration_s', 'tcp_path_length_m', 'max_consecutive_policy_jump_rad', 'invalid_action_count', 'warnings_errors'):
            value = summary.get(key)
            if isinstance(value, (int, float)):
                metrics[key].append(value)
        for key in ('max_policy_measured_gap_rad', 'max_command_measured_gap_rad'):
            values = summary.get(key)
            if values:
                metrics[key + '_largest_joint'].append(max(values))
        metrics['gripper_prediction_transition_count'].append(len(summary.get('gripper_prediction_transitions', [])))
        messages = [e.get('message', '').lower() for e in attempt['derived_telemetry'].get('errors', [])]
        for category, needles in {
            'stale_input_or_feedback': ('stale',),
            'policy_timeout': ('policy command timed out',),
            'publisher_disconnect_or_competition': ('publisher disconnected', 'competing publisher'),
            'trajectory_step_envelope': ('target-step envelope',),
            'planning_service_or_timeout': ('planning timed out', 'planning service unavailable'),
        }.items():
            if any(any(needle in message for needle in needles) for message in messages):
                error_categories[category] += 1
    report['analysis']['capture_metric_distributions'] = {
        key: {'n': len(values), 'minimum': min(values), 'median': statistics.median(values),
              'mean': statistics.mean(values), 'maximum': max(values)}
        for key, values in metrics.items() if values
    }
    report['analysis']['captures_with_logged_error_categories'] = dict(error_categories)
    report['analysis']['metric_caveats'] = [
        'Metrics cover whole captures, including waiting, stopped publishers and restarts; no active-motion segmentation is claimed.',
        'Policy target jumps can include episode boundaries. Joint tracking gaps are not Cartesian object error.',
        'Error-category counts overlap and include stop/disconnect aftermath; they are not counts of failed grasps or proven causes.',
        'A change in predicted gripper requests is not necessarily a physical open/close: engine arbitration and IO acknowledgments are separate.',
    ]
    reference = Path(report['observed_attempts'][0]['folder']).parents[1] / 'reference'
    report['cross_location_reference'] = json.loads((reference / 'cross_locations.json').read_text())
    report['numbered_reference_image'] = str(reference / 'numbered_reference.png')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    root=args.session.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((root/'results.csv').open()));observed=[];pending=[]
    from concurrent.futures import ProcessPoolExecutor
    folders=[root/f"cross_{int(row['cross']):02}/attempt_{int(row['attempt']):02}" for row in rows if row['status']=='finished']
    print('Reading recorded telemetry with eight independent workers.',flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        telemetry_results=dict(zip(folders,pool.map(read_telemetry,folders)))
    for row in rows:
        if row['status']!='finished':pending.append(row);continue
        cross,attempt=int(row['cross']),int(row['attempt']);folder=root/f'cross_{cross:02}/attempt_{attempt:02}'
        summary=json.loads((folder/'summary.json').read_text());revpath=folder/'configuration/evaluation_revision.json'
        rev=json.loads(revpath.read_text()) if revpath.exists() else {'revision':'baseline / unspecified'}
        runtime={}
        for which in ('start','end'):
            p=folder/f'configuration/execution_{which}.json'
            if p.exists():runtime[which]=json.loads(p.read_text())
        print(f'Scanning cross {cross:02} attempt {attempt}: {len(observed)+1}/45',flush=True)
        telemetry=telemetry_results[folder]
        events=telemetry.get('events',[]);releases=[e['t_s'] for e in events if e['kind']=='automatic_release'];starts=[e for e in events if e['kind']=='policy_start']
        files=[p for p in folder.rglob('*') if p.is_file()]
        observed.append({'cross':cross,'attempt':attempt,'record_type':'OBSERVED_OPERATOR_RESULT','actual_outcome':row['operator_outcome'],
            'summary':summary,'configuration':rev,'runtime_snapshots':runtime,'derived_telemetry':telemetry,
            'release_events_s':releases,'restart_or_multiple_run_evidence':len(starts)>1 or (len(starts)>=1 and starts[0]['t_s']>20),
            'policy_start_events_in_capture':len(starts),'folder':str(folder),'assets':{'file_count':len(files),'bytes':sum(p.stat().st_size for p in files),
                'video_files':[str(p) for p in (folder/'video').glob('*')],'snapshot_count':len(list((folder/'camera_snapshots').glob('*'))),
                'depth_file_count':len(list((folder/'depth').glob('*')))}})
    if len(observed)!=45 or len(pending)!=55:raise ValueError(f'Expected 45 completed / 55 pending; found {len(observed)} / {len(pending)}')
    passed=sum(x['actual_outcome']=='pass' for x in observed);a,b=passed+1,len(observed)-passed+1
    recent=[x for x in observed if x['configuration']['revision']=='revision_16_isolated_observations']
    rp=sum(x['actual_outcome']=='pass' for x in recent)
    model={'method':'Exchangeable Beta-Binomial scenario forecast; uniform Beta(1,1) prior, reported success/failure likelihood.',
        'observed_passes':passed,'observed_attempts':len(observed),'posterior_alpha':a,'posterior_beta':b,
        'success_probability':a/(a+b),'probability_95_interval':[beta_quantile(.025,a,b),beta_quantile(.975,a,b)],
        'remaining_55':predictive(55,a,b),'per_unseen_cross_5':predictive(5,a,b),
        'current_revision_sensitivity':{'observed_n':len(recent),'reported_successes':rp,'posterior_mean':(rp+1)/(len(recent)+2),'remaining_55':predictive(55,rp+1,len(recent)-rp+1)},
        'limitations':['Descriptive scenario forecast, not a validated model of unseen robot positions.','All observed trials were selected/adjudicated by the operator; excluded retries and code changes bias rates.','Assumes exchangeability despite spatial and configuration differences; intervals omit that structural uncertainty.','Does not predict measured trajectories, collision locations, force or exact completion times.','Success definition here is operator-reported task completion; a separate 300-second result must not be conflated with it.']}
    rng=random.Random(20261016);donors=list(range(45))+[rng.randrange(45) for _ in range(10)];rng.shuffle(donors)
    simulated=[]
    for row,idx in zip(pending,donors):
        donor=observed[idx]
        simulated.append({'cross':int(row['cross']),'attempt':int(row['attempt']),'record_type':'PREDICTED_UNPERFORMED_SYNTHETIC_REPLAY',
            'actual_outcome':None,'actual_telemetry_available':False,'predicted_success_probability':a/(a+b),
            'probability_95_interval':model['probability_95_interval'],'synthetic_scenario_outcome':donor['actual_outcome'],
            'analogue_source':{'cross':donor['cross'],'attempt':donor['attempt'],'recording_available':donor['derived_telemetry']['available']},
            'simulated_telemetry_summary':copy.deepcopy(donor['summary']),'simulated_derived_telemetry':copy.deepcopy(donor['derived_telemetry']),
            'source_runtime_snapshots':copy.deepcopy(donor['runtime_snapshots']),
            'warning':'Every copied field is past source telemetry used as a synthetic analogue, not a measurement or guaranteed forecast of this future attempt. Source coordinates, timings, logs and outcomes refer to the donor. Missing source telemetry stays missing.'})
    excluded=[]
    for folder in sorted(root.glob('cross_*/attempt_*')):
        if not folder.is_dir() or not any(t in folder.name for t in ('original','aborted')):continue
        p=folder/'summary.json'
        if p.exists():excluded.append({'folder':str(folder),'summary':json.loads(p.read_text())})
    audit_paths = set(root.glob('cross_*/*audit.json')) | set(root.glob('cross_*/attempt_*/outcome_corrections.json'))
    corrections=[{'file':str(p),'content':json.loads(p.read_text())} for p in sorted(audit_paths)]
    cross_table=[]
    for cross in range(1,21):
        group=[x for x in observed if x['cross']==cross]
        cross_table.append({'cross':cross,'observed_attempts':len(group),'reported_successes':sum(x['actual_outcome']=='pass' for x in group),
            'actual_success_rate':sum(x['actual_outcome']=='pass' for x in group)/len(group) if group else None,
            'forecast_attempts':5-len(group),'forecast_expected_successes':(5-len(group))*a/(a+b)})
    config_groups=collections.defaultdict(list)
    for x in observed:config_groups[x['configuration']['revision']].append(x)
    configurations=[{'revision':rev,'attempts':len(v),'reported_successes':sum(x['actual_outcome']=='pass' for x in v)} for rev,v in config_groups.items()]
    review={'observed_n':45,'reported_successes':passed,'reported_failures':45-passed,'reported_success_rate':passed/45,
        'recorded_attempts':sum(x['derived_telemetry']['available'] for x in observed),'unrecorded_attempts':[[x['cross'],x['attempt']] for x in observed if not x['derived_telemetry']['available']],
        'capture_duration_is_not_completion_time':True,'automatic_release_count':sum(bool(x['release_events_s']) for x in observed),
        'capture_with_restart_evidence_count':sum(x['restart_or_multiple_run_evidence'] for x in observed),
        'reported_passes_with_capture_below_300s':sum(x['actual_outcome']=='pass' and x['summary'].get('duration_s') is not None and x['summary']['duration_s']<300 for x in observed),
        'reported_passes_with_first_logged_release_after_300s':sum(x['actual_outcome']=='pass' and bool(x['release_events_s']) and min(x['release_events_s'])>300 for x in observed),
        'total_raw_jsonl_bytes':sum(x['derived_telemetry'].get('bytes',0) for x in observed),
        'excluded_recordings':len(excluded),'configuration_groups':configurations}
    report={'title':'UR5 evaluation: 45 observed attempts and 55 forecast scenarios',
        'generated_client_time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4))).isoformat(),
        'timezone':'America/Toronto','session':json.loads((root/'session.json').read_text()),'analysis':review,'crosses':cross_table,'forecast_model':model,
        'synthetic_scenario_policy':'Every one of the 45 observed profiles is used at least once across the 55 synthetic scenarios; 10 additional donor profiles are selected with seed 20261016. This is a coverage-balanced replay set, not 55 independent probabilistic draws or actual test results.',
        'observed_attempts':observed,'predicted_attempts':simulated,'excluded_runs':excluded,'retest_audits':corrections,
        'telemetry_scope':'Full existing summary fields and derived numeric telemetry statistics are embedded. Approximately 4 GB of native timestamped raw JSONL and camera/depth assets remain in original folders and are linked, not embedded in the report. No raw telemetry exists for unperformed attempts; their replay references point to historical donor traces.'}
    add_telemetry_analysis(report)
    dump(out/'all_100_attempts_data.json',report)
    flat=[]
    for x in observed:
        s=x['summary'];flat.append({'cross':x['cross'],'attempt':x['attempt'],'type':'OBSERVED','actual_outcome':x['actual_outcome'],'probability_success':'','scenario_outcome':'','donor_cross':'','donor_attempt':'','capture_or_scenario_seconds':s.get('duration_s'),'recording_available':x['derived_telemetry']['available'],'release_seconds':';'.join(map(str,x['release_events_s']))})
    for x in simulated:
        flat.append({'cross':x['cross'],'attempt':x['attempt'],'type':'FORECAST_SYNTHETIC','actual_outcome':'','probability_success':x['predicted_success_probability'],'scenario_outcome':x['synthetic_scenario_outcome'],'donor_cross':x['analogue_source']['cross'],'donor_attempt':x['analogue_source']['attempt'],'capture_or_scenario_seconds':x['simulated_telemetry_summary'].get('duration_s'),'recording_available':False,'release_seconds':''})
    flat.sort(key=lambda x:(x['cross'],x['attempt']))
    with (out/'all_100_attempts.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    make_html(report,out,root,flat);make_pdf(report,out,flat)
    print(json.dumps({'output':str(out),'analysis':review,'forecast_expected':model['remaining_55']['expected_successes'],'forecast_interval':model['remaining_55']['predictive_95_interval_successes']},indent=2),flush=True)


def make_html(report,out,root,flat):
    esc=html.escape
    review=report['analysis'];model=report['forecast_model']
    def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+esc(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
    def link(path,label):
        import os
        return '<a href="'+esc(os.path.relpath(path,out),quote=True)+'">'+esc(label)+'</a>'
    content=['<h1>UR5 evaluation: 100 planned attempts</h1><p><b>45 observed results + 55 forecast scenarios. Forecast telemetry is synthetic historical replay, never measured future data.</b></p>',
        '<p>Generated '+esc(report['generated_client_time'])+' (America/Toronto).</p>',
        f'<div class="cards"><div><b>{review["reported_successes"]}/45</b><br>reported successes ({review["reported_success_rate"]:.1%})</div><div><b>44 recorded + 1 unrecorded</b><br>operator report only for cross 5 / attempt 2</div><div><b>{model["remaining_55"]["expected_successes"]:.1f}/55 expected</b><br>scenario forecast, not observed results</div></div>',
        '<h2>Evidence and limits</h2><p>'+esc(report['telemetry_scope'])+'</p>',
        '<p>Retest exclusions and software revisions changed during evaluation. These are selected operational results, not an untouched fixed-policy benchmark. Joint tracking error is not cube-position error; IO acknowledgments are not force/contact sensing. A release log proves an OPEN request, not successful placement.</p>',
        '<p>Capture duration ends at the operator report and may include waiting, restarts and more than one execution. Do not label it task time. The 300-second limit is reported separately; unknown timing remains unknown.</p>',
        '<h2>Observed performance by cross</h2>',table(['Cross','Observed','Reported passes','Forecast remaining'],[(x['cross'],x['observed_attempts'],x['reported_successes'],x['forecast_attempts']) for x in report['crosses']]),
        '<h2>Configuration changes</h2>',table(['Revision','Observed attempts','Reported passes'],[(x['revision'],x['attempts'],x['reported_successes']) for x in review['configuration_groups']]),
        '<h2>Interpretation</h2><p>Crosses 3, 5 and 6 have 5/5 selected reported successes; cross 5 includes an unrecorded result. Cross 4 has 2/5 and experienced software faults. Cross 1 has 4/5, cross 2 has 3/5, cross 11 has 4/5, cross 12 has 4/5, and cross 13 has 3/5. Five trials per location are too few to establish reliable location-specific rates.</p>',
        f'<p>{review["capture_with_restart_evidence_count"]} captures contain evidence of a late executor start or restart. {review["excluded_recordings"]} archived runs/reports are retained outside the selected 45. {review["reported_passes_with_first_logged_release_after_300s"]} reported successes have their first logged automatic release after 300 seconds from capture start; that clock may include pre-run waiting/restarts.</p>',
        '<p>The logs identify several engineering failures: stale observation buffers, callback scheduling, target-step rejection during carry-lift activation, and an overly broad release region allowing a rim-side drop. Publisher-disconnect faults can follow stopping a run and cannot automatically be assigned as the failure cause. Successful pickup/placement claims remain operator adjudications unless video review verifies them.</p>',
        '<h2>Forecast methodology and uncertainty</h2>',
        f'<p>All-45 Beta(1,1) pooling yields Beta({model["posterior_alpha"]},{model["posterior_beta"]}), with mean success probability {model["success_probability"]:.1%} and a 95% parameter interval of {model["probability_95_interval"][0]:.1%}–{model["probability_95_interval"][1]:.1%}. The remaining 55 have expected {model["remaining_55"]["expected_successes"]:.1f} successes; the model-based 95% predictive interval is {model["remaining_55"]["predictive_95_interval_successes"]}. Structural location/configuration uncertainty is additional.</p>',
        f'<p>Current-revision sensitivity: {model["current_revision_sensitivity"]["reported_successes"]}/{model["current_revision_sensitivity"]["observed_n"]} reported successes, with smoothed mean {model["current_revision_sensitivity"]["posterior_mean"]:.1%}. This is a sensitivity comparison, not proof the revision caused improvement.</p>',
        '<p>'+esc(report['synthetic_scenario_policy'])+'</p><p>Each unseen attempt has the same pooled probability: there is insufficient calibrated spatial evidence to justify different precise probabilities. Its scenario label and every telemetry field are explicitly borrowed from a donor. The placement photo supplies cross IDs, not calibrated robot XYZ.</p>',
        '<h2>All 100 attempt register</h2><p>Observed outcomes and synthetic scenario outcomes occupy separate columns. Forecast capture seconds are donor replay durations, not predicted task-completion times.</p>',
        table(['Cross','Attempt','Evidence type','Actual','P(success)','Synthetic scenario','Donor','Capture / replay seconds'],[(x['cross'],x['attempt'],x['type'],x['actual_outcome'],f"{x['probability_success']:.1%}" if x['probability_success']!='' else '',x['scenario_outcome'],f"{x['donor_cross']}/{x['donor_attempt']}" if x['donor_cross']!='' else '',nfmt(x['capture_or_scenario_seconds'])) for x in flat]),
        '<h2>Full telemetry profiles: 45 observed</h2>']
    content.insert(-1, '<h2>Capture telemetry distributions</h2>' + table(
        ['Metric', 'N', 'Minimum', 'Median', 'Maximum'],
        [(key, v['n'], nfmt(v['minimum'], 4), nfmt(v['median'], 4), nfmt(v['maximum'], 4))
         for key, v in review.get('capture_metric_distributions', {}).items()]) +
        '<p>' + esc(' '.join(review.get('metric_caveats', []))) + '</p><p>Captures with error categories (overlapping, not proven causes): ' +
        esc(json.dumps(review.get('captures_with_logged_error_categories', {}))) + '</p>')
    import os
    content.insert(4, '<h2>Cross location reference</h2><p>Photo labels identify locations; pixels are not calibrated robot coordinates.</p><img alt="Numbered cross locations" style="max-width:100%;max-height:800px" src="' + esc(os.path.relpath(report['numbered_reference_image'],out),quote=True) + '">')
    for x in report['observed_attempts']:
        p=Path(x['folder']);s=x['summary'];tele=x['derived_telemetry']
        content.append(f'<details><summary>OBSERVED · Cross {x["cross"]:02} / attempt {x["attempt"]:02} · {esc(x["actual_outcome"].upper())} · capture {nfmt(s.get("duration_s"))} s</summary>')
        content.append('<p>'+link(p/'summary.json','Original summary')+' · '+link(p,'All original files')+(' · '+link(p/'observations.jsonl','Complete native timestamped telemetry JSONL') if tele['available'] else ' · NO RECORDING')+'</p>')
        for video in x['assets']['video_files']:content.append('<p>'+link(Path(video),Path(video).name)+'</p>')
        content.append('<pre>'+esc(json.dumps(x,indent=2))+'</pre></details>')
    content.append('<h2>Full synthetic telemetry profiles: 55 unperformed attempts</h2><p>NOT MEASURED. Every nested summary, event, coordinate and runtime field refers to the historical donor. No actual outcome exists yet.</p>')
    for x in report['predicted_attempts']:
        donor=x['analogue_source'];p=x['simulated_derived_telemetry'].get('source_jsonl')
        content.append(f'<details class="forecast"><summary>FORECAST · Cross {x["cross"]:02} / attempt {x["attempt"]:02} · scenario {x["synthetic_scenario_outcome"].upper()} · donor {donor["cross"]}/{donor["attempt"]}</summary><p>{esc(x["warning"])}</p>')
        if p:content.append('<p>'+link(Path(p),'Historical donor raw telemetry — replay reference only')+'</p>')
        content.append('<pre>'+esc(json.dumps(x,indent=2))+'</pre></details>')
    content.append('<h2>Archived runs and adjudication audit</h2><pre>'+esc(json.dumps({'excluded_runs':report['excluded_runs'],'retest_audits':report['retest_audits']},indent=2))+'</pre>')
    content.append('<h2>Machine-readable data</h2><p>'+link(out/'all_100_attempts_data.json','All 100 labelled records, complete summaries, statistics and audit')+' · '+link(out/'all_100_attempts.csv','100-row register CSV')+'</p>')
    css='body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#182432}h1,h2{color:#173a60}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}th,td{border:1px solid #ccd5df;padding:8px;text-align:left}th{background:#edf2f8}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f4f6f9;padding:16px}.cards{display:flex;gap:16px}.cards div{background:#edf5f1;padding:20px;flex:1}.cards b{font-size:24px}details{border:1px solid #ccd5df;margin:12px 0;padding:12px}summary{cursor:pointer;font-weight:650}.forecast{border-left:6px solid #ca8b25}a{color:#135d92}@media print{details{break-inside:avoid}.cards{display:block}}'
    (out/'UR5_100_attempts_report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>UR5 100-attempt evidence and forecast report</title><style>'+css+'</style><body>'+''.join(content)+'</body></html>')


def make_pdf(report,out,flat):
    sys.path.insert(0,'/tmp/ur5_report_packages')
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_LEFT
    from pypdf import PdfReader,PdfWriter
    from xml.sax.saxutils import escape
    styles=getSampleStyleSheet();styles['BodyText'].fontSize=9;styles['BodyText'].leading=12
    styles['Code'].fontSize=6.5;styles['Code'].leading=8
    story=[]
    def para(text,style='BodyText'):story.append(Paragraph(escape(str(text)),styles[style]));story.append(Spacer(1,6))
    def table(headers,rows,widths=None):
        vals=[[Paragraph(escape(str(x)),styles['BodyText']) for x in row] for row in [headers,*rows]]
        t=Table(vals,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eff7')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#c9d3de')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t);story.append(Spacer(1,12))
    a=report['analysis'];m=report['forecast_model']
    para('UR5 evaluation: 100 planned attempts','Title');para('45 observed results + 55 synthetic forecast scenarios','Heading1')
    para('Generated '+report['generated_client_time']+' (America/Toronto).')
    para('FORECAST TELEMETRY IS SYNTHETIC HISTORICAL REPLAY. The remaining 55 attempts have not happened; no measured future telemetry or actual outcomes exist. Missing historical data remains missing.')
    para(f"Observed: {a['reported_successes']}/45 reported successes ({a['reported_success_rate']:.1%}); 44 recorded attempts and one unrecorded operator report (cross 5, attempt 2).")
    para(report['telemetry_scope'])
    para('This PDF embeds all_100_attempts_data.json and the 100-row CSV as file attachments. The JSON contains the full existing telemetry summaries, derived statistics, source runtime snapshots, synthetic donor profiles, and adjudication audit. Use a PDF reader supporting attachments, or open the companion HTML for expandable full-field profiles and direct raw-data links.')
    para('Evidence limitations','Heading1')
    for text in ['Five trials per location are a small sample. Retakes, exclusions, software revisions and manual adjudication prevent treating these as an unbiased fixed-configuration benchmark.','A release event is an OPEN request, not proof of successful placement. Closed IO is not force or object-contact sensing. Joint tracking error is not cube-position error.','Capture duration is not task-completion time: it may include pre-run waiting, restarts, multiple executions and delayed reporting. Completion within 300 seconds is unknown without verified timing.','The ~4 GB native raw JSONL and video/depth files remain in the original recording folders. They are linked/referenced, not embedded in this PDF. Historical raw traces attached to forecast profiles are donor replays, not future measurements.']:
        para(text)
    story.append(PageBreak());para('Numbered cross locations','Heading1')
    para('Photo labels identify test locations. Pixel positions are not calibrated base-frame coordinates; future donor trajectories do not establish a trajectory at another cross.')
    picture=Image(report['numbered_reference_image']);scale=min(480/picture.imageWidth,600/picture.imageHeight);picture.drawHeight=scale*picture.imageHeight;picture.drawWidth=scale*picture.imageWidth
    story.append(picture);story.append(PageBreak())
    para('Observed cross results','Heading1');table(['Cross','Reported successes','Data status'],[(x['cross'],f"{x['reported_successes']}/5",'5 selected outcomes; cross 5 includes unrecorded attempt 2' if x['cross']==5 else '5 selected outcomes') for x in report['crosses'] if x['observed_attempts']])
    para('Configuration groups','Heading1');table(['Revision','Observed N','Reported passes'],[(x['revision'],x['attempts'],x['reported_successes']) for x in a['configuration_groups']])
    para('Telemetry findings','Heading1')
    para(f"{a['capture_with_restart_evidence_count']} captures have evidence of a late start or restart. {a['excluded_recordings']} historical archived runs/reports are retained separately. {a['reported_passes_with_first_logged_release_after_300s']} successful reports have their first automatic release log after 300 seconds from capture start.")
    para('Recorded engineering issues included callback/buffer freshness faults, abrupt carry-height activation at a chunk boundary, repeated gripper requests, and a 50 mm bowl release region permitting a rim-side drop. Configuration changes are documented per attempt; these causes cannot be assigned to every failure. Most bare operator failure reports did not identify the failed task stage.')
    para('Capture telemetry distributions','Heading2')
    table(['Metric','N','Minimum','Median','Maximum'],
          [(key, v['n'], nfmt(v['minimum'],4), nfmt(v['median'],4), nfmt(v['maximum'],4))
           for key,v in a.get('capture_metric_distributions',{}).items()], [205,28,90,90,90])
    for text in a.get('metric_caveats',[]):para(text)
    para('Captures with error categories (overlapping; not proven failure causes): '+json.dumps(a.get('captures_with_logged_error_categories',{})))
    para('Forecast, not empirical results','Heading1')
    para(f"Pooled Beta({m['posterior_alpha']},{m['posterior_beta']}) posterior: expected success probability {m['success_probability']:.1%}; 95% probability interval {m['probability_95_interval'][0]:.1%}–{m['probability_95_interval'][1]:.1%}. Remaining 55: expected {m['remaining_55']['expected_successes']:.1f} successes; model-based 95% predictive interval {m['remaining_55']['predictive_95_interval_successes']}.")
    sens=m['current_revision_sensitivity'];para(f"Current-revision sensitivity uses {sens['reported_successes']}/{sens['observed_n']} reports and gives {sens['posterior_mean']:.1%} smoothed probability. This is not a causal comparison or a validated spatial predictor.")
    para(report['synthetic_scenario_policy'])
    for text in m['limitations']:para(text)
    story.append(PageBreak());para('All 100 attempt register','Heading1')
    table(['Cross / attempt','Evidence','Actual result','Synthetic scenario','Donor'],[(f"{x['cross']:02}/{x['attempt']:02}",x['type'],x['actual_outcome'] or 'NOT OBSERVED',x['scenario_outcome'] or '—',f"{x['donor_cross']}/{x['donor_attempt']}" if x['donor_cross']!='' else '—') for x in flat],[66,100,82,100,55])
    for x in report['observed_attempts']:
        story.append(PageBreak());s=x['summary'];t=x['derived_telemetry'];para(f"OBSERVED — cross {x['cross']:02}, attempt {x['attempt']:02}",'Heading1')
        para('Operator outcome: '+x['actual_outcome'].upper());para(s.get('notes',''))
        table(['Parameter','Recorded value'],[['Capture duration (not task time)',nfmt(s.get('duration_s'))+' s'],['Automatic release logs',', '.join(nfmt(v) for v in x['release_events_s']) or 'none captured'],['Telemetry available',t['available']],['Configuration',x['configuration']['revision']],['Maximum policy/measured gap (rad)',str(s.get('max_policy_measured_gap_rad','unknown'))],['Maximum command/measured gap (rad)',str(s.get('max_command_measured_gap_rad','unknown'))],['Tool path length (includes noise)',str(s.get('tcp_path_length_m','unknown'))],['Gripper request changes',len(s.get('gripper_prediction_transitions',[]))],['Warning/error count',s.get('warnings_errors','unknown')],['Video frames',str(s.get('video_frames','unknown'))]], [160,343])
        para('Topic rates (Hz): '+json.dumps(s.get('rates_hz',{})))
        para('Source folder: '+x['folder'])
        para('Full per-field telemetry summary, numerical statistics, IO transitions, event logs, file inventory, runtime settings and raw-trace hash are included in the attached JSON under this observed attempt. Raw file: '+t.get('source_jsonl','NO RECORDING'))
    for x in report['predicted_attempts']:
        story.append(PageBreak());d=x['analogue_source'];s=x['simulated_telemetry_summary'];t=x['simulated_derived_telemetry']
        para(f"SYNTHETIC FORECAST — cross {x['cross']:02}, attempt {x['attempt']:02}",'Heading1');para('NOT PERFORMED. NO MEASURED FUTURE TELEMETRY.');para(x['warning'])
        para(f"Probability of success: {x['predicted_success_probability']:.1%}. Replay scenario label: {x['synthetic_scenario_outcome'].upper()}. Donor: cross {d['cross']}, attempt {d['attempt']}.")
        table(['Synthetic replay metric','Historical donor value'],[['Capture/replay duration; NOT predicted task time',nfmt(s.get('duration_s'))+' s'],['Historical telemetry available',t['available']],['Maximum policy/measured joint gaps (rad)',str(s.get('max_policy_measured_gap_rad','unknown'))],['Maximum command/measured joint gaps (rad)',str(s.get('max_command_measured_gap_rad','unknown'))],['Historical tool path (m)',s.get('tcp_path_length_m','unknown')],['Historical topic rates',json.dumps(s.get('rates_hz',{}))],['Historical gripper request changes',len(s.get('gripper_prediction_transitions',[]))],['Historical warning/error count',s.get('warnings_errors','unknown')],['Historical video frames',str(s.get('video_frames','unknown'))]], [160,343])
        para('Every source field is preserved in the attached JSON, including full summary, derived statistics, logs/events and runtime snapshots. Missing source fields remain unknown. Historical full trace replay reference: '+t.get('source_jsonl','NO RECORDING AVAILABLE IN DONOR'))
    story.append(PageBreak());para('Retests and excluded history','Heading1')
    for x in report['excluded_runs']:para(x['folder']+' — '+str(x['summary'].get('operator_outcome')))
    para('The attached JSON preserves retest-audit contents. Original failures/aborted runs and unrecorded reports have not been deleted or presented as successful physical repetitions.')
    temp=out/'report_without_attachments.pdf';doc=SimpleDocTemplate(str(temp),pagesize=(595.28,841.89),rightMargin=46,leftMargin=46,topMargin=42,bottomMargin=42)
    def footer(c,d):c.setFont('Helvetica',8);c.drawString(46,24,'Observed data and synthetic forecasts are distinct');c.drawRightString(549,24,str(d.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    writer=PdfWriter();writer.append(PdfReader(str(temp)))
    for name in ('all_100_attempts_data.json','all_100_attempts.csv'):writer.add_attachment(name,(out/name).read_bytes())
    writer.add_metadata({'/Title':report['title'],'/Subject':'45 observed attempts and 55 explicitly synthetic forecast/replay scenarios','/Author':'UR5 evaluation analysis'})
    with (out/'UR5_100_attempts_final_report.pdf').open('wb') as f:writer.write(f)
    temp.unlink()


if __name__=='__main__':main()
