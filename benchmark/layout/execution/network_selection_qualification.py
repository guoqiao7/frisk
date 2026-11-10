"""New network epoch using explicit providers of the accepted workflow engine.

No copied numerical/safety/timing loop and no production cache identity bypass.
The context binds only this command to its fixed suite and artifact root.
"""
import argparse
import time
from contextlib import contextmanager
from pathlib import Path
import partial_cost_qualification as engine
import gf2_decision_fixture as preceding
import network_selection_fixture as suite
import resource_decision
from partial_cost_selection import cost_interval

ROOT=engine.ROOT
BASE=ROOT/'guoqiao/optimize_plan/network_selection_execution_artifacts'


@contextmanager
def _epoch():
    saved=(engine.suite,engine.BASE,engine.SEALED)
    try:
        engine.suite=suite;engine.BASE=BASE
        engine.SEALED=saved[2]+(
            ('gf2_decision_refinement_artifacts','d9ab1a589446a9fbb1237d4dc47a4948eab2c06b56be92712b5fd6b87ff21774'),
            ('gf2_shuffle_network_artifacts','4fc515af30851a9cb8001be4416340a0e384944ca4d37145492e5b12448954d1'))
        yield
    finally:engine.suite,engine.BASE,engine.SEALED=saved


def decisions(pool,analyses,old,new,evidence):
    result=engine.calculate_decisions(pool,analyses,old,new,evidence,registered_labels=preceding.POLICY_LABELS)
    for mode in ('joint','local'):
        info=pool['modes'][mode];arms=info['arms'];original=info['fallback']
        intervals={a:cost_interval(analyses[a],new) for a in arms}
        ev={a:evidence[a] for a in arms};order=[original]+sorted(a for a in arms if a!=original)
        result[mode+'-resource-frontier']=resource_decision.select(intervals,ev,new,info['static_order'],order)
    if set(result)!=set(suite.POLICY_LABELS):raise ValueError('complete new decision ablation matrix required')
    return result



def validate_compilation_execution(configuration):
    """Optional preregistered scheduling only; missing field is historical serial."""
    if not isinstance(configuration,dict):raise ValueError('configuration object required')
    if 'compilation_execution' not in configuration:return None
    value=configuration['compilation_execution']
    expected=dict(contract='fixed-ttgir-four-worker-batch-v1',mode='parallel',workers=4,
        cpus=[8,9,10,11],threads_per_worker=1,
        partition='exact-generated-ttgir-sha256-modulo-workers-v1',compiler_input_reuse=True)
    if (not isinstance(value,dict) or value!=expected or set(value)!=set(expected)
        or type(value.get('workers')) is not int or type(value.get('threads_per_worker')) is not int
        or not isinstance(value.get('cpus'),list) or any(type(c) is not int for c in value['cpus'])
        or value.get('compiler_input_reuse') is not True):
        raise ValueError('registered fixed four-worker compilation contract differs')
    return dict(value,cpus=list(value['cpus']))


def init(run,configuration):
    # The strict profile loader validates known-only re-extraction/refitting.
    # The historical command still enforces its original LOCAL6418 contract.
    from predicate_cost_recalibration import validate_configuration
    cfg=engine.read(configuration);validate_compilation_execution(cfg);validate_configuration(cfg)
    with _epoch():return engine.init(run,configuration,expected_local_weight=None)


def archive_check(record,archive,mode,capacity):
    from network_archive import check_archive
    names={'joint':'J','local':'LI'}
    if mode not in names:raise ValueError('explicit workflow joint/local mode required')
    return check_archive(record,archive,names[mode],capacity)


def generate(run):
    from network_archive import replay_args,search_args,check_replay
    args=[a for a in search_args() if not a.startswith('--candidate-archive=')]
    with _epoch():return engine.generate(run,archive_validator=archive_check,
        archive_arguments=args,replay_argument_provider=replay_args,replay_validator=check_replay)


def compile_batch(run,cache_dir=None):
    run=Path(run)
    with _epoch():
        # Dispatch-only read: the unchanged serial engine performs its own full
        # validation. The optional branch independently verifies registration.
        scheduling=validate_compilation_execution(engine.read(run/'protocol.json').get('configuration',{}))
        if scheduling is None:return engine.compile_batch(run,cache_dir=cache_dir)
        protocol=engine.verify(run)
        if validate_compilation_execution(protocol['configuration'])!=scheduling:
            raise ValueError('compilation registration changed during dispatch')
        manifest=engine.read(run/'manifest.json');registration=engine.read(run/'registration.json')
        if manifest['status']!='generated' or manifest.get('compiled'):
            raise ValueError('fresh generated matrix required')
        observation=run/'parallel-native-compilation'
        if (run/'batch-compilation.json').exists() or observation.exists():
            raise FileExistsError('fresh parallel batch refuses prior attempt overwrite')
        from parallel_compile_batch import compile_many_parallel
        start=time.perf_counter_ns()
        try:
            result=compile_many_parallel([Path(r['path']) for r in manifest['rows']],
                run/'native-artifact-cache' if cache_dir is None else cache_dir,
                output=observation,execution_modules=Path(__file__).resolve().parent,
                cpus=tuple(scheduling['cpus']),compiler_input_reuse=True,
                source=registration['source'],tools=registration['tools'])
        except Exception as error:
            engine.write(run/'batch-compilation.json',dict(status='failed',wall_ns=time.perf_counter_ns()-start,
                compilation_execution=scheduling,error=type(error).__name__+': '+str(error)))
            raise
        n=len(manifest['rows'])
        valid=(isinstance(result,dict) and result.get('contract')==scheduling['contract']
            and result.get('affinity')==scheduling['cpus']
            and all(type(result.get(k)) is int for k in ('request_cap','attempted_requests','succeeded_requests','failed_requests','not_attempted_requests'))
            and result.get('status')=='complete' and result.get('provider_mode')=='production'
            and result.get('request_cap')==n and result.get('attempted_requests')==n
            and result.get('succeeded_requests')==n and result.get('failed_requests')==0
            and result.get('not_attempted_requests')==0 and result.get('descendants_reaped') is True
            and type(result.get('compiler_api_calls')) is int and type(result.get('group_count')) is int
            and 0<=result['compiler_api_calls']<=result['group_count']<=n)
        engine.write(run/'batch-compilation.json',dict(status='complete' if valid else 'failed',
            wall_ns=time.perf_counter_ns()-start,result=result,compilation_execution=scheduling))
        if not valid:raise ValueError('parallel compilation incomplete or request/API budget differs')
        manifest['compiled']=True;manifest['batch_compilation_sha256']=engine.sha(run/'batch-compilation.json')
        engine.write(run/'manifest.json',manifest);engine.verify(run)
        return result


def execute(directory,load_only=False,safety=False,safety_tool=None):
    with _epoch():return engine.execute(directory,load_only,safety,safety_tool)


def analyze(run):
    with _epoch():return engine.analyze(run,evidence_provider=resource_decision.candidate_evidence,decision_provider=decisions)


def measurement_groups(run):
    from network_archive import check_archive
    with _epoch():return engine.measurement_groups(run,evidence_verifier=resource_decision.verify_candidate_evidence,
        decision_provider=decisions,archive_validator=archive_check)


def assess(run):
    from network_selection_analysis import decompose,ranking
    with _epoch():result=engine.assess(run,group_provider=measurement_groups)
    groups=engine.read(run/'group-quality.json');pools=engine.read(run/'candidate-pools.json')
    aliases=engine.read(run/'measurement-aliases.json')['groups'];choices=engine.read(run/'partial-decisions.json')['groups']
    m=engine.read(run/'manifest.json');details={}
    for key,g in groups.items():
        pool=pools[key];rows=[r for r in m['rows'] if r['spec']==pool['spec']]
        raw=engine.read(run/'measurement'/(key+'-raw.json'))
        details[key]=dict(spec=pool['spec'],decomposition=decompose(g,pool,aliases[key],choices[key]['choices']),
            within_pool_ranking=ranking(raw,rows,pool,aliases[key],choices[key]['choices'],seed=suite.SEED))
    engine.write(run/'pool-ranking-quality.json',details)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','generate','compile','validate','execute','analyze','measure','assess','verify-groups'])
    p.add_argument('directory',type=Path);p.add_argument('--configuration',type=Path);p.add_argument('--cache-dir',type=Path)
    p.add_argument('--load-only',action='store_true');p.add_argument('--safety',action='store_true');p.add_argument('--safety-tool')
    a=p.parse_args();d=a.directory.resolve()
    if a.command=='init':init(d,a.configuration)
    elif a.command=='generate':generate(d)
    elif a.command=='compile':compile_batch(d,a.cache_dir)
    elif a.command=='validate':
        with _epoch():engine.quality.validate(d,worker_script=__file__)
    elif a.command=='execute':
        try:execute(d,a.load_only,a.safety,a.safety_tool)
        except Exception as e:
            engine.write(d/'worker-failure.json',dict(stage=getattr(e,'stage','worker'),code=getattr(e,'code','exception'),reason=type(e).__name__+': '+str(e)));raise
    elif a.command=='analyze':analyze(d)
    elif a.command=='measure':
        with _epoch():engine.quality.measure(d,d/'measurement',suite=suite,groups=measurement_groups(d))
    elif a.command=='verify-groups':measurement_groups(d)
    else:assess(d)

if __name__=='__main__':main()
